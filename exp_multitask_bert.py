import random, numpy as np, argparse
from types import SimpleNamespace
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from bert import BertModel
from lora_bert import inject_lora
import copy


from globals import BERT_HIDDEN_SIZE, N_SENTIMENT_CLASSES

class MultitaskBERT(nn.Module):
    '''
    This module should use BERT for 3 tasks:

    - Sentiment classification (predict_sentiment)
    - Paraphrase detection (predict_paraphrase)
    - Semantic Textual Similarity (predict_similarity)
    '''
    def __init__(self, config):
        super(MultitaskBERT, self).__init__()
        
        NUM_BERTS = config.num_berts
        assert NUM_BERTS >= 1, "Must have >=1 backbone."
        self.berts = nn.ModuleList([BertModel.from_pretrained('bert-base-uncased') for _ in range(NUM_BERTS)])
        # last-layer mode does not require updating BERT paramters.
        assert config.fine_tune_mode in ["last-layer", "full-model", "iterative"]
        
        # ==== lora ===== 
        assert config.lora_dict['mode'] in ['none', 'attn', 'attn-only', 'all-lin', 'all-lin-only']
        assert config.lora_dict['r'] > 0 or config.lora_dict['mode'] == 'none'
        assert config.lora_dict['dora'] in [0, 1]
        # LoRA settings
        if config.fine_tune_mode != 'full-model' and config.lora_dict['mode'] != 'none':
            raise ValueError("LoRA can only be used in full-model fine-tuning mode.")
        
        # Pretrain mode does not require updating BERT parameters.
        if config.fine_tune_mode == 'last-layer' or config.lora_dict['mode'] in ['attn-only', 'all-lin-only']:
            self.manage_freezing('freezeall')
        elif config.fine_tune_mode == 'full-model':  # iterative handles itself separately in multitask-classifier.py training
            self.manage_freezing('unfreezeall')
        
        if config.lora_dict['mode'] != 'none':
            for bert in self.berts:
                bert = inject_lora(bert, config.lora_dict['mode'], 
                                              config.lora_dict['r'], 
                                              config.lora_dict['dora'])

        assert config.clf in ["linear", "nonlinear", "conv"]
        self.clf = config.clf
        # You will want to add layers here to perform the downstream tasks.
        ### 
        
        EMBEDDINGS_SIZE = NUM_BERTS * BERT_HIDDEN_SIZE  # we concatenate NUM_BERTS embeddings # NOT RN-> : then project to lower dimension
        FEATURES_SIZE = 2 * EMBEDDINGS_SIZE + 1
        
        # ===== helper layers ======
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # learn linear mapping which can be shared among STS and Quora tasks for comparison features
        self.comparison_features_fcn = nn.Sequential(
            nn.Linear(FEATURES_SIZE, FEATURES_SIZE),
            nn.ReLU(),
            nn.Linear(FEATURES_SIZE, FEATURES_SIZE)
        )
        self.embeddings_linear = nn.Linear(EMBEDDINGS_SIZE*2, EMBEDDINGS_SIZE)
        self.embeddings_fcn = nn.Sequential(
            nn.Linear(EMBEDDINGS_SIZE*2, EMBEDDINGS_SIZE),
            nn.ReLU(),
            nn.Linear(EMBEDDINGS_SIZE, EMBEDDINGS_SIZE)
        )
        # ===== sentiment =====
        self.sentiment_linear = nn.Linear(EMBEDDINGS_SIZE, N_SENTIMENT_CLASSES)
        self.sentiment_nonlinear = nn.Sequential(nn.Linear(EMBEDDINGS_SIZE, EMBEDDINGS_SIZE//2),
                                        nn.ReLU(),
                                        nn.Linear(EMBEDDINGS_SIZE//2, N_SENTIMENT_CLASSES))
        
        self.sentiment_conv = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=4, kernel_size=3), # note: input needs to be unsqueezed
            nn.Flatten(),  # Flatten layer to convert (N, 4, D-2) to (N, 4*(D-2))
            nn.Linear(4 * (EMBEDDINGS_SIZE - 2), EMBEDDINGS_SIZE * 2),  
            nn.ReLU(),  
            nn.Linear(EMBEDDINGS_SIZE * 2, N_SENTIMENT_CLASSES)  
        )
        
            
        # ===== paraphrase ======
        self.paraphrase_linear = nn.Linear(FEATURES_SIZE, 1)
        self.paraphrase_nonlinear = nn.Sequential(nn.Linear(FEATURES_SIZE, FEATURES_SIZE//2),
                                        nn.ReLU(),
                                        nn.Linear(FEATURES_SIZE//2, 1))
        self.paraphrase_conv =  nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=4, kernel_size=3), # note: input needs to be unsqueezed
            nn.Flatten(),  # Flatten layer to convert (N, 4, D-2) to (N, 4*(D-2))
            nn.Linear(4 * (FEATURES_SIZE - 2), FEATURES_SIZE * 2),  
            nn.ReLU(),  
            nn.Linear(FEATURES_SIZE * 2, 1)  
        )
        

        # ===== similarity ========
        self.similarity_linear = nn.Linear(FEATURES_SIZE, 1)
        self.similarity_nonlinear = nn.Sequential(nn.Linear(FEATURES_SIZE, FEATURES_SIZE//2),
                                        nn.ReLU(),
                                        nn.Linear(FEATURES_SIZE//2, 1))
        self.similarity_conv = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=4, kernel_size=3), # note: input needs to be unsqueezed
            nn.Flatten(),  # Flatten layer to convert (N, 4, D-2) to (N, 4*(D-2))
            nn.Linear(4 * (FEATURES_SIZE - 2), FEATURES_SIZE * 2),  
            nn.ReLU(),  
            nn.Linear(FEATURES_SIZE * 2, 1)  
        )

        
    def manage_freezing(self, structure: str):
        """ 
        Manages freezing of BERT Attention layers
  
        Adjusts the trainable status of layers in a model based on a specified
        structure command. This function can freeze all layers, unfreeze all layers,
        or unfreeze only the top N layers of the model.

        Args:
            model (Module): The PyTorch model whose layer training settings are
                to be modified.
            structure (str): A command string that dictates how layers should be
                            frozen or unfrozen.It can be 'freezeall', 'unfreezeall',
                            or 'unfreezetopN' where N is an integer indicating the
                            number of top layers to unfreeze.

        Raises:
            ValueError: If the structure parameter does not follow the expected
                format or specifies an invalid option.
        """
        for bert in self.berts:
            children = list(bert.children())
            # The ModuleList containing the 12 self-attention layers is at index 5
            attn_idx = 5
            attention_layers = children[attn_idx]
            total_attn_layers = len(attention_layers)
            
            # print(f"\nTotal # of Model Children: {len(children)}")
            # print(f"Total Number of Attention Layers; {total_attn_layers}")
            # print("=============Children====================")
            # print(children)
            # print("=================================\n")
            # print("=============Attn Layers====================")
            # print(attention_layers)
            # print("=================================\n")

            if structure == "freezeall":
                # Freeze all layers
                for param in bert.parameters():
                    param.requires_grad = False

            elif structure == "unfreezeall":
                # Unfreeze all layers
                for param in bert.parameters():
                    param.requires_grad = True

            elif structure.startswith("unfreezetop"):
                # Attempt to extract the number of attn. layers to unfreeze from the structure string
                try:
                    n_layers = int(structure[len("unfreezetop") :])
                except ValueError:
                    raise ValueError(
                        (
                            "Invalid layer specification. Ensure it follows 'unfreezetopN' "
                            "format where N is a number."
                        )
                    )

                # Freeze all layers first
                for param in bert.parameters():
                    param.requires_grad = False
                    
                # Unfreeze children after the attention layers 
                for i in range(attn_idx+1, len(children)):
                    for param in children[i].parameters():
                        param.requires_grad = True
                # Unfreeze the last n attention_layers
                for i in range(total_attn_layers - n_layers, total_attn_layers):
                    # print("\nUnfreezing Layer:\n")
                    # print(children[i])
                    for param in attention_layers[i].parameters():
                        param.requires_grad = True
            else:
                raise ValueError(
                    (
                        "Invalid structure parameter. Use 'freezeall', 'unfreezeall', or "
                        "'unfreezetopN' where N is a number."
                    )
                )
        return None
        

    def forward(self, input_ids, attention_mask):
        """Takes a batch of sentences and produces embeddings for them.
        
        which_bert - indicate which bert backbone to update
        
        Shape returned = Batch size, bert hidden dimension (768)  (N, D)
        """
        # The final BERT embedding is the hidden state of [CLS] token (the first token)
        # Here, you can start by just returning the embeddings straight from BERT.
        # When thinking of improvements, you can later try modifying this
        # (e.g., by adding other layers).
        ### TODO
        embeddings = []
        for bert in self.berts:
            embeddings.append(bert(input_ids=input_ids, attention_mask=attention_mask)['pooler_output'])
        all_embeds = torch.concat(embeddings, dim = 1)
        # all_embeds = self.embeddings_linear(all_embeds)
        return all_embeds
    

    def predict_sentiment(self, input_ids, attention_mask):
        '''Given a batch of sentences, outputs logits for classifying sentiment.
        There are 5 sentiment classes:
        (0 - negative, 1- somewhat negative, 2- neutral, 3- somewhat positive, 4- positive)
        Thus, your output should contain 5 logits for each sentence.
        '''
        ### TODO
        embeddings = self.forward(input_ids, attention_mask)
        
        if self.clf == "linear":
            out = self.sentiment_linear(embeddings)
        elif self.clf == "nonlinear":
            out = self.sentiment_nonlinear(embeddings)
        else: # self.clf == "conv"
            out = self.sentiment_conv(torch.unsqueeze(embeddings, 1))
        return out
    
    def extract_comparison_features(self,
                           input_ids_1, attention_mask_1,
                           input_ids_2, attention_mask_2):
        """ 
        Given a batch of pairs of sentences, extract comparison features for Quora and STS tasks.
        
        Useful for both Sim tasks.
        """
        
        embeds_1 = self.forward(input_ids_1, attention_mask_1)
        embeds_2 = self.forward(input_ids_2, attention_mask_2)
        cosine_sim = F.cosine_similarity(embeds_1, embeds_2).unsqueeze(-1)
        elem_prods = embeds_1 * embeds_2
        diff = torch.abs(embeds_1 - embeds_2)
        features = torch.cat([diff, elem_prods, cosine_sim], dim=1)
        features = self.comparison_features_fcn(features)
        
        return features
        


    def predict_paraphrase(self,
                           input_ids_1, attention_mask_1,
                           input_ids_2, attention_mask_2):
        '''Given a batch of pairs of sentences, outputs a single logit for predicting whether they are paraphrases.
        Note that your output should be unnormalized (a logit); it will be passed to the sigmoid function
        during evaluation.
        '''
        ### TODO
        features = self.extract_comparison_features(input_ids_1, attention_mask_1,
                                            input_ids_2, attention_mask_2)

        if self.clf == "linear":
            out = self.paraphrase_linear(features)
        elif self.clf == "nonlinear":
            out = self.paraphrase_nonlinear(features)
        else: # self.clf == "conv"
            out = self.paraphrase_conv(torch.unsqueeze(features, 1))
        return out


    def predict_similarity(self,
                           input_ids_1, attention_mask_1,
                           input_ids_2, attention_mask_2):
        '''Given a batch of pairs of sentences, outputs a single logit corresponding to how similar they are.
        Note that your output should be unnormalized (a logit).
        '''
        ### 
        features = self.extract_comparison_features(input_ids_1, attention_mask_1,
                                            input_ids_2, attention_mask_2)

        if self.clf == "linear":
            out = self.similarity_linear(features)
        elif self.clf == "nonlinear":
            out = self.similarity_nonlinear(features)
        else: # self.clf == "conv"
            out = self.similarity_conv(torch.unsqueeze(features, 1))
        return out
    
    
# ===== for EMA =====
class EMA:
    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = copy.deepcopy(model.state_dict())
        self.ema_model = copy.deepcopy(model)

    def update(self):
        for name, param in self.model.state_dict().items():
            if name in self.shadow:
                self.shadow[name] = self.decay * self.shadow[name] + (1.0 - self.decay) * param

    def apply_shadow(self):
        self.ema_model.load_state_dict(self.shadow)
        return self.ema_model
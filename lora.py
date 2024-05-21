import argparse
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch.utils.data import  DataLoader

from bert import BertModel
from optimizer import AdamW
from tqdm import tqdm

### Added imports below, removed some from above
from classifier import model_eval, save_model, load_data, test, seed_everything
from classifier import SentimentDataset, BertSentimentClassifier

from lora_bert import LoRABertSelfAttention, LoRALayer
###

TQDM_DISABLE=False


class BertSentimentClassifier(torch.nn.Module):
    '''
    This module performs sentiment classification using BERT embeddings on the SST dataset.

    In the SST dataset, there are 5 sentiment categories (from 0 - "negative" to 4 - "positive").
    Thus, your forward() should return one logit for each of the 5 classes.
    '''
    def __init__(self, config, r):
        super(BertSentimentClassifier, self).__init__()
        self.num_labels = config.num_labels
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        ### Apply LoRA self-attention layers to the BERT model.
        self.bert = replace_attention_layers(self.bert, r)
        ###

        # Pretrain mode does not require updating BERT paramters.
        assert config.fine_tune_mode in ["last-linear-layer", "full-model"]
        for param in self.bert.parameters():
            if config.fine_tune_mode == 'last-linear-layer':
                param.requires_grad = False
            elif config.fine_tune_mode == 'full-model':
                param.requires_grad = True

        # Create any instance variables you need to classify the sentiment of BERT embeddings.
        self.linear = torch.nn.Linear(config.hidden_size, config.num_labels)
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)

        return


    def forward(self, input_ids, attention_mask):
        '''Takes a batch of sentences and returns logits for sentiment classes'''
        # The final BERT contextualized embedding is the hidden state of [CLS] token (the first token).
        # HINT: You should consider what is an appropriate return value given that
        # the training loop currently uses F.cross_entropy as the loss function.
        ### 

        bert_out = self.bert(input_ids, attention_mask)
        pooler_out = bert_out['pooler_output']   # (N, D)  (the embeddings of the [CLS] (first) token, 'embeds[:, 0, :]')
        out = self.linear(self.dropout(pooler_out))

        return out

### NEW FUNCTION BELOW    
# Replace the self-attention layers in the BERT model with LoRA self-attention layers.
def replace_attention_layers(bert_model, r):
    for i, layer in enumerate(bert_model.bert_layers):
        layer.self_attention = LoRABertSelfAttention(bert_model.config, r)
    return bert_model
###

def train(args):
    device = torch.device('cpu')
    if args.use_gpu:
        if torch.cuda.is_available():
            device = torch.device('cuda') 
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
    # Create the data and its corresponding datasets and dataloader.
    train_data, num_labels = load_data(args.train, 'train')
    dev_data = load_data(args.dev, 'valid')

    train_dataset = SentimentDataset(train_data, args)
    dev_dataset = SentimentDataset(dev_data, args)

    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size,
                                  collate_fn=train_dataset.collate_fn)
    dev_dataloader = DataLoader(dev_dataset, shuffle=False, batch_size=args.batch_size,
                                collate_fn=dev_dataset.collate_fn)

    # Init model.
    config = {'hidden_dropout_prob': args.hidden_dropout_prob,
              'num_labels': num_labels,
              'hidden_size': 768,
              'data_dir': '.',
              'fine_tune_mode': args.fine_tune_mode}

    config = SimpleNamespace(**config)
    ### Adapted to include the LoRA hyperparameter r.
    lora_r = args.lora_r
    model = BertSentimentClassifier(config, lora_r)

    # Freeze the original BERT weights
    for param in model.bert.parameters():
        param.requires_grad = False

    # Ensure LoRA layers' parameters are trainable
    for param in model.bert.modules():
        if isinstance(param, LoRALayer):
            param.lora_A.weight.requires_grad = True
            param.lora_B.weight.requires_grad = True
    ###

    model = model.to(device)

    lr = args.lr
    optimizer = AdamW(model.parameters(), lr=lr)
    best_dev_acc = 0

    # Run for the specified number of epochs.
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        num_batches = 0
        for batch in tqdm(train_dataloader, desc=f'train-{epoch}', disable=TQDM_DISABLE):
            b_ids, b_mask, b_labels = (batch['token_ids'],
                                       batch['attention_mask'], batch['labels'])

            b_ids = b_ids.to(device)
            b_mask = b_mask.to(device)
            b_labels = b_labels.to(device)

            optimizer.zero_grad()
            logits = model(b_ids, b_mask)
            loss = F.cross_entropy(logits, b_labels.view(-1), reduction='sum') / args.batch_size

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            num_batches += 1

        train_loss = train_loss / (num_batches)

        train_acc, train_f1, *_  = model_eval(train_dataloader, model, device)
        dev_acc, dev_f1, *_ = model_eval(dev_dataloader, model, device)

        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            save_model(model, optimizer, args, config, args.filepath)

        print(f"Epoch {epoch}: train loss :: {train_loss :.3f}, train acc :: {train_acc :.3f}, dev acc :: {dev_acc :.3f}")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=11711)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--fine-tune-mode", type=str,
                        help='last-linear-layer: the BERT parameters are frozen and the task specific head parameters are updated; full-model: BERT parameters are updated as well',
                        choices=('last-linear-layer', 'full-model'), default="last-linear-layer")
    parser.add_argument("--use_gpu", action='store_true')

    parser.add_argument("--batch_size", help='sst: 64, cfimdb: 8 can fit a 12GB GPU', type=int, default=8)
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.3)
    parser.add_argument("--lr", type=float, help="learning rate, default lr for 'pretrain': 1e-3, 'finetune': 1e-5",
                        default=1e-3)
    ### Added LoRA hyperparameter r argument
    parser.add_argument("--lora_r", type=int, help="rank for LoRA matrices", default=2)
    ###

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    seed_everything(args.seed)

    print('Training Sentiment Classifier on SST...')
    config = SimpleNamespace(
        ### Added LoRA to filenames
        filepath='LoRA-sst-classifier.pt',
        ###
        lr=args.lr,
        use_gpu=args.use_gpu,
        epochs=args.epochs,
        batch_size=args.batch_size,
        hidden_dropout_prob=args.hidden_dropout_prob,
        train='data/ids-sst-train.csv',
        dev='data/ids-sst-dev.csv',
        test='data/ids-sst-test-student.csv',
        fine_tune_mode=args.fine_tune_mode,
        ### Added LoRA to filenames
        dev_out = 'predictions/' + args.fine_tune_mode + '-LoRA-sst-dev-out.csv',
        test_out = 'predictions/' + args.fine_tune_mode + '-LoRA-sst-test-out.csv',
        ###
        ### Pass loRA hyperparameter r to config
        lora_r = args.lora_r
        ###
    )

    train(config)

    print('Evaluating on SST...')
    test(config)

    print('Training Sentiment Classifier on cfimdb...')
    config = SimpleNamespace(
        ### Added LoRA to filenames
        filepath='LoRA-cfimdb-classifier.pt',
        ###
        lr=args.lr,
        use_gpu=args.use_gpu,
        epochs=args.epochs,
        batch_size=8,
        hidden_dropout_prob=args.hidden_dropout_prob,
        train='data/ids-cfimdb-train.csv',
        dev='data/ids-cfimdb-dev.csv',
        test='data/ids-cfimdb-test-student.csv',
        fine_tune_mode=args.fine_tune_mode,
        ### Added LoRA to filenames
        dev_out = 'predictions/' + args.fine_tune_mode + '-LoRA-cfimdb-dev-out.csv',
        test_out = 'predictions/' + args.fine_tune_mode + '-LoRA-cfimdb-test-out.csv',
        ###
        ### Pass loRA hyperparameter r to config
        lora_r = args.lora_r
        ###
    )

    train(config)

    print('Evaluating on cfimdb...')
    test(config)

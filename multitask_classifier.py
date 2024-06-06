'''
Multitask BERT class, starter training code, evaluation, and test code.

Of note are:
* class MultitaskBERT: Your implementation of multitask BERT.
* function train_multitask: Training procedure for MultitaskBERT. Starter code
    copies training procedure from `classifier.py` (single-task SST).
* function test_multitask: Test procedure for MultitaskBERT. This function generates
    the required files for submission.

Running `python multitask_classifier.py` trains and tests your MultitaskBERT and
writes all required submission files.
'''

import random, numpy as np, argparse
from types import SimpleNamespace

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from bert import BertModel
from optimizer import AdamW
from tqdm import tqdm

from datasets import (
    SentenceClassificationDataset,
    SentenceClassificationTestDataset,
    SentencePairDataset,
    SentencePairTestDataset,
    load_multitask_data
)

from evaluation import model_eval_sst, model_eval_para, model_eval_sts, model_eval_multitask, model_eval_test_multitask, get_leaderboard_score
from lora_bert import inject_lora
import time


import warnings
import os
import json
import csv


TQDM_DISABLE=False


# Fix the random seed.
def seed_everything(seed=11711):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


BERT_HIDDEN_SIZE = 768
N_SENTIMENT_CLASSES = 5


class MultitaskBERT(nn.Module):
    '''
    This module should use BERT for 3 tasks:

    - Sentiment classification (predict_sentiment)
    - Paraphrase detection (predict_paraphrase)
    - Semantic Textual Similarity (predict_similarity)
    '''
    def __init__(self, config):
        super(MultitaskBERT, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        # last-layer mode does not require updating BERT paramters.
        assert config.fine_tune_mode in ["last-layer", "full-model"]
        assert config.lora_dict['mode'] in ['none', 'attn', 'attn-only', 'all-lin', 'all-lin-only']
        assert config.lora_dict['r'] > 0 or config.lora_dict['mode'] == 'none'
        assert config.lora_dict['dora'] in [0, 1]
        # LoRA settings
        if config.fine_tune_mode != 'full-model' and config.lora_dict['mode'] != 'none':
            raise ValueError("LoRA can only be used in full-model fine-tuning mode.")
        
        # Pretrain mode does not require updating BERT parameters.
        for param in self.bert.parameters():
            if config.fine_tune_mode == 'last-layer' or config.lora_dict['mode'] in ['attn-only', 'all-lin-only']:
                param.requires_grad = False
            elif config.fine_tune_mode == 'full-model':
                param.requires_grad = True
        
        if config.lora_dict['mode'] != 'none':
            self.bert = inject_lora(self.bert, config.lora_dict['mode'], config.lora_dict['r'], config.lora_dict['dora'])

        assert config.clf in ["linear", "nonlinear", "conv"]
        self.clf = config.clf
        # You will want to add layers here to perform the downstream tasks.
        ### 

        # ===== sentiment =====
        self.sentiment_linear = nn.Linear(BERT_HIDDEN_SIZE, N_SENTIMENT_CLASSES)
        self.sentiment_nonlinear = nn.Sequential(nn.Linear(BERT_HIDDEN_SIZE, BERT_HIDDEN_SIZE//2),
                                        nn.ReLU(),
                                        nn.Linear(BERT_HIDDEN_SIZE//2, N_SENTIMENT_CLASSES))
        
        self.sentiment_conv = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=4, kernel_size=3), # note: input needs to be unsqueezed
            nn.Flatten(),  # Flatten layer to convert (N, 4, D-2) to (N, 4*(D-2))
            nn.Linear(4 * (BERT_HIDDEN_SIZE - 2), BERT_HIDDEN_SIZE // 2),  
            nn.ReLU(),  
            nn.Linear(BERT_HIDDEN_SIZE // 2, N_SENTIMENT_CLASSES)  
        )

        # ===== paraphrase ======
        self.paraphrase_linear = nn.Linear(BERT_HIDDEN_SIZE, 1)
        self.paraphrase_nonlinear = nn.Sequential(nn.Linear(BERT_HIDDEN_SIZE, BERT_HIDDEN_SIZE//2),
                                        nn.ReLU(),
                                        nn.Linear(BERT_HIDDEN_SIZE//2, 1))
        self.paraphrase_conv = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=4, kernel_size=3), # note: input needs to be unsqueezed
            nn.Flatten(),  # Flatten layer to convert (N, 4, D-2) to (N, 4*(D-2))
            nn.Linear(4 * (BERT_HIDDEN_SIZE - 2), BERT_HIDDEN_SIZE // 2),  
            nn.ReLU(),  
            nn.Linear(BERT_HIDDEN_SIZE // 2, 1)  
        )

        # ===== similarity ========
        self.similarity_linear = nn.Linear(2*BERT_HIDDEN_SIZE+1, 1)
        self.similarity_nonlinear = nn.Sequential(nn.Linear(2*BERT_HIDDEN_SIZE+1, BERT_HIDDEN_SIZE),
                                        nn.ReLU(),
                                        nn.Linear(BERT_HIDDEN_SIZE, 1))
        self.similarity_conv = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=4, kernel_size=3), # note: input needs to be unsqueezed
            nn.Flatten(),  # Flatten layer to convert (N, 4, D-2) to (N, 4*(D-2))
            nn.Linear(4 * (2*BERT_HIDDEN_SIZE+1 - 2), BERT_HIDDEN_SIZE // 2),  
            nn.ReLU(),  
            nn.Linear(BERT_HIDDEN_SIZE // 2, 1)  
        )

        # ===== helper layers ======
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, attention_mask):
        """Takes a batch of sentences and produces embeddings for them.
        
        Shape returned = Batch size, bert hidden dimension (768)  (N, D)
        """
        # The final BERT embedding is the hidden state of [CLS] token (the first token)
        # Here, you can start by just returning the embeddings straight from BERT.
        # When thinking of improvements, you can later try modifying this
        # (e.g., by adding other layers).
        ### TODO
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        embeddings = outputs['pooler_output'] 
        return embeddings

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


    def predict_paraphrase(self,
                           input_ids_1, attention_mask_1,
                           input_ids_2, attention_mask_2):
        '''Given a batch of pairs of sentences, outputs a single logit for predicting whether they are paraphrases.
        Note that your output should be unnormalized (a logit); it will be passed to the sigmoid function
        during evaluation.
        '''
        ### TODO
        embeds_1 = self.forward(input_ids_1, attention_mask_1)
        embeds_2 = self.forward(input_ids_2, attention_mask_2)
        diff = torch.abs(embeds_1 - embeds_2)

        if self.clf == "linear":
            out = self.paraphrase_linear(diff)
        elif self.clf == "nonlinear":
            out = self.paraphrase_nonlinear(diff)
        else: # self.clf == "conv"
            out = self.paraphrase_conv(torch.unsqueeze(diff, 1))
        return out


    def predict_similarity(self,
                           input_ids_1, attention_mask_1,
                           input_ids_2, attention_mask_2):
        '''Given a batch of pairs of sentences, outputs a single logit corresponding to how similar they are.
        Note that your output should be unnormalized (a logit).
        '''
        ### TODO
        embeds_1 = self.forward(input_ids_1, attention_mask_1)
        embeds_2 = self.forward(input_ids_2, attention_mask_2)
        cosine_sim = F.cosine_similarity(embeds_1, embeds_2).unsqueeze(-1)
        elem_prods = embeds_1 * embeds_2
        diff = torch.abs(embeds_1 - embeds_2)
        features = torch.cat([diff, elem_prods, cosine_sim], dim=1)

        if self.clf == "linear":
            out = self.similarity_linear(features)
        elif self.clf == "nonlinear":
            out = self.similarity_nonlinear(features)
        else: # self.clf == "conv"
            out = self.similarity_conv(torch.unsqueeze(features, 1))
        return out


def save_model(model, optimizer, args, config, filepath):
    save_info = {
        'model': model.state_dict(),
        'optim': optimizer.state_dict(),
        'args': args,
        'model_config': config,
        'system_rng': random.getstate(),
        'numpy_rng': np.random.get_state(),
        'torch_rng': torch.random.get_rng_state(),
    }

    torch.save(save_info, filepath)
    print(f"Saved the model to {filepath}!")


def train_multitask(args, benchmark=False):
    device = torch.device('cpu')
    if args.use_gpu:
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            device = torch.device('mps')

    print(f"Device Set: {device}\n")

    # Create datasets and dataloaders
    sst_train_data, num_labels, para_train_data, sts_train_data = load_multitask_data(args.sst_train, args.para_train, args.sts_train, split='train')
    sst_dev_data, _, para_dev_data, sts_dev_data = load_multitask_data(args.sst_dev, args.para_dev, args.sts_dev, split='dev')

    sst_train_data = SentenceClassificationDataset(sst_train_data, args)
    sst_dev_data = SentenceClassificationDataset(sst_dev_data, args)

    sst_train_dataloader = DataLoader(sst_train_data, shuffle=True, batch_size=args.batch_size, collate_fn=sst_train_data.collate_fn)
    sst_dev_dataloader = DataLoader(sst_dev_data, shuffle=False, batch_size=args.batch_size, collate_fn=sst_dev_data.collate_fn)

    para_train_data = SentencePairDataset(para_train_data, args)
    para_dev_data = SentencePairDataset(para_dev_data, args)

    para_train_dataloader = DataLoader(para_train_data, shuffle=True, batch_size=args.batch_size, collate_fn=para_train_data.collate_fn)
    para_dev_dataloader = DataLoader(para_dev_data, shuffle=False, batch_size=args.batch_size, collate_fn=para_dev_data.collate_fn)

    sts_train_data = SentencePairDataset(sts_train_data, args)
    sts_dev_data = SentencePairDataset(sts_dev_data, args, isRegression=True)

    sts_train_dataloader = DataLoader(sts_train_data, shuffle=True, batch_size=args.batch_size, collate_fn=sts_train_data.collate_fn)
    sts_dev_dataloader = DataLoader(sts_dev_data, shuffle=False, batch_size=args.batch_size, collate_fn=sts_dev_data.collate_fn)

    # Initialize model
    config = {
        'hidden_dropout_prob': args.hidden_dropout_prob,
        'num_labels': num_labels,
        'hidden_size': 768,
        'data_dir': '.',
        'fine_tune_mode': args.fine_tune_mode,
        'clf': args.clf,
        'lora_dict': args.lora_dict}
    config = SimpleNamespace(**config)
    model = MultitaskBERT(config)
    model = model.to(device)

    lr = args.lr
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # lr scheduler halves the learning rate every epoch
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 0.5 ** epoch)
    log_dir = os.path.join("logs", args.filepath)
    writer = SummaryWriter(log_dir = log_dir)

    best_leaderboard_score = 0

    # ===== AMP =====
    if args.amp:
        print("\nTurning on Multi-Precision Training...\n")
        if device.type == 'mps':
            print("\nMPS Device Detected! Deactivating AMP (incompatible).\n")
            args.amp = False
    if args.amp:  # and device is not mps
        gradscaler = torch.cuda.amp.GradScaler()

    if benchmark:
        total_sst_time = 0
        total_para_time = 0
        total_sts_time = 0

        total_sst_memory = 0
        total_para_memory = 0
        total_sts_memory = 0
    # Train for the specified number of epochs
    for epoch in range(args.epochs):
        model.train()  # put model in training mode

        if benchmark:
            torch.cuda.reset_peak_memory_stats()
        if args.train_sst:
            print(f"\n================== Training SST (Epoch {epoch}) ==================\n")
            sst_train_loss = 0
            sst_num_batches = 0
       
            start_time = time.time()
            for batch in tqdm(sst_train_dataloader, desc=f'train-{epoch}', disable=TQDM_DISABLE):
                b_ids, b_mask, b_labels = (batch['token_ids'], batch['attention_mask'], batch['labels'])
                b_ids = b_ids.to(device)
                b_mask = b_mask.to(device)
                b_labels = b_labels.to(device)

                optimizer.zero_grad()

                if args.amp:  # auto multi-precision
                    with torch.cuda.amp.autocast():
                        logits = model.predict_sentiment(b_ids, b_mask)
                        loss = F.cross_entropy(logits, b_labels.view(-1), reduction='sum') / args.batch_size

                    gradscaler.scale(loss).backward()
                    gradscaler.step(optimizer)
                    gradscaler.update()
                else:  # vanilla
                    logits = model.predict_sentiment(b_ids, b_mask)
                    loss = F.cross_entropy(logits, b_labels.view(-1), reduction='sum') / args.batch_size
                    loss.backward()
                    optimizer.step()

                sst_train_loss += loss.item()
                sst_num_batches += 1
            
            if benchmark:
                total_sst_time += time.time() - start_time
                peak_memory_sst = torch.cuda.max_memory_allocated() / (1024 ** 3)  # Convert bytes to GB
                total_sst_memory += peak_memory_sst
            sst_train_loss = sst_train_loss / sst_num_batches
            
        if benchmark:
            torch.cuda.reset_peak_memory_stats()
        if args.train_quora:
            print(f"\n================== Training Quora (Epoch {epoch}) ==================\n")
            para_train_loss = 0
            para_num_batches = 0
            
            start_time = time.time()
            for batch in tqdm(para_train_dataloader, desc=f'train-{epoch}', disable=TQDM_DISABLE):
                (b_ids1, b_mask1, b_ids2, b_mask2, b_labels, b_sent_ids) = (batch['token_ids_1'], batch['attention_mask_1'], batch['token_ids_2'], batch['attention_mask_2'], batch['labels'], batch['sent_ids'])

                b_ids1 = b_ids1.to(device)
                b_mask1 = b_mask1.to(device)
                b_ids2 = b_ids2.to(device)
                b_mask2 = b_mask2.to(device)
                b_labels = b_labels.to(device)

                optimizer.zero_grad()

                if args.amp:  # auto multi-precision
                    with torch.cuda.amp.autocast():
                        logits = model.predict_paraphrase(b_ids1, b_mask1, b_ids2, b_mask2).flatten()
                        b_labels = b_labels.flatten().float()  # reshape to match logits
                        criterion = torch.nn.BCEWithLogitsLoss()  # more stable, can autocast
                        loss = criterion(logits, b_labels)

                    gradscaler.scale(loss).backward()
                    gradscaler.step(optimizer)
                    gradscaler.update()
                else:  # vanilla
                    logits = model.predict_paraphrase(b_ids1, b_mask1, b_ids2, b_mask2).flatten()
                    b_labels = b_labels.flatten().float()  # reshape to match logits
                    criterion = torch.nn.BCEWithLogitsLoss()  # more stable, can autocast
                    loss = criterion(logits, b_labels)
                    loss.backward()
                    optimizer.step()

                para_train_loss += loss.item()
                para_num_batches += 1

            if benchmark:
                total_para_time += time.time() - start_time
                peak_memory_para = torch.cuda.max_memory_allocated() / (1024 ** 3)  # Convert bytes to GB
                total_para_memory += peak_memory_para
            para_train_loss = para_train_loss / para_num_batches
            
        if benchmark:
            torch.cuda.reset_peak_memory_stats()
        if args.train_sts:
            print(f"\n================== Training STS (Epoch {epoch}) ==================\n")
            sts_train_loss = 0
            sts_num_batches = 0

            start_time = time.time()
            for batch in tqdm(sts_train_dataloader, desc=f'train-{epoch}', disable=TQDM_DISABLE):
                (b_ids1, b_mask1, b_ids2, b_mask2, b_labels, b_sent_ids) = (batch['token_ids_1'], batch['attention_mask_1'], batch['token_ids_2'], batch['attention_mask_2'], batch['labels'], batch['sent_ids'])

                b_ids1 = b_ids1.to(device)
                b_mask1 = b_mask1.to(device)
                b_ids2 = b_ids2.to(device)
                b_mask2 = b_mask2.to(device)
                b_labels = b_labels.to(device)

                optimizer.zero_grad()

                if args.amp:  # auto multi-precision
                    with torch.cuda.amp.autocast():
                        logits = model.predict_similarity(b_ids1, b_mask1, b_ids2, b_mask2).flatten()
                        b_labels = b_labels.flatten().float()  # reshape to match logits
                        criterion = torch.nn.MSELoss()  # MSE loss since labels are 0-5
                        loss = criterion(logits, b_labels)

                    gradscaler.scale(loss).backward()
                    gradscaler.step(optimizer)
                    gradscaler.update()
                else:  # vanilla
                    logits = model.predict_similarity(b_ids1, b_mask1, b_ids2, b_mask2).flatten()
                    b_labels = b_labels.flatten().float()  # reshape to match logits
                    criterion = torch.nn.MSELoss()  # MSE loss since labels are 0-5
                    loss = criterion(logits, b_labels)
                    loss.backward()
                    optimizer.step()

                sts_train_loss += loss.item()
                sts_num_batches += 1
            if benchmark:
                total_sts_time += time.time() - start_time
                peak_memory_sts = torch.cuda.max_memory_allocated() / (1024 ** 3)  # Convert bytes to GB
                total_sts_memory += peak_memory_sts
            sts_train_loss = sts_train_loss / sts_num_batches

        print(f"\n============== End of Epoch Evaluation ==============")

        # ====== Compute SST Accs ========
        print(f"Epoch {epoch}\n")

        if args.train_sst:
            sst_train_acc, sst_train_f1, *_ = model_eval_sst(sst_train_dataloader, model, device)
            sst_dev_acc, sst_dev_f1, *_ = model_eval_sst(sst_dev_dataloader, model, device)

            writer.add_scalar('SST/Train_Loss', sst_train_loss, epoch)
            writer.add_scalar('SST/Train_Accuracy', sst_train_acc, epoch)
            writer.add_scalar('SST/Train_F1', sst_train_f1, epoch)
            writer.add_scalar('SST/Dev_Accuracy', sst_dev_acc, epoch)
            writer.add_scalar('SST/Dev_F1', sst_dev_f1, epoch)

        # ====== Compute Quora Accs ======
        if args.train_quora and args.quora_epoch_eval:  # extremely costly
            para_train_acc, para_train_f1, *_ = model_eval_para(para_train_dataloader, model, device)
            para_dev_acc, para_dev_f1, *_ = model_eval_para(para_dev_dataloader, model, device)

            writer.add_scalar('Quora/Train_Loss', para_train_loss, epoch)
            writer.add_scalar('Quora/Train_Accuracy', para_train_acc, epoch)
            writer.add_scalar('Quora/Train_F1', para_train_f1, epoch)
            writer.add_scalar('Quora/Dev_Accuracy', para_dev_acc, epoch)
            writer.add_scalar('Quora/Dev_F1', para_dev_f1, epoch)

        # ====== Compute STS Accs =======
        if args.train_sts:
            sts_train_corr, *_ = model_eval_sts(sts_train_dataloader, model, device)
            sts_dev_corr, *_ = model_eval_sts(sts_dev_dataloader, model, device)

            writer.add_scalar('STS/Train_Loss', sts_train_loss, epoch)
            writer.add_scalar('STS/Train_Correlation', sts_train_corr, epoch)
            writer.add_scalar('STS/Dev_Correlation', sts_dev_corr, epoch)

        if args.train_sst:
            print(f"SST—— train loss :: {sst_train_loss :.3f}, train acc :: {sst_train_acc :.3f}, dev acc :: {sst_dev_acc :.3f}")
        if args.train_quora and args.quora_epoch_eval:
            print(f"Para—— train loss :: {para_train_loss :.3f}, train acc :: {para_train_acc :.3f}, dev acc :: {para_dev_acc :.3f}")
        else:
            print(f"Skipping Quora Eval until the end (costly compute)")
        if args.train_sts:
            print(f"STS—— train loss :: {sts_train_loss :.3f}, train corr :: {sts_train_corr :.3f}, dev corr :: {sts_dev_corr :.3f}")

        print(f"================ Checkpointing =============")
        # ===== Checkpointing ====
        dev_leaderboard_score = get_leaderboard_score(sst_dev_acc if args.train_sst else 0, 
                                            para_dev_acc if args.train_quora and args.quora_epoch_eval else 0, 
                                            sts_dev_corr if args.train_sts else 0)
        if dev_leaderboard_score > best_leaderboard_score:  # TODO: come up with more clever checkpointing than just caring about SST
            best_leaderboard_score = dev_leaderboard_score
            print(f"New Best Leaderboard Avg. Score: {best_leaderboard_score:.3f}!")
            save_model(model, optimizer, args, config, args.filepath)

        # step the learning rate scheduler (halves lr)
        scheduler.step()
    writer.close()
    
    if benchmark:
        average_sst_time = total_sst_time / args.epochs if args.train_sst else 0
        average_para_time = total_para_time / args.epochs if args.train_sst else 0
        average_sts_time = total_sts_time / args.epochs if args.train_sst else 0
        
        average_sst_memory = total_sst_memory / args.epochs if args.train_sst else 0
        average_para_memory = total_para_memory / args.epochs if args.train_sst else 0
        average_sts_memory = total_sts_memory / args.epochs if args.train_sst else 0
        return average_sst_time, average_para_time, average_sts_time, average_sst_memory, average_para_memory, average_sts_memory



def test_multitask(args, benchmark=False):
    '''Test and save predictions on the dev and test sets of all three tasks.'''
    with torch.no_grad():
        device = torch.device('cpu')
        if args.use_gpu:
            if torch.cuda.is_available():
                device = torch.device('cuda')
            elif torch.backends.mps.is_available():
                device = torch.device('mps')

        print(f"Device Set: {device}\n")

        
        saved = torch.load(args.filepath)
        config = saved['model_config']

        model = MultitaskBERT(config)
        model.load_state_dict(saved['model'])
        model = model.to(device)
        print(f"Loaded model to test from {args.filepath}")

        sst_test_data, num_labels,para_test_data, sts_test_data = \
            load_multitask_data(args.sst_test,args.para_test, args.sts_test, split='test')

        sst_dev_data, num_labels,para_dev_data, sts_dev_data = \
            load_multitask_data(args.sst_dev,args.para_dev,args.sts_dev,split='dev')

        sst_test_data = SentenceClassificationTestDataset(sst_test_data, args)
        sst_dev_data = SentenceClassificationDataset(sst_dev_data, args)

        sst_test_dataloader = DataLoader(sst_test_data, shuffle=True, batch_size=args.batch_size,
                                         collate_fn=sst_test_data.collate_fn)
        sst_dev_dataloader = DataLoader(sst_dev_data, shuffle=False, batch_size=args.batch_size,
                                        collate_fn=sst_dev_data.collate_fn)

        para_test_data = SentencePairTestDataset(para_test_data, args)
        para_dev_data = SentencePairDataset(para_dev_data, args)

        para_test_dataloader = DataLoader(para_test_data, shuffle=True, batch_size=args.batch_size,
                                          collate_fn=para_test_data.collate_fn)
        para_dev_dataloader = DataLoader(para_dev_data, shuffle=False, batch_size=args.batch_size,
                                         collate_fn=para_dev_data.collate_fn)

        sts_test_data = SentencePairTestDataset(sts_test_data, args)
        sts_dev_data = SentencePairDataset(sts_dev_data, args, isRegression=True)

        sts_test_dataloader = DataLoader(sts_test_data, shuffle=True, batch_size=args.batch_size,
                                         collate_fn=sts_test_data.collate_fn)
        sts_dev_dataloader = DataLoader(sts_dev_data, shuffle=False, batch_size=args.batch_size,
                                        collate_fn=sts_dev_data.collate_fn)
        
        print(f"\n============= Evaluating Dev Scores (SST, Para, STS) =============\n")

        dev_sentiment_accuracy,dev_sst_y_pred, dev_sst_sent_ids, \
            dev_paraphrase_accuracy, dev_para_y_pred, dev_para_sent_ids, \
            dev_sts_corr, dev_sts_y_pred, dev_sts_sent_ids = model_eval_multitask(sst_dev_dataloader,
                                                                    para_dev_dataloader,
                                                                    sts_dev_dataloader, model, device)
        
        if not benchmark:
            print(f"\n============= Evaluating Test Scores (SST, Para, STS) =============\n")

            test_sst_y_pred, \
                test_sst_sent_ids, test_para_y_pred, test_para_sent_ids, test_sts_y_pred, test_sts_sent_ids = \
                    model_eval_test_multitask(sst_test_dataloader,
                                            para_test_dataloader,
                                            sts_test_dataloader, model, device)

            with open(args.sst_dev_out, "w+") as f:
                print(f"dev sentiment acc :: {dev_sentiment_accuracy :.3f}")
                f.write(f"id \t Predicted_Sentiment \n")
                for p, s in zip(dev_sst_sent_ids, dev_sst_y_pred):
                    f.write(f"{p} , {s} \n")

            with open(args.sst_test_out, "w+") as f:
                f.write(f"id \t Predicted_Sentiment \n")
                for p, s in zip(test_sst_sent_ids, test_sst_y_pred):
                    f.write(f"{p} , {s} \n")

            with open(args.para_dev_out, "w+") as f:
                print(f"dev paraphrase acc :: {dev_paraphrase_accuracy :.3f}")
                f.write(f"id \t Predicted_Is_Paraphrase \n")
                for p, s in zip(dev_para_sent_ids, dev_para_y_pred):
                    f.write(f"{p} , {s} \n")

            with open(args.para_test_out, "w+") as f:
                f.write(f"id \t Predicted_Is_Paraphrase \n")
                for p, s in zip(test_para_sent_ids, test_para_y_pred):
                    f.write(f"{p} , {s} \n")

            with open(args.sts_dev_out, "w+") as f:
                print(f"dev sts corr :: {dev_sts_corr :.3f}")
                f.write(f"id \t Predicted_Similiary \n")
                for p, s in zip(dev_sts_sent_ids, dev_sts_y_pred):
                    f.write(f"{p} , {s} \n")

            with open(args.sts_test_out, "w+") as f:
                f.write(f"id \t Predicted_Similiary \n")
                for p, s in zip(test_sts_sent_ids, test_sts_y_pred):
                    f.write(f"{p} , {s} \n")
        else:
            return dev_sentiment_accuracy, dev_paraphrase_accuracy, dev_sts_corr


def get_args():
    parser = argparse.ArgumentParser()

    # multi-precision tuning
    parser.add_argument("--amp",  action='store_true', help='Turn on auto multi-precision for training with bfloat16')

    # clf type (linear, nonlinear, conv)
    parser.add_argument("--clf", type=str, default="linear", 
                        help="Type of classifier layer. Supported: linear, nonlinear, or conv.",
                        choices=("linear", "nonlinear", "conv"))

    # lr scheduling 

    # dataset selections for training
    parser.add_argument("--train_sst", action='store_true', help='Train against SST sentiment dataset (CELoss)')
    parser.add_argument("--train_quora", action='store_true', help='Train against Quora paraphrase dataset (BCELoss) (costly!)')
    parser.add_argument("--train_sts", action='store_true', help='Train against STS similarity dataset (BCELoss)')
    parser.add_argument("--quora_epoch_eval", action='store_true', help='Evaluate quora performance every epoch (Costly!!)')

    parser.add_argument("--sst_train", type=str, default="data/ids-sst-train.csv")
    parser.add_argument("--sst_dev", type=str, default="data/ids-sst-dev.csv")
    parser.add_argument("--sst_test", type=str, default="data/ids-sst-test-student.csv")

    parser.add_argument("--para_train", type=str, default="data/quora-train.csv")
    parser.add_argument("--para_dev", type=str, default="data/quora-dev.csv")
    parser.add_argument("--para_test", type=str, default="data/quora-test-student.csv")

    parser.add_argument("--sts_train", type=str, default="data/sts-train.csv")
    parser.add_argument("--sts_dev", type=str, default="data/sts-dev.csv")
    parser.add_argument("--sts_test", type=str, default="data/sts-test-student.csv")

    parser.add_argument("--seed", type=int, default=11711)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--fine-tune-mode", type=str,
                        help='last-layer: the BERT parameters are frozen and the task specific head parameters are updated; full-model: BERT parameters are updated as well',
                        choices=('last-layer', 'full-model'), default="last-layer")
    parser.add_argument("--use_gpu", action='store_true')

    parser.add_argument("--sst_dev_out", type=str, default="predictions/sst-dev-output.csv")
    parser.add_argument("--sst_test_out", type=str, default="predictions/sst-test-output.csv")

    parser.add_argument("--para_dev_out", type=str, default="predictions/para-dev-output.csv")
    parser.add_argument("--para_test_out", type=str, default="predictions/para-test-output.csv")

    parser.add_argument("--sts_dev_out", type=str, default="predictions/sts-dev-output.csv")
    parser.add_argument("--sts_test_out", type=str, default="predictions/sts-test-output.csv")

    parser.add_argument("--batch_size", help='sst: 64, cfimdb: 8 can fit a 12GB GPU', type=int, default=8)
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.3)
    parser.add_argument("--lr", type=float, help="learning rate", default=1e-5)

    parser.add_argument("--lora_dict", type=str, default='{"mode": "none", "r": 0, "dora": 0}')
    parser.add_argument("--benchmark-results", type=str, default="benchmark-results-dora-last-layer.csv")

    args = parser.parse_args()
    args.lora_dict = json.loads(args.lora_dict)

    return args

def append_results_to_csv(args, metrics):
    fields = ['Total Accuracy', 'SST Dev Accuracy', 'Para Dev Accuracy', 'STS Dev Correlation', 'SST Time', 'Para Time', 'STS Time', 'SST Memory', 'Para Memory', 'STS Memory', 'LoRA Mode', 'LoRA R', 'DoRA', 'Batch Size', 'Learning Rate', 'Epochs', 'Fine-Tune Mode', 'Dropout Probability']
    file_exists = os.path.isfile(args.benchmark_results)
    
    with open(args.benchmark_results, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fields)
        
        if not file_exists:
            writer.writeheader()  # Write headers only once
        
        writer.writerow(metrics)

if __name__ == "__main__":
    args = get_args()
    args.filepath = f'{args.fine_tune_mode}-{args.epochs}-{args.lr}-{args.clf}-multitask.pt' # Save path.
    seed_everything(args.seed)  # Fix the seed for reproducibility.
    average_sst_time, average_para_time, average_sts_time, average_sst_memory, average_para_memory, average_sts_memory = train_multitask(args)
    dev_sentiment_accuracy, dev_paraphrase_accuracy, dev_sts_corr = test_multitask(args)

    total_accuracy = get_leaderboard_score(dev_sentiment_accuracy, dev_paraphrase_accuracy, dev_sts_corr)
    metrics = {
    'Fine-Tune Mode': args.fine_tune_mode,
    'Epochs': args.epochs,
    'Batch Size': args.batch_size,
    'Dropout Probability': args.hidden_dropout_prob,
    'Learning Rate': args.lr,
    'LoRA Mode': args.lora_dict['mode'],
    'LoRA R': args.lora_dict['r'],
    'DoRA': args.lora_dict['dora'],
    'SST Dev Accuracy': dev_sentiment_accuracy,
    'SST Time': average_sst_time,
    'SST Memory': average_sst_memory,
    'Para Dev Accuracy': dev_paraphrase_accuracy,
    'Para Time': average_para_time,
    'Para Memory': average_para_memory,
    'STS Dev Correlation': dev_sts_corr,
    'STS Time': average_sts_time,
    'STS Memory': average_sts_memory,
    'Total Accuracy': total_accuracy
    }
    append_results_to_csv(args, metrics)


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
from torch import optim
import torch.optim.lr_scheduler as lr_scheduler
from tqdm import tqdm
from datasets import (
    SentenceClassificationDataset,
    SentenceClassificationTestDataset,
    SentencePairDataset,
    SentencePairTestDataset,
    load_multitask_data
)
from evaluation import model_eval_sst, model_eval_para, model_eval_sts, model_eval_multitask, model_eval_test_multitask, get_leaderboard_score
from multitask_bert import MultitaskBERT, EMA
import time
import warnings
import os
import json
import csv
from datetime import datetime
from more_utils import seed_everything, save_model
from spacecutter.losses import CumulativeLinkLoss
from copy import copy

from globals import BERT_HIDDEN_SIZE, N_SENTIMENT_CLASSES, TQDM_DISABLE




# ============= additional Loss funcs for STS (corr) ============
def pearson_correlation_loss(preds, target):
    vx = preds - torch.mean(preds)
    vy = target - torch.mean(target)
    cost = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))
    return 1 - cost  # return 1 - correlation to make it a loss

def log_cosh_loss(preds, target):
    def log_cosh(x):
        return x + torch.nn.functional.softplus(-2. * x) - torch.log(torch.tensor(2.0))
    return torch.mean(log_cosh(preds - target))


# =========== additional loss funcs for SST (sent) ========
class OrdinalCrossEntropyLoss(nn.Module):
    def __init__(self):
        super(OrdinalCrossEntropyLoss, self).__init__()

    def forward(self, logits, targets):
        targets_one_hot = F.one_hot(targets, num_classes=logits.size(1)).float()
        # Create cumulative probability for each class
        cumulative_probs = torch.cumsum(F.softmax(logits, dim=1), dim=1)
        loss = - torch.sum(targets_one_hot * torch.log(cumulative_probs) + (1 - targets_one_hot) * torch.log(1 - cumulative_probs))
        return loss 


def train_multitask(args, benchmark=False):
    device = torch.device('cpu')
    if args.use_gpu:
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            device = torch.device('mps')

    print(f"Device Set: {device}\n")

    # Create datasets and dataloaders
    sst_train_data, num_labels, para_train_data, sts_train_data = load_multitask_data(args.sst_train, args.para_train, args.sts_train, split='train', extra_clean = args.extra_clean)
    sst_dev_data, _, para_dev_data, sts_dev_data = load_multitask_data(args.sst_dev, args.para_dev, args.sts_dev, split='dev', extra_clean = args.extra_clean)

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

    # =================== Initialize model =====================
    config = {
        'hidden_dropout_prob': args.hidden_dropout_prob,
        'num_labels': num_labels,
        'hidden_size': 768,
        'num_berts': args.num_berts,
        'data_dir': '.',
        'fine_tune_mode': args.fine_tune_mode,
        'clf': args.clf,
        'lora_dict': args.lora_dict}
    config = SimpleNamespace(**config)
    model = MultitaskBERT(config)
    model = model.to(device)
    
    if args.use_ema:
        decay = 0.999
        ema = EMA(model, decay = decay)   # smaller decay beta since we use few epochs


    # ============= Optimizer and LR =================
    # Set up the optimizer
    
    # our handmade one uWu
    # optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    
    # Original experiments, grid search, freezing etc. used Adam as follows:
    # optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)

    # AdamW optimizer: variant of Adam with better handling of weight decay
    # optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-1)

    # SGD with Nesterov momentum: often better for image classification tasks
    # optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, nesterov=True, weight_decay=1e-1)
    #optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, nesterov=True, weight_decay = 1e-5)  # 

    # Plain SGD: simple and often effective, without Nesterov momentum
    # optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # RAdam (Rectified Adam): combines the benefits of adaptive learning rate and robustness
    # optimizer = optim.RAdam(model.parameters(), lr=args.lr, weight_decay=1e-5)

    # SparseAdam: a variant of Adam meant to handle sparse gradients more efficiently
    # optimizer = optim.SparseAdam(model.parameters(), lr=args.lr)

    # Adamax: a variant of Adam based on the infinity norm, suitable for embeddings and sparse data
    SST_LR_SCALE = 5
    PARA_LR_SCALE = 1
    STS_LR_SCALE = 3
    sst_lr = SST_LR_SCALE * args.lr
    para_lr = PARA_LR_SCALE * args.lr
    sts_lr = STS_LR_SCALE * args.lr
    sst_optimizer = optim.Adamax(model.parameters(), lr=sst_lr, weight_decay=1e-5)   # SST needs larger learning rate
    para_optimizer = optim.Adamax(model.parameters(), lr=para_lr, weight_decay=1e-3)
    sts_optimizer = optim.RAdam(model.parameters(), lr=sts_lr, weight_decay=5e-4)
    
    CYCLE_LENGTH = args.epochs // 5  # Adjust cycle length based on your epochs
    
    
    if args.fine_tune_mode == "iterative":
        # Lambda lr scheduler halves the learning rate every epoch (and also when unfreezing)
        sst_scheduler = lr_scheduler.LambdaLR(sst_optimizer, lr_lambda=lambda epoch: 0.7 ** epoch)
        para_scheduler = lr_scheduler.LambdaLR(para_optimizer, lr_lambda=lambda epoch: 0.7 ** epoch)
        sts_scheduler = lr_scheduler.LambdaLR(sts_optimizer, lr_lambda=lambda epoch: 0.5 ** epoch)
        
        # Deleted cyclic LR it was trash
        
        # Cosine
        
        # sst_scheduler = lr_scheduler.CosineAnnealingLR(sst_optimizer, T_max=args.epochs, eta_min=sst_lr/ 5)
        # para_scheduler = lr_scheduler.CosineAnnealingLR(para_optimizer, T_max=args.epochs, eta_min=para_lr / 4)
        # sts_scheduler = lr_scheduler.CosineAnnealingLR(sts_optimizer, T_max=args.epochs, eta_min=sts_lr / 4)
    else:
        # Cosine Annealing
        sst_scheduler = lr_scheduler.CosineAnnealingLR(sst_optimizer, T_max=args.epochs, eta_min=sst_lr/ 5)
        para_scheduler = lr_scheduler.CosineAnnealingLR(para_optimizer, T_max=args.epochs, eta_min=para_lr / 4)
        sts_scheduler = lr_scheduler.CosineAnnealingLR(sts_optimizer, T_max=args.epochs, eta_min=sts_lr / 4)

        
    # =================================================
    log_dir = os.path.join("tensorboard_logs", f"{args.filepath.split('.')[0]}_{datetime.now()}")
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
        
        # ====================== Manage Iterative Unfreezing ==========================
        if args.fine_tune_mode == 'iterative':
            if args.iterative_quora:
                args.train_quora = False     # quora training doesn't work well without unfreezing
            checkpoint_slice = CYCLE_LENGTH
            if epoch < checkpoint_slice // 2:
                model.manage_freezing('freezeall')
                print(f"\n====== All BERT Layers Frozen ======\n")
            elif epoch < checkpoint_slice * 2:
                if args.iterative_quora:
                    args.train_quora = True
                model.manage_freezing('unfreezetop3')
                print(f"\n====== Top 3 BERT Layers Unfrozen ======\n")
            elif epoch < checkpoint_slice * 3:
                model.manage_freezing('unfreezetop6')
                print(f"\n====== Top 6 BERT Layers Unfrozen ======\n")
            elif epoch < checkpoint_slice * 4:
                model.manage_freezing('unfreezetop9')
                print(f"\n====== Top 9 BERT Layers Unfrozen ======\n")
            else:
                model.manage_freezing('unfreezeall')
                print(f"\n====== All BERT Layers Unfrozen ======\n")   # 12 total attn layers
        
        
        model.train()  # put model in training mode

        if benchmark:
            torch.cuda.reset_peak_memory_stats()
            
        global_batch_counter = 0  # Initialize the global batch counter
        total_global_batches = 0
        if args.train_sst:
            total_global_batches += len(sst_train_dataloader)
        if args.train_quora: 
            total_global_batches += len(para_train_dataloader)
        if args.train_sts:
            total_global_batches += len(sts_train_dataloader)

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
                
                # ======= various criterion options ==========
                criterion_ce = nn.CrossEntropyLoss(reduction='mean')  # Cross Entropy Loss
                criterion_bce = nn.BCEWithLogitsLoss(reduction='mean')  # Binary Cross Entropy Loss
                criterion_kl = nn.KLDivLoss(reduction='batchmean')  # Kullback-Leibler Divergence Loss
                criterion_ord = OrdinalCrossEntropyLoss()
                # criterion_cumlink = CumulativeLinkLoss() # trash

                # SELECT AN CRITERION MY GOOD SIR!
                criterion = criterion_bce
                

                sst_optimizer.zero_grad()

                if args.amp:  # auto mixed-precision
                    with torch.cuda.amp.autocast():
                        logits = model.predict_sentiment(b_ids, b_mask)
                        if criterion == criterion_ce:
                            loss = criterion(logits, b_labels.view(-1)) 
                        elif criterion == criterion_bce:
                            labels_one_hot = F.one_hot(b_labels, num_classes=5).float()
                            loss = criterion(logits, labels_one_hot)
                        elif criterion == criterion_kl:
                            labels_one_hot = F.one_hot(b_labels, num_classes=5).float()
                            log_probs = F.log_softmax(logits, dim=-1)
                            loss = criterion(log_probs, labels_one_hot) 
                        elif criterion == "mixed": 
                            # == setup ==
                            lambda_bce = 1
                            lambda_mse = 0.1
                            # ===========
                            labels_one_hot = F.one_hot(b_labels, num_classes=5).float()
                            loss_bce = nn.BCEWithLogitsLoss()(logits, labels_one_hot)
                            probabilities = F.softmax(logits, dim=1)

                            # calculate expected score over the softmax distribution
                            scores = torch.tensor([1, 2, 3, 4, 5], dtype=torch.float32, device=device).unsqueeze(0)  # (1, 5)
                            expected_scores = torch.matmul(probabilities, scores.T).squeeze(1)  # (N,)

                            # MSE between the expected scores and true scores
                            # print(expected_scores)
                            # print(b_labels)
                            loss_mse = nn.MSELoss()(expected_scores, b_labels.float())
                            loss = lambda_bce * loss_bce + lambda_mse * loss_mse
                            
                            
                        # elif criterion == criterion_ord:
                        #     loss = criterion(logits, b_labels) 
                        # elif criterion == criterion_cumlink:
                        #     loss = criterion(F.softmax(logits, dim = 1), F.one_hot(b_labels, num_classes = 5))
                            
                    gradscaler.scale(loss).backward()
                    gradscaler.step(sst_optimizer)
                    gradscaler.update()
                    if args.use_ema:
                        ema.update()
                else:  # vanilla
                    logits = model.predict_sentiment(b_ids, b_mask)
                    if criterion == criterion_ce:
                        loss = criterion(logits, b_labels.view(-1)) 
                    elif criterion == criterion_bce:
                        labels_one_hot = F.one_hot(b_labels, num_classes=5).float()
                        loss = criterion(logits, labels_one_hot) 
                    elif criterion == criterion_kl:
                        labels_one_hot = F.one_hot(b_labels, num_classes=5).float()
                        log_probs = F.log_softmax(logits, dim=-1)
                        loss = criterion(log_probs, labels_one_hot) 
                    
                    loss.backward()
                    sst_optimizer.step()
                    if args.use_ema:
                        ema.update()

                sst_train_loss += loss.item()
                sst_num_batches += 1
                
                if global_batch_counter % args.log_every == 0:
                    fractional_epoch = epoch + (global_batch_counter / total_global_batches)
                    writer.add_scalar('SST/Train_Loss', sst_train_loss / sst_num_batches, fractional_epoch)
            
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

                para_optimizer.zero_grad()

                if args.amp:  # auto multi-precision
                    with torch.cuda.amp.autocast():
                        logits = model.predict_paraphrase(b_ids1, b_mask1, b_ids2, b_mask2).flatten()
                        b_labels = b_labels.flatten().float()  # reshape to match logits
                        criterion = torch.nn.BCEWithLogitsLoss()  # more stable, can autocast
                        loss = criterion(logits, b_labels) 

                    gradscaler.scale(loss).backward()
                    gradscaler.step(para_optimizer)
                    gradscaler.update()
                    if args.use_ema:
                        ema.update()
                else:  # vanilla
                    logits = model.predict_paraphrase(b_ids1, b_mask1, b_ids2, b_mask2).flatten()
                    b_labels = b_labels.flatten().float()  # reshape to match logits
                    criterion = torch.nn.BCEWithLogitsLoss()  # more stable, can autocast
                    loss = criterion(logits, b_labels) 
                    loss.backward()
                    para_optimizer.step()
                    if args.use_ema:
                        ema.update()

                para_train_loss += loss.item()
                para_num_batches += 1
                
                if global_batch_counter % args.log_every == 0:
                    fractional_epoch = epoch + (global_batch_counter / total_global_batches)
                    writer.add_scalar('Quora/Train_Loss', para_train_loss / para_num_batches, fractional_epoch)

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

                sts_optimizer.zero_grad()

                # =========== Set Up STS Criterion ============
                criterion_mse = nn.MSELoss()  # MSE loss since labels are 0-5
                criterion_pearson = pearson_correlation_loss
                
                criterion = criterion_pearson

                if args.amp:  # auto multi-precision
                    with torch.cuda.amp.autocast():
                        logits = model.predict_similarity(b_ids1, b_mask1, b_ids2, b_mask2).flatten()
                        b_labels = b_labels.flatten().float()  # reshape to match logits
                        if criterion in [criterion_mse, criterion_pearson]:
                            loss = criterion(logits, b_labels) 
                        elif criterion == "mixed":
                            lambda_mse = 0.5 
                            lambda_pears = 1.0
                            mse_loss = criterion_mse(logits, b_labels) 
                            pears_loss = pearson_correlation_loss(logits, b_labels)
                            loss = lambda_mse * mse_loss + lambda_pears * pears_loss

                    gradscaler.scale(loss).backward()
                    gradscaler.step(sts_optimizer)
                    gradscaler.update()
                    if args.use_ema:
                        ema.update()
                else:  # vanilla
                    logits = model.predict_similarity(b_ids1, b_mask1, b_ids2, b_mask2).flatten()
                    b_labels = b_labels.flatten().float()  # reshape to match logits
                    if criterion in [criterion_mse, criterion_pearson]:
                        loss = criterion(logits, b_labels) 
                    elif criterion == "mixed":
                        lambda_mse = 0.5 
                        lambda_pears = 1.0
                        mse_loss = nn.MSELoss(logits, b_labels) 
                        pears_loss = pearson_correlation_loss(logits)
                        loss = lambda_mse * mse_loss + lambda_pears * pears_loss
                    loss.backward()
                    sts_optimizer.step()
                    if args.use_ema:
                        ema.update()

                sts_train_loss += loss.item()
                sts_num_batches += 1
                
                if global_batch_counter % args.log_every == 0:
                    fractional_epoch = epoch + (global_batch_counter / total_global_batches)
                    writer.add_scalar('STS/Train_Loss', sts_train_loss / sts_num_batches, fractional_epoch)

            if benchmark:
                total_sts_time += time.time() - start_time
                peak_memory_sts = torch.cuda.max_memory_allocated() / (1024 ** 3)  # Convert bytes to GB
                total_sts_memory += peak_memory_sts
            sts_train_loss = sts_train_loss / sts_num_batches
    
        if args.use_ema:
            print("=== Averaging the EMA model and Updating BNs ====")
            ema_model = ema.apply_shadow()
            dataloaders = [sst_train_dataloader, para_train_dataloader, sts_train_dataloader]
            # Update batch norm statistics for each dataloader
            for dataloader in dataloaders:
                torch.optim.swa_utils.update_bn(dataloader, ema_model)
            print("=== Swapping in EMA Model for Eval ====")
            og_model = model
            model = ema_model
            
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
        if dev_leaderboard_score > best_leaderboard_score:  
            best_leaderboard_score = dev_leaderboard_score
            print(f"New Best Leaderboard Avg. Score: {best_leaderboard_score:.3f}!")
            save_model(model, 
                       (sst_optimizer, para_optimizer, sts_optimizer), 
                       args, 
                       config, 
                       args.filepath)
        if args.use_ema:
            print("=== Swapping OG Model Back In Place of EMA for Training ====")
            model = og_model

        # ========== step the learning rate schedulers ==========
        sst_lr = sst_scheduler.get_last_lr()[0] 
        para_lr = para_scheduler.get_last_lr()[0]
        sts_lr = sts_scheduler.get_last_lr()[0]
        
        # print(f"\nLRs: {sst_lr}, {para_lr}, {sts_lr}")
        
        # Record the learning rates
        writer.add_scalar('Learning_Rate/sst_lr', sst_lr, epoch)
        writer.add_scalar('Learning_Rate/para_lr', para_lr, epoch )
        writer.add_scalar('Learning_Rate/sts_lr', sts_lr, epoch )
        sst_scheduler.step()
        if args.train_quora:
            para_scheduler.step()
        sts_scheduler.step()
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
            load_multitask_data(args.sst_test,args.para_test, args.sts_test, split='test', extra_clean = args.extra_clean)

        sst_dev_data, num_labels,para_dev_data, sts_dev_data = \
            load_multitask_data(args.sst_dev,args.para_dev,args.sts_dev,split='dev', extra_clean = args.extra_clean)

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
    parser.add_argument("--amp",  action='store_true', help='Turn on auto multi-precision for training with float16')

    # clf type (linear, nonlinear, conv)
    parser.add_argument("--clf", type=str, default="linear", 
                        help="Type of classifier layer. Supported: linear, nonlinear, or conv.",
                        choices=("linear", "nonlinear", "conv"))

    # dataset extra cleaning with nltk
    parser.add_argument("--extra_clean", action = 'store_true', default = False, help = "Perform extra nltk etc. data cleaning.")

    # Exponential Moving Average (EMA) model
    parser.add_argument("--use_ema", action = 'store_true', default = False, help="Use an Exponential Moving Average (EMA) model framework.")
    
    # number of backbone BERTS
    parser.add_argument("--num_berts", type=int, default=1, help = "Number of BERT backbones for the model. ")
    
    parser.add_argument("--log_every", type = int, default = 50, help = "Every # of batches to log train stats.")
    
    # lr scheduling 
    #TODO

    # dataset selections for training
    parser.add_argument("--train_sst", action='store_true', help='Train against SST sentiment dataset (CELoss)')
    parser.add_argument("--train_quora", action='store_true', help='Train against Quora paraphrase dataset (BCELoss) (costly!)')
    parser.add_argument("--train_sts", action='store_true', help='Train against STS similarity dataset (BCELoss)')
    parser.add_argument("--iterative_quora", action = 'store_true', default = False, help = "Don't train quora until unfreezing starts." )
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
                        help=('last-layer: the BERT parameters are frozen and the task specific head parameters are updated;'
                              'full-model: BERT parameters are updated as well'
                              'iterative: BERT parameters are progressively unfrozen more deeply throughout training'),
                        choices=('last-layer', 'full-model', 'iterative'), default="last-layer")
    parser.add_argument("--use_gpu", action='store_true')

    parser.add_argument("--sst_dev_out", type=str, default="predictions/sst-dev-output.csv")
    parser.add_argument("--sst_test_out", type=str, default="predictions/sst-test-output.csv")

    parser.add_argument("--para_dev_out", type=str, default="predictions/para-dev-output.csv")
    parser.add_argument("--para_test_out", type=str, default="predictions/para-test-output.csv")

    parser.add_argument("--sts_dev_out", type=str, default="predictions/sts-dev-output.csv")
    parser.add_argument("--sts_test_out", type=str, default="predictions/sts-test-output.csv")

    parser.add_argument("--batch_size", help='sst: 64, cfimdb: 8 can fit a 12GB GPU', type=int, default=8)
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.3)
    parser.add_argument("--lr", type=float, help="Initial learning rate", default=1e-5)

    parser.add_argument("--lora_dict", type=str, default='{"mode": "none", "r": 0, "dora": 0}')
    parser.add_argument("--benchmark-results", type=str, default="benchmark-results-dora-last-layer.csv")

    parser.add_argument("--benchmark", action='store_true', help='Benchmark the model for training time and memory usage')
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
    # seed_everything(args.seed)  # Fix the seed for reproducibility.
    if args.benchmark:
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
    else:
        train_multitask(args)
        test_multitask(args)

# duality of robust men (RAdam, 5x instead of 10x LR for SST), 1e-4 weight decay for para
python multitask_classifier.py --fine-tune-mode iterative --optim RAdam --sst_lr_multiplier 5  --para_weight_decay 1e-4 --lr 1e-4 --use_gpu --amp --batch_size 64 --train_sst --train_quora --train_sts  --clf conv --epochs 10

# duality of faster men - same as above, higher LR
python multitask_classifier.py --fine-tune-mode iterative --sst_lr_multiplier 5 --para_weight_decay 1e-4 --lr 2.1e-4 --use_gpu --amp --batch_size 64 --train_sst --train_quora --train_sts  --clf conv

# duality of slow men 
python multitask_classifier.py --fine-tune-mode full-model --sst_lr_multiplier 5 --para_weight_decay 1e-4 --lr 5e-5 --use_gpu --amp --batch_size 64 --train_sst --train_quora --train_sts  --clf conv

# duality of nonlinear men
python multitask_classifier.py --fine-tune-mode iterative --sst_lr_multiplier 5 --para_weight_decay 1e-4 --lr 1e-4 --use_gpu --amp --batch_size 64 --train_sst --train_quora --train_sts  --clf nonlinear

# mega duality (20 epochs)
python multitask_classifier.py --fine-tune-mode iterative --sst_lr_multiplier 5 --para_weight_decay 1e-4 --lr 1e-4 --use_gpu --amp --batch_size 64 --train_sst --train_quora --train_sts  --clf conv --epochs 20

# need this in multitask bert 
# def extract_comparison_features(self,
#                            input_ids_1, attention_mask_1,
#                            input_ids_2, attention_mask_2):
#         """ 
#         Given a batch of pairs of sentences, extract comparison features for Quora and STS tasks.
#         
#         Useful for both Sim tasks.
#         """
#         
#         embeds_1 = self.forward(input_ids_1, attention_mask_1, which_bert = "sim")
#         embeds_2 = self.forward(input_ids_2, attention_mask_2, which_bert = "sim")
#         cosine_sim = F.cosine_similarity(embeds_1, embeds_2).unsqueeze(-1)
#         elem_prods = embeds_1 * embeds_2
#         diff = torch.abs(embeds_1 - embeds_2)
#         features = torch.cat([diff, elem_prods, cosine_sim], dim=1)
#         features = self.comparison_features_fcn(features)
#         
#         return features

python ensemble.py --filepaths g6-duality-of-man.pt duality-of-fast-man.pt duality-of-long-distance-runners-20.pt duality-of-man-nonlinear.pt duality-of-slow-man.pt duality-of-robust-men.pt
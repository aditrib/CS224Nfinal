
# G6 yin yang
python multitask_classifier.py --fine-tune-mode iterative --lr 1e-4 --use_gpu --amp --batch_size 64 --train_sst --train_quora --train_sts --clf conv --sst_weight_decay 5e-3 --para_weight_decay 1e-5 --sts_weight_decay 8e-3 --lr_lambda 0.5 --optim Adamax --sst_lr_multiplier 10 --para_lr_multiplier 1.0 --sts_lr_multiplier 3

# G6 yang yin 
python multitask_classifier.py --fine-tune-mode iterative --lr 1e-4 --use_gpu --amp --batch_size 64 --train_sst --train_quora --train_sts --clf conv --sst_weight_decay 5e-3 --para_weight_decay 1e-5 --sts_weight_decay 8e-3 --lr_lambda 0.5 --optim Adamax --sst_lr_multiplier 3 --para_lr_multiplier 4 --sts_lr_multiplier 1

# G6 yin and the yang
python multitask_classifier.py --fine-tune-mode full-model --lr 1e-4 --use_gpu --amp --batch_size 64 --train_sst --train_quora --train_sts --clf conv --sst_weight_decay 5e-3 --para_weight_decay 1e-5 --sts_weight_decay 8e-3 --lr_lambda 0.5 --optim Adamax --sst_lr_multiplier 10 --para_lr_multiplier 1.0 --sts_lr_multiplier 3

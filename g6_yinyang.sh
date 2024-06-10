#v1 used 5e-3 sst decay, 8e-3 sts decay0.5 lambda sst multiplier 7
# G6 yin yang
# python multitask_classifier.py --fine-tune-mode iterative --lr 1e-4 --use_gpu --amp --batch_size 64 --train_sst --train_quora --train_sts --clf conv --sst_weight_decay 8e-3 --para_weight_decay 1e-5 --sts_weight_decay 9e-3 --lr_lambda 0.55 --optim Adamax --sst_lr_multiplier 7 --para_lr_multiplier 1.0 --sts_lr_multiplier 3 --epochs 7

# G6 yang yin 
# python multitask_classifier.py --fine-tune-mode iterative --lr 1e-4 --use_gpu --amp --batch_size 64 --train_sst --train_quora --train_sts --clf conv --sst_weight_decay 8e-3 --para_weight_decay 1e-5 --sts_weight_decay 9e-3 --lr_lambda 0.55 --optim Adamax --sst_lr_multiplier 3 --para_lr_multiplier 4 --sts_lr_multiplier 1 --epochs 7

# G6 yin yang fast
# python multitask_classifier.py --fine-tune-mode iterative --lr 1.5e-4 --use_gpu --amp --batch_size 64 --train_sst --train_quora --train_sts --clf conv --sst_weight_decay 8e-3 --para_weight_decay 1e-5 --sts_weight_decay 9e-3 --lr_lambda 0.55 --optim Adamax --sst_lr_multiplier 5 --para_lr_multiplier 5 --sts_lr_multiplier 2 --epochs 7

# G6 yin yang steady hand
# python multitask_classifier.py --fine-tune-mode iterative --lr 1e-4 --use_gpu --amp --batch_size 64 --train_sst --train_quora --train_sts --clf conv --sst_weight_decay 9e-3 --para_weight_decay 1e-5 --sts_weight_decay 1e-2 --lr_lambda 0.5 --optim Adamax --sst_lr_multiplier 4 --para_lr_multiplier 5 --sts_lr_multiplier 3 --epochs 7

# G6 yang yin the parabola
# python multitask_classifier.py --fine-tune-mode iterative --lr 1e-4 --use_gpu --amp --batch_size 64 --train_sst --train_quora --train_sts --clf conv --sst_weight_decay 9e-3 --para_weight_decay 1e-5 --sts_weight_decay 1e-2 --lr_lambda 0.5 --optim Adamax --sst_lr_multiplier 3 --para_lr_multiplier 8 --sts_lr_multiplier 1 --epochs 7

# G6 yin yang 2
# python multitask_classifier.py --fine-tune-mode iterative --lr 1e-4 --use_gpu --amp --batch_size 64 --train_sst --train_quora --train_sts --clf conv --sst_weight_decay 9e-3 --para_weight_decay 1e-5 --sts_weight_decay 1e-2 --lr_lambda 0.5 --optim Adamax --sst_lr_multiplier 4 --para_lr_multiplier 1.0 --sts_lr_multiplier 3 --epochs 7

# ensemble 
# python ensemble.py --filepaths g6-yin-yang.pt g6-yang-yin.pt g6-yin-yang-fast.pt g6-yin-yang-steady-hand.pt g6-yang-yin-the-parabola.pt g6-yin-yang-2.pt

# ===== more models based primarily off steady hand with minor variations ===
# G6 yin yang steady hand 2
python multitask_classifier.py --fine-tune-mode iterative --lr 1e-4 --use_gpu --amp --batch_size 64 --train_sst --train_quora --train_sts --clf conv --sst_weight_decay 9e-3 --para_weight_decay 1e-5 --sts_weight_decay 1e-2 --lr_lambda 0.5 --optim Adamax --sst_lr_multiplier 3 --para_lr_multiplier 4 --sts_lr_multiplier 2 --epochs 7

# G6 yin yang steady hand 3
python multitask_classifier.py --fine-tune-mode iterative --lr 1e-4 --use_gpu --amp --batch_size 64 --train_sst --train_quora --train_sts --clf conv --sst_weight_decay 9e-3 --para_weight_decay 1e-5 --sts_weight_decay 1e-2 --lr_lambda 0.55 --optim Adamax --sst_lr_multiplier 4 --para_lr_multiplier 5 --sts_lr_multiplier 3 --epochs 7

# G6 yin yang steady hand 4
python multitask_classifier.py --fine-tune-mode iterative --lr 1e-4 --use_gpu --amp --batch_size 64 --train_sst --train_quora --train_sts --clf conv --sst_weight_decay 9e-3 --para_weight_decay 1e-5 --sts_weight_decay 1e-2 --lr_lambda 0.6 --optim Adamax --sst_lr_multiplier 2 --para_lr_multiplier 3 --sts_lr_multiplier 2 --epochs 7

# G6 yin yang steady hand 5
python multitask_classifier.py --fine-tune-mode iterative --lr 1e-4 --use_gpu --amp --batch_size 64 --train_sst --train_quora --train_sts --clf conv --sst_weight_decay 9e-3 --para_weight_decay 1e-5 --sts_weight_decay 1e-2 --lr_lambda 0.5 --optim Adamax --sst_lr_multiplier 5 --para_lr_multiplier 6 --sts_lr_multiplier 4 --epochs 7

# G6 yin yang steady hand 6
python multitask_classifier.py --fine-tune-mode iterative --lr 1e-4 --use_gpu --amp --batch_size 64 --train_sst --train_quora --train_sts --clf conv --sst_weight_decay 9e-3 --para_weight_decay 1e-5 --sts_weight_decay 1e-2 --lr_lambda 0.5 --optim Adamax --sst_lr_multiplier 6 --para_lr_multiplier 7 --sts_lr_multiplier 5 --epochs 7

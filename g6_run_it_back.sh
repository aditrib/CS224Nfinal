# G6 run it back 
# python multitask_classifier.py --fine-tune-mode iterative --lr 1e-4 --use_gpu --amp --batch_size 64 --train_sst --train_quora --train_sts --clf conv --sst_weight_decay 8e-3 --para_weight_decay 1e-5 --sts_weight_decay 9e-3 --lr_lambda 0.55 --optim Adamax --sst_lr_multiplier 4 --para_lr_multiplier 1.0 --sts_lr_multiplier 3 --epochs 7 --num_sst_trains 2 --num_quora_trains 1 --num_sts_trains 5 --sst_weight_decay 9e-3 --sts_weight_decay 1e-2

# G6 run it back more para 
python multitask_classifier.py --fine-tune-mode iterative --lr 1e-4 --use_gpu --amp --batch_size 64 --train_sst --train_quora --train_sts --clf conv --sst_weight_decay 8e-3 --para_weight_decay 1e-5 --sts_weight_decay 9e-3 --lr_lambda 0.55 --optim Adamax --sst_lr_multiplier 3 --para_lr_multiplier 4 --sts_lr_multiplier 2 --epochs 7 --num_sst_trains 2 --num_quora_trains 1 --num_sts_trains 2 --sst_weight_decay 9e-3 --sts_weight_decay 1e-2

# G6 run it back fast 
python multitask_classifier.py --fine-tune-mode iterative --lr 1.5e-4 --use_gpu --amp --batch_size 64 --train_sst --train_quora --train_sts --clf conv --sst_weight_decay 8e-3 --para_weight_decay 1e-5 --sts_weight_decay 9e-3 --lr_lambda 0.55 --optim Adamax --sst_lr_multiplier 5 --para_lr_multiplier 5 --sts_lr_multiplier 2 --epochs 7 --num_sst_trains 2 --num_quora_trains 1 --num_sts_trains 3 --sst_weight_decay 9e-3 --sts_weight_decay 1e-2

# G6 run it back steady hand 
python multitask_classifier.py --fine-tune-mode iterative --lr 1e-4 --use_gpu --amp --batch_size 64 --train_sst --train_quora --train_sts --clf conv --sst_weight_decay 9e-3 --para_weight_decay 1e-5 --sts_weight_decay 1e-2 --lr_lambda 0.5 --optim Adamax --sst_lr_multiplier 4 --para_lr_multiplier 5 --sts_lr_multiplier 3 --epochs 7 --num_sst_trains 1 --num_quora_trains 1 --num_sts_trains 5 --sst_weight_decay 9e-3 --sts_weight_decay 1e-2

# G6 run it back the parabola r
python multitask_classifier.py --fine-tune-mode iterative --lr 1e-4 --use_gpu --amp --batch_size 64 --train_sst --train_quora --train_sts --clf conv --sst_weight_decay 9e-3 --para_weight_decay 1e-5 --sts_weight_decay 1e-2 --lr_lambda 0.5 --optim Adamax --sst_lr_multiplier 3 --para_lr_multiplier 8 --sts_lr_multiplier 2 --epochs 7 --num_sst_trains 2 --num_quora_trains 2 --num_sts_trains 2 --sst_weight_decay 9e-3 --sts_weight_decay 1e-2

# G6 run it back 2 
python multitask_classifier.py --fine-tune-mode iterative --lr 1e-4 --use_gpu --amp --batch_size 64 --train_sst --train_quora --train_sts --clf conv --sst_weight_decay 9e-3 --para_weight_decay 1e-5 --sts_weight_decay 1e-2 --lr_lambda 0.5 --optim Adamax --sst_lr_multiplier 4 --para_lr_multiplier 2.0 --sts_lr_multiplier 4 --epochs 7 --num_sst_trains 1 --num_quora_trains 1 --num_sts_trains 10 --sst_weight_decay 9e-3 --sts_weight_decay 1e-2

# ensemble 
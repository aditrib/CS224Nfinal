# lr 1e-5 bs 386
python3 multitask_classifier.py --amp --benchmark --lr 1e-5 --lora_dict '{"mode":"attn-only", "r":1, "dora":1}' --fine-tune-mode full-model --use_gpu --train_sst --train_sts --epochs 3 --batch_size 384
python3 multitask_classifier.py --amp --benchmark --lr 1e-5 --lora_dict '{"mode":"attn-only", "r":5, "dora":1}' --fine-tune-mode full-model --use_gpu --train_sst --train_sts --epochs 3 --batch_size 384
python3 multitask_classifier.py --amp --benchmark --lr 1e-5 --lora_dict '{"mode":"attn-only", "r":10, "dora":1}' --fine-tune-mode full-model --use_gpu --train_sst --train_sts --epochs 3 --batch_size 384
--benchmark --amp 
python3 multitask_classifier.py --amp --benchmark --lr 1e-5 --lora_dict '{"mode":"attn", "r":1, "dora":1}' --fine-tune-mode full-model --use_gpu --train_sst --train_sts --epochs 3 --batch_size 384
python3 multitask_classifier.py --amp --benchmark --lr 1e-5 --lora_dict '{"mode":"attn", "r":5, "dora":1}' --fine-tune-mode full-model --use_gpu --train_sst --train_sts --epochs 3 --batch_size 384
python3 multitask_classifier.py --amp --benchmark --lr 1e-5 --lora_dict '{"mode":"attn", "r":10, "dora":1}' --fine-tune-mode full-model --use_gpu --train_sst --train_sts --epochs 3 --batch_size 384
--benchmark --amp 
python3 multitask_classifier.py --amp --benchmark --lr 1e-5 --lora_dict '{"mode":"all-lin-only", "r":1, "dora":1}' --fine-tune-mode full-model --use_gpu --train_sst --train_sts --epochs 3 --batch_size 384
python3 multitask_classifier.py --amp --benchmark --lr 1e-5 --lora_dict '{"mode":"all-lin-only", "r":5, "dora":1}' --fine-tune-mode full-model --use_gpu --train_sst --train_sts --epochs 3 --batch_size 384
python3 multitask_classifier.py --amp --benchmark --lr 1e-5 --lora_dict '{"mode":"all-lin-only", "r":10, "dora":1}' --fine-tune-mode full-model --use_gpu --train_sst --train_sts --epochs 3 --batch_size 384
--benchmark --amp 
python3 multitask_classifier.py --amp --benchmark --lr 1e-5 --lora_dict '{"mode":"all-lin", "r":1, "dora":1}' --fine-tune-mode full-model --use_gpu --train_sst --train_sts --epochs 3 --batch_size 384
python3 multitask_classifier.py --amp --benchmark --lr 1e-5 --lora_dict '{"mode":"all-lin", "r":5, "dora":1}' --fine-tune-mode full-model --use_gpu --train_sst --train_sts --epochs 3 --batch_size 384
python3 multitask_classifier.py --amp --benchmark --lr 1e-5 --lora_dict '{"mode":"all-lin", "r":10, "dora":1}' --fine-tune-mode full-model --use_gpu --train_sst --train_sts --epochs 3 --batch_size 384
--benchmark --amp 
# lr 1e-5 bs 256--benchmark --amp 
python3 multitask_classifier.py --amp --benchmark --lr 1e-5 --lora_dict '{"mode":"attn-only", "r":1, "dora":1}' --fine-tune-mode full-model --use_gpu --train_sst --train_sts --epochs 3 --batch_size 256
python3 multitask_classifier.py --amp --benchmark --lr 1e-5 --lora_dict '{"mode":"attn-only", "r":5, "dora":1}' --fine-tune-mode full-model --use_gpu --train_sst --train_sts --epochs 3 --batch_size 256
python3 multitask_classifier.py --amp --benchmark --lr 1e-5 --lora_dict '{"mode":"attn-only", "r":10, "dora":1}' --fine-tune-mode full-model --use_gpu --train_sst --train_sts --epochs 3 --batch_size 256
--benchmark --amp 
python3 multitask_classifier.py --amp --benchmark --lr 1e-5 --lora_dict '{"mode":"attn", "r":1, "dora":1}' --fine-tune-mode full-model --use_gpu --train_sst --train_sts --epochs 3 --batch_size 256
python3 multitask_classifier.py --amp --benchmark --lr 1e-5 --lora_dict '{"mode":"attn", "r":5, "dora":1}' --fine-tune-mode full-model --use_gpu --train_sst --train_sts --epochs 3 --batch_size 256
python3 multitask_classifier.py --amp --benchmark --lr 1e-5 --lora_dict '{"mode":"attn", "r":10, "dora":1}' --fine-tune-mode full-model --use_gpu --train_sst --train_sts --epochs 3 --batch_size 256
--benchmark --amp 
python3 multitask_classifier.py --amp --benchmark --lr 1e-5 --lora_dict '{"mode":"all-lin-only", "r":1, "dora":1}' --fine-tune-mode full-model --use_gpu --train_sst --train_sts --epochs 3 --batch_size 256
python3 multitask_classifier.py --amp --benchmark --lr 1e-5 --lora_dict '{"mode":"all-lin-only", "r":5, "dora":1}' --fine-tune-mode full-model --use_gpu --train_sst --train_sts --epochs 3 --batch_size 256
python3 multitask_classifier.py --amp --benchmark --lr 1e-5 --lora_dict '{"mode":"all-lin-only", "r":10, "dora":1}' --fine-tune-mode full-model --use_gpu --train_sst --train_sts --epochs 3 --batch_size 256
--benchmark --amp 
python3 multitask_classifier.py --amp --benchmark --lr 1e-5 --lora_dict '{"mode":"all-lin", "r":1, "dora":1}' --fine-tune-mode full-model --use_gpu --train_sst --train_sts --epochs 3 --batch_size 256
python3 multitask_classifier.py --amp --benchmark --lr 1e-5 --lora_dict '{"mode":"all-lin", "r":5, "dora":1}' --fine-tune-mode full-model --use_gpu --train_sst --train_sts --epochs 3 --batch_size 256
python3 multitask_classifier.py --amp --benchmark --lr 1e-5 --lora_dict '{"mode":"all-lin", "r":10, "dora":1}' --fine-tune-mode full-model --use_gpu --train_sst --train_sts --epochs 3 --batch_size 256
--benchmark --amp 
# lr 1e-5 bs 128--benchmark --amp 
python3 multitask_classifier.py --amp --benchmark --lr 1e-5 --lora_dict '{"mode":"attn-only", "r":1, "dora":1}' --fine-tune-mode full-model --use_gpu --train_sst --train_sts --epochs 3 --batch_size 128
python3 multitask_classifier.py --amp --benchmark --lr 1e-5 --lora_dict '{"mode":"attn-only", "r":5, "dora":1}' --fine-tune-mode full-model --use_gpu --train_sst --train_sts --epochs 3 --batch_size 128
python3 multitask_classifier.py --amp --benchmark --lr 1e-5 --lora_dict '{"mode":"attn-only", "r":10, "dora":1}' --fine-tune-mode full-model --use_gpu --train_sst --train_sts --epochs 3 --batch_size 128
--benchmark --amp 
python3 multitask_classifier.py --amp --benchmark --lr 1e-5 --lora_dict '{"mode":"attn", "r":1, "dora":1}' --fine-tune-mode full-model --use_gpu --train_sst --train_sts --epochs 3 --batch_size 128
python3 multitask_classifier.py --amp --benchmark --lr 1e-5 --lora_dict '{"mode":"attn", "r":5, "dora":1}' --fine-tune-mode full-model --use_gpu --train_sst --train_sts --epochs 3 --batch_size 128
python3 multitask_classifier.py --amp --benchmark --lr 1e-5 --lora_dict '{"mode":"attn", "r":10, "dora":1}' --fine-tune-mode full-model --use_gpu --train_sst --train_sts --epochs 3 --batch_size 128
--benchmark --amp 
python3 multitask_classifier.py --amp --benchmark --lr 1e-5 --lora_dict '{"mode":"all-lin-only", "r":1, "dora":1}' --fine-tune-mode full-model --use_gpu --train_sst --train_sts --epochs 3 --batch_size 128
python3 multitask_classifier.py --amp --benchmark --lr 1e-5 --lora_dict '{"mode":"all-lin-only", "r":5, "dora":1}' --fine-tune-mode full-model --use_gpu --train_sst --train_sts --epochs 3 --batch_size 128
python3 multitask_classifier.py --amp --benchmark --lr 1e-5 --lora_dict '{"mode":"all-lin-only", "r":10, "dora":1}' --fine-tune-mode full-model --use_gpu --train_sst --train_sts --epochs 3 --batch_size 128
--benchmark --amp 
python3 multitask_classifier.py --amp --benchmark --lr 1e-5 --lora_dict '{"mode":"all-lin", "r":1, "dora":1}' --fine-tune-mode full-model --use_gpu --train_sst --train_sts --epochs 3 --batch_size 128
python3 multitask_classifier.py --amp --benchmark --lr 1e-5 --lora_dict '{"mode":"all-lin", "r":5, "dora":1}' --fine-tune-mode full-model --use_gpu --train_sst --train_sts --epochs 3 --batch_size 128
python3 multitask_classifier.py --amp --benchmark --lr 1e-5 --lora_dict '{"mode":"all-lin", "r":10, "dora":1}' --fine-tune-mode full-model --use_gpu --train_sst --train_sts --epochs 3 --batch_size 128
--benchmark --amp 
# lr 1e-5 bs 64--benchmark --amp 
python3 multitask_classifier.py --amp --benchmark --lr 1e-5 --lora_dict '{"mode":"attn-only", "r":1, "dora":1}' --fine-tune-mode full-model --use_gpu --train_sst --train_sts --epochs 3 --batch_size 64
python3 multitask_classifier.py --amp --benchmark --lr 1e-5 --lora_dict '{"mode":"attn-only", "r":5, "dora":1}' --fine-tune-mode full-model --use_gpu --train_sst --train_sts --epochs 3 --batch_size 64
python3 multitask_classifier.py --amp --benchmark --lr 1e-5 --lora_dict '{"mode":"attn-only", "r":10, "dora":1}' --fine-tune-mode full-model --use_gpu --train_sst --train_sts --epochs 3 --batch_size 64
--benchmark --amp 
python3 multitask_classifier.py --amp --benchmark --lr 1e-5 --lora_dict '{"mode":"attn", "r":1, "dora":1}' --fine-tune-mode full-model --use_gpu --train_sst --train_sts --epochs 3 --batch_size 64
python3 multitask_classifier.py --amp --benchmark --lr 1e-5 --lora_dict '{"mode":"attn", "r":5, "dora":1}' --fine-tune-mode full-model --use_gpu --train_sst --train_sts --epochs 3 --batch_size 64
python3 multitask_classifier.py --amp --benchmark --lr 1e-5 --lora_dict '{"mode":"attn", "r":10, "dora":1}' --fine-tune-mode full-model --use_gpu --train_sst --train_sts --epochs 3 --batch_size 64
--benchmark --amp 
python3 multitask_classifier.py --amp --benchmark --lr 1e-5 --lora_dict '{"mode":"all-lin-only", "r":1, "dora":1}' --fine-tune-mode full-model --use_gpu --train_sst --train_sts --epochs 3 --batch_size 64
python3 multitask_classifier.py --amp --benchmark --lr 1e-5 --lora_dict '{"mode":"all-lin-only", "r":5, "dora":1}' --fine-tune-mode full-model --use_gpu --train_sst --train_sts --epochs 3 --batch_size 64
python3 multitask_classifier.py --amp --benchmark --lr 1e-5 --lora_dict '{"mode":"all-lin-only", "r":10, "dora":1}' --fine-tune-mode full-model --use_gpu --train_sst --train_sts --epochs 3 --batch_size 64
--benchmark --amp 
python3 multitask_classifier.py --amp --benchmark --lr 1e-5 --lora_dict '{"mode":"all-lin", "r":1, "dora":1}' --fine-tune-mode full-model --use_gpu --train_sst --train_sts --epochs 3 --batch_size 64
python3 multitask_classifier.py --amp --benchmark --lr 1e-5 --lora_dict '{"mode":"all-lin", "r":5, "dora":1}' --fine-tune-mode full-model --use_gpu --train_sst --train_sts --epochs 3 --batch_size 64
python3 multitask_classifier.py --amp --benchmark --lr 1e-5 --lora_dict '{"mode":"all-lin", "r":10, "dora":1}' --fine-tune-mode full-model --use_gpu --train_sst --train_sts --epochs 3 --batch_size 64
--benchmark --amp 
# lr 1e-4 bs 386--benchmark --amp 
python3 multitask_classifier.py --amp --benchmark --lr 1e-4 --lora_dict '{"mode":"attn-only", "r":1, "dora":1}' --fine-tune-mode full-model --use_gpu --train_sst --train_sts --epochs 3 --batch_size 384
python3 multitask_classifier.py --amp --benchmark --lr 1e-4 --lora_dict '{"mode":"attn-only", "r":5, "dora":1}' --fine-tune-mode full-model --use_gpu --train_sst --train_sts --epochs 3 --batch_size 384
python3 multitask_classifier.py --amp --benchmark --lr 1e-4 --lora_dict '{"mode":"attn-only", "r":10, "dora":1}' --fine-tune-mode full-model --use_gpu --train_sst --train_sts --epochs 3 --batch_size 3844
--benchmark --amp 
python3 multitask_classifier.py --amp --benchmark --lr 1e-4 --lora_dict '{"mode":"attn", "r":1, "dora":1}' --fine-tune-mode full-model --use_gpu --train_sst --train_sts --epochs 3 --batch_size 384
python3 multitask_classifier.py --amp --benchmark --lr 1e-4 --lora_dict '{"mode":"attn", "r":5, "dora":1}' --fine-tune-mode full-model --use_gpu --train_sst --train_sts --epochs 3 --batch_size 384
python3 multitask_classifier.py --amp --benchmark --lr 1e-4 --lora_dict '{"mode":"attn", "r":10, "dora":1}' --fine-tune-mode full-model --use_gpu --train_sst --train_sts --epochs 3 --batch_size 3844
--benchmark --amp 
python3 multitask_classifier.py --amp --benchmark --lr 1e-4 --lora_dict '{"mode":"all-lin-only", "r":1, "dora":1}' --fine-tune-mode full-model --use_gpu --train_sst --train_sts --epochs 3 --batch_size 384
python3 multitask_classifier.py --amp --benchmark --lr 1e-4 --lora_dict '{"mode":"all-lin-only", "r":5, "dora":1}' --fine-tune-mode full-model --use_gpu --train_sst --train_sts --epochs 3 --batch_size 384
python3 multitask_classifier.py --amp --benchmark --lr 1e-4 --lora_dict '{"mode":"all-lin-only", "r":10, "dora":1}' --fine-tune-mode full-model --use_gpu --train_sst --train_sts --epochs 3 --batch_size 3844
--benchmark --amp 
python3 multitask_classifier.py --amp --benchmark --lr 1e-4 --lora_dict '{"mode":"all-lin", "r":1, "dora":1}' --fine-tune-mode full-model --use_gpu --train_sst --train_sts --epochs 3 --batch_size 384
python3 multitask_classifier.py --amp --benchmark --lr 1e-4 --lora_dict '{"mode":"all-lin", "r":5, "dora":1}' --fine-tune-mode full-model --use_gpu --train_sst --train_sts --epochs 3 --batch_size 384
python3 multitask_classifier.py --amp --benchmark --lr 1e-4 --lora_dict '{"mode":"all-lin", "r":10, "dora":1}' --fine-tune-mode full-model --use_gpu --train_sst --train_sts --epochs 3 --batch_size 3844
# lr 1e-4 bs 254--benchmark --amp 
python3 multitask_classifier.py --amp --benchmark --lr 1e-4 --lora_dict '{"mode":"attn-only", "r":1, "dora":1}' --fine-tune-mode full-model --use_gpu --train_sst --train_sts --epochs 3 --batch_size 256
python3 multitask_classifier.py --amp --benchmark --lr 1e-4 --lora_dict '{"mode":"attn-only", "r":5, "dora":1}' --fine-tune-mode full-model --use_gpu --train_sst --train_sts --epochs 3 --batch_size 256
python3 multitask_classifier.py --amp --benchmark --lr 1e-4 --lora_dict '{"mode":"attn-only", "r":10, "dora":1}' --fine-tune-mode full-model --use_gpu --train_sst --train_sts --epochs 3 --batch_size 2564
--benchmark --amp 
python3 multitask_classifier.py --amp --benchmark --lr 1e-4 --lora_dict '{"mode":"attn", "r":1, "dora":1}' --fine-tune-mode full-model --use_gpu --train_sst --train_sts --epochs 3 --batch_size 256
python3 multitask_classifier.py --amp --benchmark --lr 1e-4 --lora_dict '{"mode":"attn", "r":5, "dora":1}' --fine-tune-mode full-model --use_gpu --train_sst --train_sts --epochs 3 --batch_size 256
python3 multitask_classifier.py --amp --benchmark --lr 1e-4 --lora_dict '{"mode":"attn", "r":10, "dora":1}' --fine-tune-mode full-model --use_gpu --train_sst --train_sts --epochs 3 --batch_size 2564
--benchmark --amp 
python3 multitask_classifier.py --amp --benchmark --lr 1e-4 --lora_dict '{"mode":"all-lin-only", "r":1, "dora":1}' --fine-tune-mode full-model --use_gpu --train_sst --train_sts --epochs 3 --batch_size 256
python3 multitask_classifier.py --amp --benchmark --lr 1e-4 --lora_dict '{"mode":"all-lin-only", "r":5, "dora":1}' --fine-tune-mode full-model --use_gpu --train_sst --train_sts --epochs 3 --batch_size 256
python3 multitask_classifier.py --amp --benchmark --lr 1e-4 --lora_dict '{"mode":"all-lin-only", "r":10, "dora":1}' --fine-tune-mode full-model --use_gpu --train_sst --train_sts --epochs 3 --batch_size 2564
--benchmark --amp 
python3 multitask_classifier.py --amp --benchmark --lr 1e-4 --lora_dict '{"mode":"all-lin", "r":1, "dora":1}' --fine-tune-mode full-model --use_gpu --train_sst --train_sts --epochs 3 --batch_size 256
python3 multitask_classifier.py --amp --benchmark --lr 1e-4 --lora_dict '{"mode":"all-lin", "r":5, "dora":1}' --fine-tune-mode full-model --use_gpu --train_sst --train_sts --epochs 3 --batch_size 256
python3 multitask_classifier.py --amp --benchmark --lr 1e-4 --lora_dict '{"mode":"all-lin", "r":10, "dora":1}' --fine-tune-mode full-model --use_gpu --train_sst --train_sts --epochs 3 --batch_size 2564
# lr 1e-4 bs 124--benchmark --amp 
python3 multitask_classifier.py --amp --benchmark --lr 1e-4 --lora_dict '{"mode":"attn-only", "r":1, "dora":1}' --fine-tune-mode full-model --use_gpu --train_sst --train_sts --epochs 3 --batch_size 128
python3 multitask_classifier.py --amp --benchmark --lr 1e-4 --lora_dict '{"mode":"attn-only", "r":5, "dora":1}' --fine-tune-mode full-model --use_gpu --train_sst --train_sts --epochs 3 --batch_size 128
python3 multitask_classifier.py --amp --benchmark --lr 1e-4 --lora_dict '{"mode":"attn-only", "r":10, "dora":1}' --fine-tune-mode full-model --use_gpu --train_sst --train_sts --epochs 3 --batch_size 1284
--benchmark --amp 
python3 multitask_classifier.py --amp --benchmark --lr 1e-4 --lora_dict '{"mode":"attn", "r":1, "dora":1}' --fine-tune-mode full-model --use_gpu --train_sst --train_sts --epochs 3 --batch_size 128
python3 multitask_classifier.py --amp --benchmark --lr 1e-4 --lora_dict '{"mode":"attn", "r":5, "dora":1}' --fine-tune-mode full-model --use_gpu --train_sst --train_sts --epochs 3 --batch_size 128
python3 multitask_classifier.py --amp --benchmark --lr 1e-4 --lora_dict '{"mode":"attn", "r":10, "dora":1}' --fine-tune-mode full-model --use_gpu --train_sst --train_sts --epochs 3 --batch_size 1284
--benchmark --amp 
python3 multitask_classifier.py --amp --benchmark --lr 1e-4 --lora_dict '{"mode":"all-lin-only", "r":1, "dora":1}' --fine-tune-mode full-model --use_gpu --train_sst --train_sts --epochs 3 --batch_size 128
python3 multitask_classifier.py --amp --benchmark --lr 1e-4 --lora_dict '{"mode":"all-lin-only", "r":5, "dora":1}' --fine-tune-mode full-model --use_gpu --train_sst --train_sts --epochs 3 --batch_size 128
python3 multitask_classifier.py --amp --benchmark --lr 1e-4 --lora_dict '{"mode":"all-lin-only", "r":10, "dora":1}' --fine-tune-mode full-model --use_gpu --train_sst --train_sts --epochs 3 --batch_size 1284
--benchmark --amp 
python3 multitask_classifier.py --amp --benchmark --lr 1e-4 --lora_dict '{"mode":"all-lin", "r":1, "dora":1}' --fine-tune-mode full-model --use_gpu --train_sst --train_sts --epochs 3 --batch_size 128
python3 multitask_classifier.py --amp --benchmark --lr 1e-4 --lora_dict '{"mode":"all-lin", "r":5, "dora":1}' --fine-tune-mode full-model --use_gpu --train_sst --train_sts --epochs 3 --batch_size 128
python3 multitask_classifier.py --amp --benchmark --lr 1e-4 --lora_dict '{"mode":"all-lin", "r":10, "dora":1}' --fine-tune-mode full-model --use_gpu --train_sst --train_sts --epochs 3 --batch_size 1284
# lr 1e-4 bs 64--benchmark --amp 
python3 multitask_classifier.py --amp --benchmark --lr 1e-4 --lora_dict '{"mode":"attn-only", "r":1, "dora":1}' --fine-tune-mode full-model --use_gpu --train_sst --train_sts --epochs 3 --batch_size 64
python3 multitask_classifier.py --amp --benchmark --lr 1e-4 --lora_dict '{"mode":"attn-only", "r":5, "dora":1}' --fine-tune-mode full-model --use_gpu --train_sst --train_sts --epochs 3 --batch_size 64
python3 multitask_classifier.py --amp --benchmark --lr 1e-4 --lora_dict '{"mode":"attn-only", "r":10, "dora":1}' --fine-tune-mode full-model --use_gpu --train_sst --train_sts --epochs 3 --batch_size 644
--benchmark --amp 
python3 multitask_classifier.py --amp --benchmark --lr 1e-4 --lora_dict '{"mode":"attn", "r":1, "dora":1}' --fine-tune-mode full-model --use_gpu --train_sst --train_sts --epochs 3 --batch_size 64
python3 multitask_classifier.py --amp --benchmark --lr 1e-4 --lora_dict '{"mode":"attn", "r":5, "dora":1}' --fine-tune-mode full-model --use_gpu --train_sst --train_sts --epochs 3 --batch_size 64
python3 multitask_classifier.py --amp --benchmark --lr 1e-4 --lora_dict '{"mode":"attn", "r":10, "dora":1}' --fine-tune-mode full-model --use_gpu --train_sst --train_sts --epochs 3 --batch_size 644
--benchmark --amp 
python3 multitask_classifier.py --amp --benchmark --lr 1e-4 --lora_dict '{"mode":"all-lin-only", "r":1, "dora":1}' --fine-tune-mode full-model --use_gpu --train_sst --train_sts --epochs 3 --batch_size 64
python3 multitask_classifier.py --amp --benchmark --lr 1e-4 --lora_dict '{"mode":"all-lin-only", "r":5, "dora":1}' --fine-tune-mode full-model --use_gpu --train_sst --train_sts --epochs 3 --batch_size 64
python3 multitask_classifier.py --amp --benchmark --lr 1e-4 --lora_dict '{"mode":"all-lin-only", "r":10, "dora":1}' --fine-tune-mode full-model --use_gpu --train_sst --train_sts --epochs 3 --batch_size 644
--benchmark --amp 
python3 multitask_classifier.py --amp --benchmark --lr 1e-4 --lora_dict '{"mode":"all-lin", "r":1, "dora":1}' --fine-tune-mode full-model --use_gpu --train_sst --train_sts --epochs 3 --batch_size 64
python3 multitask_classifier.py --amp --benchmark --lr 1e-4 --lora_dict '{"mode":"all-lin", "r":5, "dora":1}' --fine-tune-mode full-model --use_gpu --train_sst --train_sts --epochs 3 --batch_size 64
python3 multitask_classifier.py --amp --benchmark --lr 1e-4 --lora_dict '{"mode":"all-lin", "r":10, "dora":1}' --fine-tune-mode full-model --use_gpu --train_sst --train_sts --epochs 3 --batch_size 64
--benchmark --amp 
# lr 5e-5 bs 386--benchmark --amp 
--benchmark --amp 
python3 multitask_classifier.py --amp --benchmark --lr 5e-5 --lora_dict '{"mode":"attn-only", "r":1, "dora":1}' --fine-tune-mode full-model --use_gpu --train_sst --train_sts --epochs 3 --batch_size 384
python3 multitask_classifier.py --amp --benchmark --lr 5e-5 --lora_dict '{"mode":"attn-only", "r":5, "dora":1}' --fine-tune-mode full-model --use_gpu --train_sst --train_sts --epochs 3 --batch_size 384
python3 multitask_classifier.py --amp --benchmark --lr 5e-5 --lora_dict '{"mode":"attn-only", "r":10, "dora":1}' --fine-tune-mode full-model --use_gpu --train_sst --train_sts --epochs 3 --batch_size 3844
--benchmark --amp 
python3 multitask_classifier.py --amp --benchmark --lr 5e-5 --lora_dict '{"mode":"attn", "r":1, "dora":1}' --fine-tune-mode full-model --use_gpu --train_sst --train_sts --epochs 3 --batch_size 384
python3 multitask_classifier.py --amp --benchmark --lr 5e-5 --lora_dict '{"mode":"attn", "r":5, "dora":1}' --fine-tune-mode full-model --use_gpu --train_sst --train_sts --epochs 3 --batch_size 384
python3 multitask_classifier.py --amp --benchmark --lr 5e-5 --lora_dict '{"mode":"attn", "r":10, "dora":1}' --fine-tune-mode full-model --use_gpu --train_sst --train_sts --epochs 3 --batch_size 3844
--benchmark --amp 
python3 multitask_classifier.py --amp --benchmark --lr 5e-5 --lora_dict '{"mode":"all-lin-only", "r":1, "dora":1}' --fine-tune-mode full-model --use_gpu --train_sst --train_sts --epochs 3 --batch_size 384
python3 multitask_classifier.py --amp --benchmark --lr 5e-5 --lora_dict '{"mode":"all-lin-only", "r":5, "dora":1}' --fine-tune-mode full-model --use_gpu --train_sst --train_sts --epochs 3 --batch_size 384
python3 multitask_classifier.py --amp --benchmark --lr 5e-5 --lora_dict '{"mode":"all-lin-only", "r":10, "dora":1}' --fine-tune-mode full-model --use_gpu --train_sst --train_sts --epochs 3 --batch_size 3844
--benchmark --amp 
python3 multitask_classifier.py --amp --benchmark --lr 5e-5 --lora_dict '{"mode":"all-lin", "r":1, "dora":1}' --fine-tune-mode full-model --use_gpu --train_sst --train_sts --epochs 3 --batch_size 384
python3 multitask_classifier.py --amp --benchmark --lr 5e-5 --lora_dict '{"mode":"all-lin", "r":5, "dora":1}' --fine-tune-mode full-model --use_gpu --train_sst --train_sts --epochs 3 --batch_size 384
python3 multitask_classifier.py --amp --benchmark --lr 5e-5 --lora_dict '{"mode":"all-lin", "r":10, "dora":1}' --fine-tune-mode full-model --use_gpu --train_sst --train_sts --epochs 3 --batch_size 3844
# lr 1e-4 bs5e-5--benchmark --amp 
python3 multitask_classifier.py --amp --benchmark --lr 5e-5 --lora_dict '{"mode":"attn-only", "r":1, "dora":1}' --fine-tune-mode full-model --use_gpu --train_sst --train_sts --epochs 3 --batch_size 256
python3 multitask_classifier.py --amp --benchmark --lr 5e-5 --lora_dict '{"mode":"attn-only", "r":5, "dora":1}' --fine-tune-mode full-model --use_gpu --train_sst --train_sts --epochs 3 --batch_size 256
python3 multitask_classifier.py --amp --benchmark --lr 5e-5 --lora_dict '{"mode":"attn-only", "r":10, "dora":1}' --fine-tune-mode full-model --use_gpu --train_sst --train_sts --epochs 3 --batch_size 2564
--benchmark --amp 
python3 multitask_classifier.py --amp --benchmark --lr 5e-5 --lora_dict '{"mode":"attn", "r":1, "dora":1}' --fine-tune-mode full-model --use_gpu --train_sst --train_sts --epochs 3 --batch_size 256
python3 multitask_classifier.py --amp --benchmark --lr 5e-5 --lora_dict '{"mode":"attn", "r":5, "dora":1}' --fine-tune-mode full-model --use_gpu --train_sst --train_sts --epochs 3 --batch_size 256
python3 multitask_classifier.py --amp --benchmark --lr 5e-5 --lora_dict '{"mode":"attn", "r":10, "dora":1}' --fine-tune-mode full-model --use_gpu --train_sst --train_sts --epochs 3 --batch_size 2564
--benchmark --amp 
python3 multitask_classifier.py --amp --benchmark --lr 5e-5 --lora_dict '{"mode":"all-lin-only", "r":1, "dora":1}' --fine-tune-mode full-model --use_gpu --train_sst --train_sts --epochs 3 --batch_size 256
python3 multitask_classifier.py --amp --benchmark --lr 5e-5 --lora_dict '{"mode":"all-lin-only", "r":5, "dora":1}' --fine-tune-mode full-model --use_gpu --train_sst --train_sts --epochs 3 --batch_size 256
python3 multitask_classifier.py --amp --benchmark --lr 5e-5 --lora_dict '{"mode":"all-lin-only", "r":10, "dora":1}' --fine-tune-mode full-model --use_gpu --train_sst --train_sts --epochs 3 --batch_size 2564
--benchmark --amp 
python3 multitask_classifier.py --amp --benchmark --lr 5e-5 --lora_dict '{"mode":"all-lin", "r":1, "dora":1}' --fine-tune-mode full-model --use_gpu --train_sst --train_sts --epochs 3 --batch_size 256
python3 multitask_classifier.py --amp --benchmark --lr 5e-5 --lora_dict '{"mode":"all-lin", "r":5, "dora":1}' --fine-tune-mode full-model --use_gpu --train_sst --train_sts --epochs 3 --batch_size 256
python3 multitask_classifier.py --amp --benchmark --lr 5e-5 --lora_dict '{"mode":"all-lin", "r":10, "dora":1}' --fine-tune-mode full-model --use_gpu --train_sst --train_sts --epochs 3 --batch_size 2564
# lr 1e-4 bs5e-5--benchmark --amp 
python3 multitask_classifier.py --amp --benchmark --lr 5e-5 --lora_dict '{"mode":"attn-only", "r":1, "dora":1}' --fine-tune-mode full-model --use_gpu --train_sst --train_sts --epochs 3 --batch_size 128
python3 multitask_classifier.py --amp --benchmark --lr 5e-5 --lora_dict '{"mode":"attn-only", "r":5, "dora":1}' --fine-tune-mode full-model --use_gpu --train_sst --train_sts --epochs 3 --batch_size 128
python3 multitask_classifier.py --amp --benchmark --lr 5e-5 --lora_dict '{"mode":"attn-only", "r":10, "dora":1}' --fine-tune-mode full-model --use_gpu --train_sst --train_sts --epochs 3 --batch_size 1284
--benchmark --amp 
python3 multitask_classifier.py --amp --benchmark --lr 5e-5 --lora_dict '{"mode":"attn", "r":1, "dora":1}' --fine-tune-mode full-model --use_gpu --train_sst --train_sts --epochs 3 --batch_size 128
python3 multitask_classifier.py --amp --benchmark --lr 5e-5 --lora_dict '{"mode":"attn", "r":5, "dora":1}' --fine-tune-mode full-model --use_gpu --train_sst --train_sts --epochs 3 --batch_size 128
python3 multitask_classifier.py --amp --benchmark --lr 5e-5 --lora_dict '{"mode":"attn", "r":10, "dora":1}' --fine-tune-mode full-model --use_gpu --train_sst --train_sts --epochs 3 --batch_size 1284
--benchmark --amp 
python3 multitask_classifier.py --amp --benchmark --lr 5e-5 --lora_dict '{"mode":"all-lin-only", "r":1, "dora":1}' --fine-tune-mode full-model --use_gpu --train_sst --train_sts --epochs 3 --batch_size 128
python3 multitask_classifier.py --amp --benchmark --lr 5e-5 --lora_dict '{"mode":"all-lin-only", "r":5, "dora":1}' --fine-tune-mode full-model --use_gpu --train_sst --train_sts --epochs 3 --batch_size 128
python3 multitask_classifier.py --amp --benchmark --lr 5e-5 --lora_dict '{"mode":"all-lin-only", "r":10, "dora":1}' --fine-tune-mode full-model --use_gpu --train_sst --train_sts --epochs 3 --batch_size 1284
--benchmark --amp 
python3 multitask_classifier.py --amp --benchmark --lr 5e-5 --lora_dict '{"mode":"all-lin", "r":1, "dora":1}' --fine-tune-mode full-model --use_gpu --train_sst --train_sts --epochs 3 --batch_size 128
python3 multitask_classifier.py --amp --benchmark --lr 5e-5 --lora_dict '{"mode":"all-lin", "r":5, "dora":1}' --fine-tune-mode full-model --use_gpu --train_sst --train_sts --epochs 3 --batch_size 128
python3 multitask_classifier.py --amp --benchmark --lr 5e-5 --lora_dict '{"mode":"all-lin", "r":10, "dora":1}' --fine-tune-mode full-model --use_gpu --train_sst --train_sts --epochs 3 --batch_size 1284
# lr 1e-4 b5e-5--benchmark --amp 
python3 multitask_classifier.py --amp --benchmark --lr 5e-5 --lora_dict '{"mode":"attn-only", "r":1, "dora":1}' --fine-tune-mode full-model --use_gpu --train_sst --train_sts --epochs 3 --batch_size 64
python3 multitask_classifier.py --amp --benchmark --lr 5e-5 --lora_dict '{"mode":"attn-only", "r":5, "dora":1}' --fine-tune-mode full-model --use_gpu --train_sst --train_sts --epochs 3 --batch_size 64
python3 multitask_classifier.py --amp --benchmark --lr 5e-5 --lora_dict '{"mode":"attn-only", "r":10, "dora":1}' --fine-tune-mode full-model --use_gpu --train_sst --train_sts --epochs 3 --batch_size 644
--benchmark --amp 
python3 multitask_classifier.py --amp --benchmark --lr 5e-5 --lora_dict '{"mode":"attn", "r":1, "dora":1}' --fine-tune-mode full-model --use_gpu --train_sst --train_sts --epochs 3 --batch_size 64
python3 multitask_classifier.py --amp --benchmark --lr 5e-5 --lora_dict '{"mode":"attn", "r":5, "dora":1}' --fine-tune-mode full-model --use_gpu --train_sst --train_sts --epochs 3 --batch_size 64
python3 multitask_classifier.py --amp --benchmark --lr 5e-5 --lora_dict '{"mode":"attn", "r":10, "dora":1}' --fine-tune-mode full-model --use_gpu --train_sst --train_sts --epochs 3 --batch_size 644
--benchmark --amp 
python3 multitask_classifier.py --amp --benchmark --lr 5e-5 --lora_dict '{"mode":"all-lin-only", "r":1, "dora":1}' --fine-tune-mode full-model --use_gpu --train_sst --train_sts --epochs 3 --batch_size 64
python3 multitask_classifier.py --amp --benchmark --lr 5e-5 --lora_dict '{"mode":"all-lin-only", "r":5, "dora":1}' --fine-tune-mode full-model --use_gpu --train_sst --train_sts --epochs 3 --batch_size 64
python3 multitask_classifier.py --amp --benchmark --lr 5e-5 --lora_dict '{"mode":"all-lin-only", "r":10, "dora":1}' --fine-tune-mode full-model --use_gpu --train_sst --train_sts --epochs 3 --batch_size 644
--benchmark --amp 
python3 multitask_classifier.py --amp --benchmark --lr 5e-5 --lora_dict '{"mode":"all-lin", "r":1, "dora":1}' --fine-tune-mode full-model --use_gpu --train_sst --train_sts --epochs 3 --batch_size 64
python3 multitask_classifier.py --amp --benchmark --lr 5e-5 --lora_dict '{"mode":"all-lin", "r":5, "dora":1}' --fine-tune-mode full-model --use_gpu --train_sst --train_sts --epochs 3 --batch_size 64
python3 multitask_classifier.py --amp --benchmark --lr 5e-5 --lora_dict '{"mode":"all-lin", "r":10, "dora":1}' --fine-tune-mode full-model --use_gpu --train_sst --train_sts --epochs 3 --batch_size 64
--benchmark --amp 
# lr 5e-6 bs 386--benchmark --amp 
python3 multitask_classifier.py --amp --benchmark --lr 5e-6 --lora_dict '{"mode":"attn-only", "r":1, "dora":1}' --fine-tune-mode full-model --use_gpu --train_sst --train_sts --epochs 3 --batch_size 384
python3 multitask_classifier.py --amp --benchmark --lr 5e-6 --lora_dict '{"mode":"attn-only", "r":5, "dora":1}' --fine-tune-mode full-model --use_gpu --train_sst --train_sts --epochs 3 --batch_size 384
python3 multitask_classifier.py --amp --benchmark --lr 5e-6 --lora_dict '{"mode":"attn-only", "r":10, "dora":1}' --fine-tune-mode full-model --use_gpu --train_sst --train_sts --epochs 3 --batch_size 3844
--benchmark --amp 
python3 multitask_classifier.py --amp --benchmark --lr 5e-6 --lora_dict '{"mode":"attn", "r":1, "dora":1}' --fine-tune-mode full-model --use_gpu --train_sst --train_sts --epochs 3 --batch_size 384
python3 multitask_classifier.py --amp --benchmark --lr 5e-6 --lora_dict '{"mode":"attn", "r":5, "dora":1}' --fine-tune-mode full-model --use_gpu --train_sst --train_sts --epochs 3 --batch_size 384
python3 multitask_classifier.py --amp --benchmark --lr 5e-6 --lora_dict '{"mode":"attn", "r":10, "dora":1}' --fine-tune-mode full-model --use_gpu --train_sst --train_sts --epochs 3 --batch_size 3844
--benchmark --amp 
python3 multitask_classifier.py --amp --benchmark --lr 5e-6 --lora_dict '{"mode":"all-lin-only", "r":1, "dora":1}' --fine-tune-mode full-model --use_gpu --train_sst --train_sts --epochs 3 --batch_size 384
python3 multitask_classifier.py --amp --benchmark --lr 5e-6 --lora_dict '{"mode":"all-lin-only", "r":5, "dora":1}' --fine-tune-mode full-model --use_gpu --train_sst --train_sts --epochs 3 --batch_size 384
python3 multitask_classifier.py --amp --benchmark --lr 5e-6 --lora_dict '{"mode":"all-lin-only", "r":10, "dora":1}' --fine-tune-mode full-model --use_gpu --train_sst --train_sts --epochs 3 --batch_size 3844
--benchmark --amp 
python3 multitask_classifier.py --amp --benchmark --lr 5e-6 --lora_dict '{"mode":"all-lin", "r":1, "dora":1}' --fine-tune-mode full-model --use_gpu --train_sst --train_sts --epochs 3 --batch_size 384
python3 multitask_classifier.py --amp --benchmark --lr 5e-6 --lora_dict '{"mode":"all-lin", "r":5, "dora":1}' --fine-tune-mode full-model --use_gpu --train_sst --train_sts --epochs 3 --batch_size 384
python3 multitask_classifier.py --amp --benchmark --lr 5e-6 --lora_dict '{"mode":"all-lin", "r":10, "dora":1}' --fine-tune-mode full-model --use_gpu --train_sst --train_sts --epochs 3 --batch_size 3844
# lr 1e-4 bs5e-6--benchmark --amp 
python3 multitask_classifier.py --amp --benchmark --lr 5e-6 --lora_dict '{"mode":"attn-only", "r":1, "dora":1}' --fine-tune-mode full-model --use_gpu --train_sst --train_sts --epochs 3 --batch_size 256
python3 multitask_classifier.py --amp --benchmark --lr 5e-6 --lora_dict '{"mode":"attn-only", "r":5, "dora":1}' --fine-tune-mode full-model --use_gpu --train_sst --train_sts --epochs 3 --batch_size 256
python3 multitask_classifier.py --amp --benchmark --lr 5e-6 --lora_dict '{"mode":"attn-only", "r":10, "dora":1}' --fine-tune-mode full-model --use_gpu --train_sst --train_sts --epochs 3 --batch_size 2564
--benchmark --amp 
python3 multitask_classifier.py --amp --benchmark --lr 5e-6 --lora_dict '{"mode":"attn", "r":1, "dora":1}' --fine-tune-mode full-model --use_gpu --train_sst --train_sts --epochs 3 --batch_size 256
python3 multitask_classifier.py --amp --benchmark --lr 5e-6 --lora_dict '{"mode":"attn", "r":5, "dora":1}' --fine-tune-mode full-model --use_gpu --train_sst --train_sts --epochs 3 --batch_size 256
python3 multitask_classifier.py --amp --benchmark --lr 5e-6 --lora_dict '{"mode":"attn", "r":10, "dora":1}' --fine-tune-mode full-model --use_gpu --train_sst --train_sts --epochs 3 --batch_size 2564
--benchmark --amp 
python3 multitask_classifier.py --amp --benchmark --lr 5e-6 --lora_dict '{"mode":"all-lin-only", "r":1, "dora":1}' --fine-tune-mode full-model --use_gpu --train_sst --train_sts --epochs 3 --batch_size 256
python3 multitask_classifier.py --amp --benchmark --lr 5e-6 --lora_dict '{"mode":"all-lin-only", "r":5, "dora":1}' --fine-tune-mode full-model --use_gpu --train_sst --train_sts --epochs 3 --batch_size 256
python3 multitask_classifier.py --amp --benchmark --lr 5e-6 --lora_dict '{"mode":"all-lin-only", "r":10, "dora":1}' --fine-tune-mode full-model --use_gpu --train_sst --train_sts --epochs 3 --batch_size 2564
--benchmark --amp 
python3 multitask_classifier.py --amp --benchmark --lr 5e-6 --lora_dict '{"mode":"all-lin", "r":1, "dora":1}' --fine-tune-mode full-model --use_gpu --train_sst --train_sts --epochs 3 --batch_size 256
python3 multitask_classifier.py --amp --benchmark --lr 5e-6 --lora_dict '{"mode":"all-lin", "r":5, "dora":1}' --fine-tune-mode full-model --use_gpu --train_sst --train_sts --epochs 3 --batch_size 256
python3 multitask_classifier.py --amp --benchmark --lr 5e-6 --lora_dict '{"mode":"all-lin", "r":10, "dora":1}' --fine-tune-mode full-model --use_gpu --train_sst --train_sts --epochs 3 --batch_size 2564
# lr 1e-4 bs5e-6--benchmark --amp 
python3 multitask_classifier.py --amp --benchmark --lr 5e-6 --lora_dict '{"mode":"attn-only", "r":1, "dora":1}' --fine-tune-mode full-model --use_gpu --train_sst --train_sts --epochs 3 --batch_size 128
python3 multitask_classifier.py --amp --benchmark --lr 5e-6 --lora_dict '{"mode":"attn-only", "r":5, "dora":1}' --fine-tune-mode full-model --use_gpu --train_sst --train_sts --epochs 3 --batch_size 128
python3 multitask_classifier.py --amp --benchmark --lr 5e-6 --lora_dict '{"mode":"attn-only", "r":10, "dora":1}' --fine-tune-mode full-model --use_gpu --train_sst --train_sts --epochs 3 --batch_size 1284
--benchmark --amp 
python3 multitask_classifier.py --amp --benchmark --lr 5e-6 --lora_dict '{"mode":"attn", "r":1, "dora":1}' --fine-tune-mode full-model --use_gpu --train_sst --train_sts --epochs 3 --batch_size 128
python3 multitask_classifier.py --amp --benchmark --lr 5e-6 --lora_dict '{"mode":"attn", "r":5, "dora":1}' --fine-tune-mode full-model --use_gpu --train_sst --train_sts --epochs 3 --batch_size 128
python3 multitask_classifier.py --amp --benchmark --lr 5e-6 --lora_dict '{"mode":"attn", "r":10, "dora":1}' --fine-tune-mode full-model --use_gpu --train_sst --train_sts --epochs 3 --batch_size 1284
--benchmark --amp 
python3 multitask_classifier.py --amp --benchmark --lr 5e-6 --lora_dict '{"mode":"all-lin-only", "r":1, "dora":1}' --fine-tune-mode full-model --use_gpu --train_sst --train_sts --epochs 3 --batch_size 128
python3 multitask_classifier.py --amp --benchmark --lr 5e-6 --lora_dict '{"mode":"all-lin-only", "r":5, "dora":1}' --fine-tune-mode full-model --use_gpu --train_sst --train_sts --epochs 3 --batch_size 128
python3 multitask_classifier.py --amp --benchmark --lr 5e-6 --lora_dict '{"mode":"all-lin-only", "r":10, "dora":1}' --fine-tune-mode full-model --use_gpu --train_sst --train_sts --epochs 3 --batch_size 1284
--benchmark --amp 
python3 multitask_classifier.py --amp --benchmark --lr 5e-6 --lora_dict '{"mode":"all-lin", "r":1, "dora":1}' --fine-tune-mode full-model --use_gpu --train_sst --train_sts --epochs 3 --batch_size 128
python3 multitask_classifier.py --amp --benchmark --lr 5e-6 --lora_dict '{"mode":"all-lin", "r":5, "dora":1}' --fine-tune-mode full-model --use_gpu --train_sst --train_sts --epochs 3 --batch_size 128
python3 multitask_classifier.py --amp --benchmark --lr 5e-6 --lora_dict '{"mode":"all-lin", "r":10, "dora":1}' --fine-tune-mode full-model --use_gpu --train_sst --train_sts --epochs 3 --batch_size 1284
# lr 1e-4 b5e-6--benchmark --amp 
python3 multitask_classifier.py --amp --benchmark --lr 5e-6 --lora_dict '{"mode":"attn-only", "r":1, "dora":1}' --fine-tune-mode full-model --use_gpu --train_sst --train_sts --epochs 3 --batch_size 64
python3 multitask_classifier.py --amp --benchmark --lr 5e-6 --lora_dict '{"mode":"attn-only", "r":5, "dora":1}' --fine-tune-mode full-model --use_gpu --train_sst --train_sts --epochs 3 --batch_size 64
python3 multitask_classifier.py --amp --benchmark --lr 5e-6 --lora_dict '{"mode":"attn-only", "r":10, "dora":1}' --fine-tune-mode full-model --use_gpu --train_sst --train_sts --epochs 3 --batch_size 644
--benchmark --amp 
python3 multitask_classifier.py --amp --benchmark --lr 5e-6 --lora_dict '{"mode":"attn", "r":1, "dora":1}' --fine-tune-mode full-model --use_gpu --train_sst --train_sts --epochs 3 --batch_size 64
python3 multitask_classifier.py --amp --benchmark --lr 5e-6 --lora_dict '{"mode":"attn", "r":5, "dora":1}' --fine-tune-mode full-model --use_gpu --train_sst --train_sts --epochs 3 --batch_size 64
python3 multitask_classifier.py --amp --benchmark --lr 5e-6 --lora_dict '{"mode":"attn", "r":10, "dora":1}' --fine-tune-mode full-model --use_gpu --train_sst --train_sts --epochs 3 --batch_size 644
--benchmark --amp 
python3 multitask_classifier.py --amp --benchmark --lr 5e-6 --lora_dict '{"mode":"all-lin-only", "r":1, "dora":1}' --fine-tune-mode full-model --use_gpu --train_sst --train_sts --epochs 3 --batch_size 64
python3 multitask_classifier.py --amp --benchmark --lr 5e-6 --lora_dict '{"mode":"all-lin-only", "r":5, "dora":1}' --fine-tune-mode full-model --use_gpu --train_sst --train_sts --epochs 3 --batch_size 64
python3 multitask_classifier.py --amp --benchmark --lr 5e-6 --lora_dict '{"mode":"all-lin-only", "r":10, "dora":1}' --fine-tune-mode full-model --use_gpu --train_sst --train_sts --epochs 3 --batch_size 644
--benchmark --amp 
python3 multitask_classifier.py --amp --benchmark --lr 5e-6 --lora_dict '{"mode":"all-lin", "r":1, "dora":1}' --fine-tune-mode full-model --use_gpu --train_sst --train_sts --epochs 3 --batch_size 64
python3 multitask_classifier.py --amp --benchmark --lr 5e-6 --lora_dict '{"mode":"all-lin", "r":5, "dora":1}' --fine-tune-mode full-model --use_gpu --train_sst --train_sts --epochs 3 --batch_size 64
python3 multitask_classifier.py --amp --benchmark --lr 5e-6 --lora_dict '{"mode":"all-lin", "r":10, "dora":1}' --fine-tune-mode full-model --use_gpu --train_sst --train_sts --epochs 3 --batch_size 64
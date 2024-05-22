#Write a shell script that calls python3 classifier.py with --lr 1e-5 --use_gpu and python3 lora.py with --lr 1e-5 --use_gpu --lora_r {1,5,10, 25, 50, 100}

python3 lora_all_layers_attn_LoRA.py --lr 1e-5 --use_gpu --lora_r 1 --fine-tune-mode full-model
python3 lora_all_layers_attn_LoRA.py --lr 1e-5 --use_gpu --lora_r 5 --fine-tune-mode full-model
python3 lora_all_layers_attn_LoRA.py --lr 1e-5 --use_gpu --lora_r 10 --fine-tune-mode full-model
python3 lora_all_layers_attn_LoRA.py --lr 1e-5 --use_gpu --lora_r 25 --fine-tune-mode full-model
python3 lora_all_layers_attn_LoRA.py --lr 1e-5 --use_gpu --lora_r 50 --fine-tune-mode full-model
python3 lora_all_layers_attn_LoRA.py --lr 1e-5 --use_gpu --lora_r 100 --fine-tune-mode full-model

python3 lora_all_layers_only_attn_LoRA.py --lr 1e-5 --use_gpu --lora_r 1 --fine-tune-mode full-model
python3 lora_all_layers_only_attn_LoRA.py --lr 1e-5 --use_gpu --lora_r 5 --fine-tune-mode full-model
python3 lora_all_layers_only_attn_LoRA.py --lr 1e-5 --use_gpu --lora_r 10 --fine-tune-mode full-model
python3 lora_all_layers_only_attn_LoRA.py --lr 1e-5 --use_gpu --lora_r 25 --fine-tune-mode full-model
python3 lora_all_layers_only_attn_LoRA.py --lr 1e-5 --use_gpu --lora_r 50 --fine-tune-mode full-model
python3 lora_all_layers_only_attn_LoRA.py --lr 1e-5 --use_gpu --lora_r 100 --fine-tune-mode full-model

python3 lora_cdimdb_only.py --lr 1e-5 --use_gpu --lora_r 1 --fine-tune-mode full-model
python3 lora_cdimdb_only.py --lr 1e-5 --use_gpu --lora_r 5 --fine-tune-mode full-model
python3 lora_cdimdb_only.py --lr 1e-5 --use_gpu --lora_r 10 --fine-tune-mode full-model
python3 lora_cdimdb_only.py --lr 1e-5 --use_gpu --lora_r 25 --fine-tune-mode full-model
python3 lora_cdimdb_only.py --lr 1e-5 --use_gpu --lora_r 50 --fine-tune-mode full-model
python3 lora_cdimdb_only.py --lr 1e-5 --use_gpu --lora_r 100 --fine-tune-mode full-model

python3 classifier.py --lr 1e-5 --use_gpu --fine-tune-mode full-model
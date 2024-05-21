#Write a shell script that calls python3 classifier.py with --lr 1e-5 --use_gpu and python3 lora.py with --lr 1e-5 --use_gpu --lora_r {1,5,10, 25, 50, 100}

python3 classifier.py --lr 1e-5 --use_gpu
python3 lora.py --lr 1e-5 --use_gpu --lora_r 1
python3 lora.py --lr 1e-5 --use_gpu --lora_r 5
python3 lora.py --lr 1e-5 --use_gpu --lora_r 10
python3 lora.py --lr 1e-5 --use_gpu --lora_r 25
python3 lora.py --lr 1e-5 --use_gpu --lora_r 50
python3 lora.py --lr 1e-5 --use_gpu --lora_r 100


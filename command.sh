# Pretraining
python3 -u main.py --dataset cifar10 --model Conv4 --pretrain true --epoch 100 --lr 0.001

# Custom Fine-tuning
python3 -u main.py --dataset cifar10 --model alexnet --freeze 00000111 --mode cus --epoch 100 --lr 0.001
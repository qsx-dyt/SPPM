# Indian
python main.py --dataset=IndianPines --patch_size=9 --epochs=100 --SpaMask_ratio=0.6 --mask_style=spa --flag=pretrain
python main.py --dataset=IndianPines --patch_size=9 --epochs=100 --pt_model=IndianPines_9_0.6_pt.pth --flag=finetune
python main.py --dataset=IndianPines --patch_size=9 --model_path=output/IndianPines.pth --flag=test

# PaviaU
python main.py --dataset=PaviaU --patch_size=9 --epochs=100 --SpaMask_ratio=0.6 --mask_style=spa --flag=pretrain
python main.py --dataset=PaviaU --patch_size=9 --epochs=100 --pt_model=PaviaU_9_0.6_pt.pth --flag=finetune
python main.py --dataset=PaviaU --patch_size=9 --model_path=output/PaviaU.pth --flag=test

# Houston2013
python main.py --dataset=Houston13 --patch_size=11 --epochs=100 --SpaMask_ratio=0.6 --mask_style=spa --flag=pretrain
python main.py --dataset=Houston13 --patch_size=11 --epochs=100 --pt_model=Houston13_9_0.6_pt.pth --flag=finetune
python main.py --dataset=Houston13 --patch_size=11 --model_path=output/Houston13.pth --flag=test
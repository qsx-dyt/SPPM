import argparse
import os
import time

import numpy as np
import torch
from scipy.io import loadmat
from torch.backends import cudnn
from torch.utils import data as Data

from finetune import finetune
from models import build_model
from pretrain import pretrain
from utils.build_data import HSIDatasetBuilder
from utils.util import save_results_to_log, validate, calculate_metrics
from utils.visualise import plot_classification

parser = argparse.ArgumentParser("HSI_SSl")
parser.add_argument('--dataset', type=str, choices=['IndianPines', 'PaviaU', 'Houston13'], default='IndianPines',
                    help='dataset name')
parser.add_argument('--patch_size', type=int, default=9, help='size of patch')
parser.add_argument('--batch_size', type=int, default=96, help='batch size')
parser.add_argument('--epochs', type=int, default=300, help='number of epochs')
parser.add_argument('--SpeMask_ratio', type=float, default=0.0, help='spectral mask ratio')
parser.add_argument('--SpaMask_ratio', type=float, default=0.6, help='spatial mask ratio')
parser.add_argument('--mask_style', type=str, choices=['spe', 'spa', 'both'], default='both', help='mask style')
parser.add_argument('--samples', type=int, default=50, help='number of per sample')
parser.add_argument('--sample_step', type=int, default=1, help='step of per sample')
parser.add_argument('--load_pt', action='store_true', default=False, help='load pretrained model')
parser.add_argument('--output_dir', type=str, default='./output', help='path where to save')
parser.add_argument('--model_path', type=str, default='./output/IndianPines_best.pth')
parser.add_argument('--pt_model', type=str, default=None, help='pretrained model name')
parser.add_argument('--gpu_id', type=int, default=0, help='gpu id')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--flag', type=str, default='test', choices=['pretrain', 'finetune', 'test'], help='types of training')

args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)

np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
# cudnn.benchmark = True
cudnn.deterministic = True


def generate_spectral_mask(x, mask_ratio=0.6):
    # x shape: (B, C, N)
    B, C, _ = x.shape
    # 计算实际要遮蔽的通道数（至少保留1个通道）
    k = max(1, int(C * (1 - mask_ratio)))  # 保留的通道数

    # 生成随机排序索引
    rand_indices = torch.rand(B, C, device=x.device).argsort(dim=1)
    # 创建保留通道掩码（选择前k个保留）
    keep_mask = torch.zeros((B, C), dtype=torch.bool, device=x.device)
    keep_mask.scatter_(1, rand_indices[:, :k], True)

    # 反转得到遮蔽掩码
    mask = ~keep_mask
    return mask


def generate_spatial_mask(x, mask_ratio=0.6):
    """
    生成空间位置掩码（保留中间位置）
    x shape: (B, C, N) 其中 N = patch_size^2 是空间位置总数
    返回: (B, N) 的布尔掩码，True 表示需要遮蔽的位置
    """
    B, C, N = x.shape
    mid_idx = (N + 1) // 2  # 计算中间位置索引
    # 生成随机排序索引（排除中间位置）
    rand_scores = torch.rand(B, N, device=x.device)
    if N == 1:
        mid_idx = 0
    rand_scores[:, mid_idx] = -1  # 确保中间位置不会被选中
    # 计算需要遮蔽的位置数（至少保留中间位置）
    k = min(int(N * mask_ratio), N - 1)
    # 选择要遮蔽的位置索引（不包括中间位置）
    selected_indices = rand_scores.argsort(dim=1, descending=True)[:, :k]
    # 创建布尔掩码
    mask = torch.zeros(B, N, dtype=torch.bool, device=x.device)
    mask.scatter_(1, selected_indices, True)
    return mask

depth = 3
heads = 2
if args.dataset == 'IndianPines':
    hsi_data = loadmat('./data/IndianPines/Indian_pines_corrected.mat')['indian_pines_corrected']
    hsi_labels = loadmat('./data/IndianPines/Indian_pines_gt.mat')['indian_pines_gt']
    depth = 3
    heads = 2
elif args.dataset == 'PaviaU':
    hsi_data = loadmat('./data/PaviaU/PaviaU.mat')['paviaU']
    hsi_data = hsi_data[:, :, :100]
    hsi_labels = loadmat('./data/PaviaU/PaviaU_gt.mat')['paviaU_gt']
    depth = 4
    heads = 5
elif args.dataset == 'Houston13':
    hsi_data = loadmat('./data/Houston2013/Houston.mat')['Houston2013']
    hsi_labels = loadmat('./data/Houston2013/Houston_gt.mat')['Houston2013_gt']
    depth = 5
    heads = 2
    args.sample_step = 3
else:
    raise ValueError("Unknown dataset")

builder = HSIDatasetBuilder(
    data_name=args.dataset,
    data=hsi_data,
    labels=hsi_labels,
    patch_size=args.patch_size,
    verbose=True
)
num_classes = builder.num_classes
band = builder.channels
height = builder.height
width = builder.width

# pretrain
if args.flag == 'pretrain':
    pre_patches, pre_labels = builder.build_pretrain_dataset(sample_step=args.sample_step)
    pre_patches = torch.from_numpy(pre_patches.transpose(0, 2, 1)).float()
    pre_labels = torch.from_numpy(pre_labels).long()
    print("pre_patches shape: ", pre_patches.shape)

    # 生成掩码
    spectral_mask = generate_spectral_mask(pre_patches, mask_ratio=args.SpeMask_ratio)
    spatial_mask = generate_spatial_mask(pre_patches, mask_ratio=args.SpaMask_ratio)

    pre_datasets = Data.TensorDataset(
        pre_patches,
        spectral_mask,
        spatial_mask,
    )
    # 创建数据加载器
    pre_loader = Data.DataLoader(
        pre_datasets,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True
    )
    pretrain(args, pre_loader, 0, band, depth, heads)

# finetune
if args.flag == 'finetune' or args.flag == 'test':
    ft_patches, ft_labels, val_patches, val_labels = builder.build_finetune_dataset(
        size_per_class=args.samples,
    )
    ft_patches = torch.from_numpy(ft_patches.transpose(0, 2, 1)).float()
    ft_labels = torch.from_numpy(ft_labels).long()
    val_patches = torch.from_numpy(val_patches.transpose(0, 2, 1)).float()
    val_labels = torch.from_numpy(val_labels).long()
    ft_datasets = Data.TensorDataset(ft_patches, ft_labels)
    val_datasets = Data.TensorDataset(val_patches, val_labels)

    # 创建数据加载器
    ft_loader = Data.DataLoader(
        ft_datasets,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True
    )
    val_loader = Data.DataLoader(
        val_datasets,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True
    )


    if args.flag == 'finetune':
        finetune(args, ft_loader, num_classes, band, depth, heads)

    if args.flag == 'test':
        all_patches, all_labels = builder.build_pretrain_dataset()
        all_patches = torch.from_numpy(all_patches.transpose(0, 2, 1)).float()
        all_labels = torch.from_numpy(all_labels).long()
        all_pos = np.argwhere(builder.labels >= 0)
        all_datasets = Data.TensorDataset(all_patches, all_labels)
        all_loader = Data.DataLoader(
            all_datasets,
            batch_size=args.batch_size,
            shuffle=False,
            pin_memory=True
        )

        model = build_model(args, num_classes, band, depth, heads)
        checkpoint = torch.load(args.model_path, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.cuda()
        print(model)
        targets, predicts = validate(model, val_loader)
        metrics = calculate_metrics(targets, predicts)
        class_acc_str = " | ".join([f"C{i}:{acc:.2%}" for i, acc in enumerate(metrics['class_acc'])])
        print(f"验证集结果：\n"
              f"OA: {metrics['OA']:.2%} | "
              f"AA: {metrics['AA']:.2%} | "
              f"Kappa: {metrics['Kappa']:.4f}\n"
              f"Class Acc: {class_acc_str}")

        # 保存结果到日志文件
        log_dir = 'logs/'
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"{args.dataset}_{time.strftime('%Y%m%d_%H%M%S')}.log")
        save_results_to_log(args, metrics, class_acc_str, log_file)
        print(f"Results saved to {log_file}")

        # 绘制分类图
        tars, pres = validate(model, all_loader)
        os.makedirs('cls_map', exist_ok=True)
        plot_classification(height, width, all_pos, pres, path='cls_map/' + args.dataset)
        print("Classification map saved to cls_map/" + args.dataset)

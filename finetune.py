import csv
import os
import time

import numpy as np
import torch
from torch import nn, GradScaler
from torch.amp import autocast
from torch.utils import data as Data

from sklearn.metrics import confusion_matrix, cohen_kappa_score
from timm.utils.metrics import AverageMeter, accuracy

from models import build_model
from utils.util import get_lr_config
from utils.visualise import plot_training_curve, plot_predictions, plot_classification
from utils.logger import ensure_log_dirs, create_logger, Logger


def train_one_epoch(model, label_train_loader, criterion, optimizer, logger=None):
    # 初始化计量器和收集器
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    all_targets, all_preds = [], []

    # 初始化进度条
    from utils.logger import ProgressBar
    progress_bar = ProgressBar(total=len(label_train_loader), prefix='训练', suffix='', length=40)

    # 记录训练开始
    if logger:
        logger.info(f"开始训练 - 批次数: {len(label_train_loader)}")
        logger.info(f"学习率: {optimizer.param_groups[0]['lr']:.6f}")

    model.train()
    for batch_idx, batch_data in enumerate(label_train_loader):
        batch_img, batch_target = batch_data
        # 数据迁移到设备
        batch_img = batch_img.cuda(non_blocking=True)
        batch_target = batch_target.cuda(non_blocking=True)
        # 前向传播
        batch_pred = model(batch_img)
        # 梯度清零
        optimizer.zero_grad(set_to_none=True)  # 内存优化
        # 计算分类损失
        loss = criterion(batch_pred, batch_target)
        # 反向传播
        loss.backward()
        optimizer.step()
        # 收集指标
        acc1, acc5 = accuracy(batch_pred.detach(), batch_target, topk=(1, 5))
        preds = torch.argmax(batch_pred.detach(), dim=1)
        targets = batch_target
        prec1 = (acc1,)
        batch_size = batch_img.size(0)

        loss_meter.update(loss.item(), batch_size)
        acc_meter.update(prec1[0].item(), batch_size)
        all_targets.append(targets.cpu().numpy())
        all_preds.append(preds.cpu().numpy())

        # 更新进度条
        batch_info = f"批次: {batch_idx + 1}/{len(label_train_loader)}"
        suffix = f"损失: {loss.item():.4f} | 准确率: {prec1[0].item():.2f}%"
        progress_bar.update(batch_idx + 1, suffix=suffix, batch_info=batch_info)

    return (
        acc_meter.avg,
        loss_meter.avg,
        np.concatenate(all_targets),
        np.concatenate(all_preds)
    )


def finetune(args, label_train_loader, num_classes, band, depth, heads):
    # 确保所有必要的目录存在
    ensure_log_dirs()

    # 创建日志记录器
    log_dir = 'checkpoint/finetune/logs'
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    log_file = f"{args.dataset}_{timestamp}.log"
    logger = create_logger(name=f"Finetune_{args.dataset}", log_dir=log_dir, log_file=log_file)

    # 记录训练配置
    logger.info(f"{'=' * 20} 训练配置 {'=' * 20}")
    logger.info(f"数据集: {args.dataset}")
    logger.info(f"批次大小: {args.batch_size}")
    logger.info(f"训练轮数: {args.epochs}")
    logger.info(f"预训练模型: {args.pt_model}")
    logger.info(f"{'=' * 50}")

    # 构建模型
    model = build_model(args, num_classes, band, depth, heads, is_pretrain=False)
    # 加载预训练模型（仅加载encoder部分）
    if args.pt_model is not None:
        pretrained_path = 'checkpoint/pretrain/' + args.pt_model
        pretrained = torch.load(pretrained_path, weights_only=True)
        # 创建新字典，只保留encoder权重并去除前缀
        encoder_weights = {
            k.replace("encoder.", ""): v
            for k, v in pretrained['model_state_dict'].items()
            if k.startswith('encoder') and not k.startswith('encoder.mask_token')
        }
        # 加载到当前模型
        model.load_state_dict(encoder_weights, strict=False)
        logger.info(f"加载预训练编码器权重: {pretrained_path}")
        print(f"Loaded encoder weights from {pretrained_path}")

    model = model.cuda()
    print(model)
    model_size = sum(p.numel() for p in model.parameters()) / 1e6
    logger.info(f"模型大小: {model_size:.4f}M")
    print("Model size: {:.4f}M".format(model_size))
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss().cuda()
    # 优化器配置
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=5e-4,  # 基础学习率
        weight_decay=0.005,  # 权重衰减
        betas=(0.9, 0.999)  # 动量参数
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=10,  # 每10个epoch调整一次学习率
        gamma=0.9  # 学习率衰减因子
    )

    # 加载检查点
    if args.pt_model is not None:
        checkpoint_path = f'checkpoint/finetune/{args.pt_model.replace(".pth", "")}_ft.pth'
    else:
        checkpoint_path = f'checkpoint/finetune/{args.dataset}_train.pth'
    times = 0
    train_losses = []
    train_accs = []
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, weights_only=True)
        logger.info(f"加载检查点: {checkpoint_path}")
        print("Load checkpoint from {}".format(checkpoint_path))
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        times = checkpoint['time']
        start_epoch = checkpoint['epoch'] + 1
        train_losses = checkpoint['train_losses']
        train_accs = checkpoint['train_accs']
        logger.info(f"从轮次 {start_epoch} 继续训练")
    else:
        start_epoch = 0
        logger.info("未找到检查点，从头开始训练")
    # if args.resume:
    # 训练模型
    print("------------------Start training------------------")
    logger.info("开始训练过程")
    best_loss = float('inf')
    tick = time.time()
    for epoch in range(start_epoch, args.epochs):
        logger.info(f"{'=' * 15} 轮次 {epoch + 1}/{args.epochs} {'=' * 15}")
        # 记录当前学习率
        current_lr = optimizer.param_groups[0]['lr']
        logger.info(f"当前学习率: {current_lr:.6f}")
        # 训练一个轮次
        avg_acc, avg_loss, targets, predicts = train_one_epoch(model, label_train_loader, criterion, optimizer, logger)
        scheduler.step()
        # 记录训练指标
        train_losses.append(avg_loss)
        train_accs.append(avg_acc)

        # 记录轮次结果
        logger.info(f"轮次 {epoch + 1} 结果 - 损失: {avg_loss:.8f} | 准确率: {avg_acc:.4f}")
        print("Epoch: {:03d} train_loss: {:.8f} train_acc: {:.4f}".format(epoch + 1, avg_loss, avg_acc))
        ckp_dir = 'checkpoint/finetune/'
        if not os.path.exists(ckp_dir):
            os.makedirs(ckp_dir)
        if epoch % 10 == 0 or epoch == args.epochs - 1:
            tock2 = time.time()
            # 保存检查点
            torch.save({'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'time': tock2 - tick,
                        'train_losses': train_losses,
                        'train_accs': train_accs,
                        },
                       checkpoint_path)
            logger.info(f"保存检查点到: {checkpoint_path}")
        if avg_loss < best_loss:
            best_loss = avg_loss
            output = args.output_dir
            os.makedirs(output, exist_ok=True)
            best_model_path = args.output_dir + '/' + args.dataset + '.pth'
            torch.save({'model_state_dict': model.state_dict()}, best_model_path)
            logger.info(f"发现更好的模型 (损失: {best_loss:.8f})，保存到: {best_model_path}")
    tock = time.time()
    training_time = tock - tick + times
    logger.info(f"训练完成，总用时: {training_time:.2f}秒")
    print("Training time: {:.2f}s".format(training_time))
    print("-----------------Training finished-----------------\n")
    # 绘制训练曲线
    logger.info("绘制训练曲线")
    plot_training_curve(train_losses, train_accs)
    # 关闭日志记录器
    logger.info("任务完成，关闭日志记录器")






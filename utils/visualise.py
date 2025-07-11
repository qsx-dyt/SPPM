import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from scipy.io import loadmat
from sklearn.metrics import ConfusionMatrixDisplay


def plot_training_curve(train_losses, train_accs=None):
    """绘制训练曲线"""
    plt.figure(figsize=(12, 5))

    # 绘制损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.title('Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # 绘制准确率曲线
    if train_accs is not None:
        plt.subplot(1, 2, 2)
        plt.plot(train_accs, label='Train Acc')
        plt.title('Accuracy Curve')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()

    plt.tight_layout(w_pad=5)  # 增加子图间距
    plt.show()


def plot_predictions(target, predict):
    """
    可视化预测结果，包括混淆矩阵和预测分布

    参数:
        target (array-like): 真实标签
        predict (array-like): 预测标签
    """
    target = np.asarray(target)
    predict = np.asarray(predict)

    fig = plt.figure(figsize=(15, 6))
    ax1 = plt.subplot(1, 2, 1)

    # 修改后的混淆矩阵绘制部分
    disp = ConfusionMatrixDisplay.from_predictions(
        target, predict,
        cmap=plt.cm.Blues,
        normalize='true',
        ax=ax1,
        values_format=".2f"
    )
    ax1.set_title('Confusion Matrix')
    ax1.tick_params(axis='both', labelsize=8)

    if disp.text_ is not None:
        for text in disp.text_.ravel():
            text.set_fontsize(8)
            text.set_horizontalalignment('center')
            text.set_verticalalignment('center')

    # 绘制预测分布
    ax2 = plt.subplot(1, 2, 2)
    unique_classes = np.unique(np.concatenate((target, predict)))
    bins = np.arange(min(unique_classes), max(unique_classes) + 1.5) - 0.5

    plt.hist([target, predict],
             bins=bins,
             label=['True', 'Predicted'],
             alpha=0.7,
             align='mid')
    plt.title('Prediction Distribution')
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.legend()

    # 调整布局
    plt.tight_layout(w_pad=5)  # 增加子图间距
    plt.show()



def plot_classification(height, width, combined_pos, pres, path):
    color_matrix = loadmat('utils/ColorMap.mat')['mycolormap']
    # 创建并填充预测矩阵
    prediction_matrix = np.zeros((height, width), dtype=float)
    for i in range(combined_pos.shape[0]):
        prediction_matrix[combined_pos[i, 0], combined_pos[i, 1]] = pres[i] + 1
    plt.subplot(1, 1, 1)
    plt.imshow(prediction_matrix,
               cmap=colors.ListedColormap(color_matrix),
               vmin=0,  # 设置最小值对应背景色
               vmax=color_matrix.shape[0])  # 设置最大值范围
    plt.xticks([])
    plt.yticks([])
    plt.savefig(f'{path}.png', dpi=300, bbox_inches='tight', pad_inches=0)
    plt.savefig(f'{path}.pdf', dpi=300, bbox_inches='tight', pad_inches=0)
    # plt.savefig(f'{path}.eps', dpi=300, bbox_inches='tight', format='eps', pad_inches=0)
    plt.show()




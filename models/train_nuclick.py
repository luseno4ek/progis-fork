from efficientunet import *
from UNet import *
import torch.optim.lr_scheduler as lr_scheduler
import time
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BatchNorm2d
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision import models

import torch.optim as optim
from sklearn.model_selection import train_test_split
from tqdm import tqdm


import scipy.ndimage as ndi


import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from scipy.ndimage.morphology import distance_transform_edt
import scipy.ndimage as ndi
from skimage.morphology import skeletonize_3d


# device_id =  1# Specify which GPU to use, e.g., GPU 0
# torch.cuda.set_device(device_id)

# 设置随机种子
random.seed(42)  # 你可以选择任何整数作为种子

#################  nuclick  ###################

def conv_bn_relu(in_channels, out_channels, kernel_size=3, stride=1, dilation=1, actv='relu', use_bias=False, use_regularizer=False, do_batch_norm=True):
    layers = []
    padding = (kernel_size - 1) // 2 * dilation
    if use_regularizer:
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation=dilation, bias=use_bias, padding_mode='zeros'))
    else:
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation=dilation, bias=use_bias))
    if do_batch_norm:
        layers.append(BatchNorm2d(out_channels))
    if actv != 'None':
        if actv == 'relu':
            layers.append(nn.ReLU(inplace=True))
        elif actv == 'selu':
            layers.append(nn.SELU(inplace=True))
    return nn.Sequential(*layers)

class MultiScaleConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, sizes, dilatation_rates, is_dense=True):
        super(MultiScaleConvBlock, self).__init__()
        self.is_dense = is_dense
        if is_dense:
            self.conv0 = conv_bn_relu(in_channels, 4*out_channels, kernel_size=1)
        else:
            self.conv0 = nn.Identity()
        
        self.conv1 = conv_bn_relu(4*out_channels if is_dense else in_channels, out_channels, kernel_size=sizes[0], dilation=dilatation_rates[0])
        self.conv2 = conv_bn_relu(4*out_channels if is_dense else in_channels, out_channels, kernel_size=sizes[1], dilation=dilatation_rates[1])
        self.conv3 = conv_bn_relu(4*out_channels if is_dense else in_channels, out_channels, kernel_size=sizes[2], dilation=dilatation_rates[2])
        self.conv4 = conv_bn_relu(4*out_channels if is_dense else in_channels, out_channels, kernel_size=sizes[3], dilation=dilatation_rates[3])
        
        if is_dense:
            self.conv_out = conv_bn_relu(4*out_channels, out_channels, kernel_size=3)
    
    def forward(self, x):
        if self.is_dense:
            x = self.conv0(x)
        conv1_out = self.conv1(x)
        conv2_out = self.conv2(x)
        conv3_out = self.conv3(x)
        conv4_out = self.conv4(x)
        output_map = torch.cat([conv1_out, conv2_out, conv3_out, conv4_out], dim=1)
        if self.is_dense:
            output_map = self.conv_out(output_map)
            output_map = torch.cat([x, output_map], dim=1)
        return output_map

class ResidualConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, actv='relu', use_bias=False, use_regularizer=False, dilation=1):
        super(ResidualConv, self).__init__()
        self.actv = actv
        self.conv1 = conv_bn_relu(in_channels, out_channels, kernel_size, dilation=dilation, actv='None', use_bias=use_bias, use_regularizer=use_regularizer, do_batch_norm=True)
        self.conv2 = conv_bn_relu(out_channels, out_channels, kernel_size, dilation=dilation, actv='None', use_bias=use_bias, use_regularizer=use_regularizer, do_batch_norm=True)

        
    def forward(self, x):
        out_1 = self.conv1(x)
        out_2 = self.conv2(out_1)
        out = out_1 + out_2
        if self.actv == 'relu':
            out = F.relu(out)
        elif self.actv == 'selu':
            out = F.selu(out)
        return out

class MultiScaleResUnet(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(MultiScaleResUnet, self).__init__()
        self.conv1_1 = conv_bn_relu(in_channels, 64, kernel_size=7)
        self.conv1_2 = conv_bn_relu(64, 32, kernel_size=5)
        self.conv1_3 = conv_bn_relu(32, 32, kernel_size=3)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        
        self.res2_1 = ResidualConv(32, 64)
        self.res2_2 = ResidualConv(64, 64)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        
        self.res3_1 = ResidualConv(64, 128)
        self.msconv3 = MultiScaleConvBlock(128, 32, sizes=[3, 3, 5, 5], dilatation_rates=[1, 3, 3, 6], is_dense=False)
        self.res3_2 = ResidualConv(128, 128)
        self.pool3 = nn.MaxPool2d(kernel_size=2)
        
        self.res4_1 = ResidualConv(128, 256)
        self.res4_2 = ResidualConv(256, 256)
        self.res4_3 = ResidualConv(256, 256)
        self.pool4 = nn.MaxPool2d(kernel_size=2)
        
        self.res5_1 = ResidualConv(256, 512)
        self.res5_2 = ResidualConv(512, 512)
        self.res5_3 = ResidualConv(512, 512)
        self.pool5 = nn.MaxPool2d(kernel_size=2)
        
        self.res51_1 = ResidualConv(512, 1024)
        self.res51_2 = ResidualConv(1024, 1024)
        
        self.up6 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.res6_1 = ResidualConv(1024, 512)
        self.res6_2 = ResidualConv(512, 256)
        
        self.up7 = nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2)
        self.res7_1 = ResidualConv(512, 256)
        self.msconv7 = MultiScaleConvBlock(256, 64, sizes=[3, 3, 5, 5], dilatation_rates=[1, 3, 2, 3], is_dense=False)
        self.res7_2 = ResidualConv(256, 256)
        
        self.up8 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.res8_1 = ResidualConv(256, 128)
        self.res8_2 = ResidualConv(128, 128)
        
        self.up9 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.res9_1 = ResidualConv(128, 64)
        self.msconv9 = MultiScaleConvBlock(64, 16, sizes=[3, 3, 5, 7], dilatation_rates=[1, 3, 3, 6], is_dense=False)
        self.res9_2 = ResidualConv(64, 64)
        
        self.up10 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.conv10_1 = conv_bn_relu(64, 64)
        self.conv10_2 = conv_bn_relu(64, 32)
        self.conv10_3 = conv_bn_relu(32, 32)
        
        self.conv11 = nn.Conv2d(32, num_classes, kernel_size=1)
    
    def forward(self, inputs):
        
        
        conv1 = self.conv1_1(inputs)
        conv1 = self.conv1_2(conv1)
        conv1 = self.conv1_3(conv1)
        pool1 = self.pool1(conv1)
        
        conv2 = self.res2_1(pool1)
        conv2 = self.res2_2(conv2)
        pool2 = self.pool2(conv2)
        
        conv3 = self.res3_1(pool2)
        conv3 = self.msconv3(conv3)
        conv3 = self.res3_2(conv3)
        pool3 = self.pool3(conv3)
        
        conv4 = self.res4_1(pool3)
        conv4 = self.res4_2(conv4)
        conv4 = self.res4_3(conv4)
        pool4 = self.pool4(conv4)
        
        conv5 = self.res5_1(pool4)
        conv5 = self.res5_2(conv5)
        conv5 = self.res5_3(conv5)
        pool5 = self.pool5(conv5)
        
        conv51 = self.res51_1(pool5)
        conv51 = self.res51_2(conv51)
        
        up6 = self.up6(conv51)
        up6 = torch.cat([up6, conv5], dim=1)
        conv6 = self.res6_1(up6)
        conv6 = self.res6_2(conv6)
        
        up7 = self.up7(conv6)
        up7 = torch.cat([up7, conv4], dim=1)
        conv7 = self.res7_1(up7)
        conv7 = self.msconv7(conv7)
        conv7 = self.res7_2(conv7)
        
        up8 = self.up8(conv7)
        up8 = torch.cat([up8, conv3], dim=1)
        conv8 = self.res8_1(up8)
        conv8 = self.res8_2(conv8)
        
        up9 = self.up9(conv8)
        up9 = torch.cat([up9, conv2], dim=1)
        conv9 = self.res9_1(up9)
        conv9 = self.msconv9(conv9)
        conv9 = self.res9_2(conv9)
        
        up10 = self.up10(conv9)
        up10 = torch.cat([up10, conv1], dim=1)
        conv10 = self.conv10_1(up10)
        conv10 = self.conv10_2(conv10)
        conv10 = self.conv10_3(conv10)
        
        conv11 = self.conv11(conv10)
        output = F.sigmoid(conv11)
        
        # min_val = torch.amin(output, dim=(2, 3), keepdim=True)[0]  # 找到每个通道的最小值
        # max_val = torch.amax(output, dim=(2, 3), keepdim=True)[0]  # 找到每个通道的最大值
        # normalized_output = (output - min_val) / (max_val - min_val + 1e-8)  # 防止除以0
        
        # 使用SLIC进行超像素分割
        # segments_slic = slic(x, n_segments = 100, sigma = 5)
        
        return output

#################  nuclick  ###################



### 评价指标

def dice_coeff(y_true, y_pred, a=1., b=1.):
    y_true_flat = y_true.view(-1)
    y_pred_flat = y_pred.view(-1)
    intersection = torch.sum(y_true_flat * y_pred_flat)
    return (2. * intersection + a) / (torch.sum(y_true_flat) + torch.sum(y_pred_flat) + b)

def dice_loss(y_true, y_pred, a=1., b=1.):
    return 1.0 - dice_coeff(y_true, y_pred, a=a, b=b)


def compute_iou(pred, target, cls):
    pred_cls = (pred == cls)
    target_cls = (target == cls)
    
    intersection = np.logical_and(pred_cls, target_cls).sum()
    union = np.logical_or(pred_cls, target_cls).sum()
    
    if union == 0:
        return float('nan')  # 如果类别在预测和实际中都不存在，忽略此类别
    else:
        return intersection / union

def compute_miou_binary(pred, target):
    # 计算背景类别（0）的IOU
    # iou_background = compute_iou(pred, target, 0)
    # 计算前景类别（1）的IOU
    iou_foreground = compute_iou(pred, target, 1)
    
    # 计算mIOU，忽略nan值
    # miou = np.nanmean([iou_background, iou_foreground])
    return iou_foreground

def calculate_binary_segmentation_accuracy(preds, labels):
    """
    计算前景背景分割的像素级准确率
    :param preds: 模型的预测值，形状为 [batch_size, height, width] 或 [batch_size, 1, height, width]
    :param labels: 真实标签，形状为 [batch_size, 1, height, width]
    :return: 每个样本的准确率和平均准确率
    """
    if preds.dim() == 4:
        preds = preds.squeeze(1)  # 去掉频道维度
    
    if labels.dim() == 4:
        labels = labels.squeeze(1)  # 去掉频道维度
    
    assert preds.shape == labels.shape, "预测值和标签的形状必须一致"

    preds = (preds > 0.5).float()  # 阈值 0.5，用于二分类
    
    correct = (preds == labels).float().sum(dim=[1, 2])  # 每个样本的正确预测像素数
    total_pixels_per_sample = labels.size(1) * labels.size(2)
    
    accuracy_per_sample = correct / total_pixels_per_sample  # 每个样本的准确率
    mean_accuracy = accuracy_per_sample.mean().item()  # 批次中的平均准确率
    
    return accuracy_per_sample, mean_accuracy


### 计算outputs中预测值为1且masks中值也为1的像素点所占的比例，计算outputs中预测值为1但masks中值为0的像素点所占的比例
def calculate_metrics(outputs, masks):
    
    # 最大最小归一化
    outputs_min = outputs.min()
    outputs_max = outputs.max()
    
    # 处理极端情况，防止除以0
    if outputs_max - outputs_min > 0:
        outputs = (outputs - outputs_min) / (outputs_max - outputs_min)
    else:
        outputs = torch.zeros_like(outputs)  # 如果没有变化，直接设为0
    
    
    # 确保outputs和masks的值为0或1
    outputs = (outputs > 0.6).float()  # 阈值设定为0.5，预测值大于0.5的被认为是1
    masks = masks.float()
    
    # 预测为1的像素点
    outputs_1 = outputs == 1
    # 真实值为1的像素点
    masks_1 = masks == 1
    # 真实值为0的像素点
    masks_0 = masks == 0
    
    # 计算预测为1且真实值为1的像素数
    true_positive = (outputs_1 & masks_1).sum().item()
    
    # 计算预测为1且真实值为0的像素数
    false_positive = (outputs_1 & masks_0).sum().item()
    
    # 计算总的预测为1的像素数
    total_pred_1 = outputs_1.sum().item()
    
    # 防止除以0的情况
    if total_pred_1 == 0:
        return 0, 0
    
    # 计算比例
    true_positive_ratio = true_positive / total_pred_1
    false_positive_ratio = false_positive / total_pred_1
    
    return true_positive_ratio, false_positive_ratio


#########################################################################################################
#计算误差区域的形态学骨架

def generateGuidingSignal(binaryMask):
    # binaryMask = binaryMask.squeeze(0)  # Remove the batch dimension if it's (1, H, W)
    binaryMask = binaryMask.to(torch.uint8)
    
    if binaryMask.sum() > 1:
        # Compute distance transform (move to CPU for NumPy operations)
        distance_map = distance_transform_edt(binaryMask.cpu().numpy())
        distance_map = torch.tensor(distance_map, dtype=torch.float32, device=binaryMask.device)
        
        # Calculate mean and std (ensure they are on CPU before NumPy operations)
        tempMean = distance_map.mean().cpu().numpy()
        tempStd = distance_map.std().cpu().numpy()
        
        # Random threshold based on mean and std
        tempThresh = np.random.uniform(tempMean - tempStd, tempMean + tempStd)
        tempThresh = torch.tensor(tempThresh, device=binaryMask.device)
        
        if tempThresh < 0:
            tempThresh = np.random.uniform(tempMean / 2, tempMean + tempStd / 2)
            tempThresh = torch.tensor(tempThresh, device=binaryMask.device)
        
        # Apply threshold to get new mask
        newMask = distance_map > tempThresh
        if newMask.sum() == 0:
            newMask = distance_map > (tempThresh / 2)
        
        if newMask.sum() == 0:
            newMask = binaryMask

        # Skeletonize (use skimage and convert back to tensor)
        skel = skeletonize_3d(newMask.cpu().numpy())
        skel = torch.tensor(skel, dtype=torch.float32, device=binaryMask.device)
    else:
        skel = torch.zeros_like(binaryMask, dtype=torch.float32, device=binaryMask.device)

    return skel

def processMasks_signal(GT_mask_all):
    """
    批量处理GT_mask和pred_mask，计算每个样本的前景和背景骨架信号。
    参数:
        pred_masks: 预测的mask，形状为 [batch_size, 1, H, W]。
        GT_masks: 真值mask，形状为 [batch_size, 1, H, W]。
    返回:
        输出张量，形状为 [batch_size, 2, H, W]。
        每个样本的第0通道为前景区域骨架信号，第1通道为背景区域骨架信号。
    """

    batch_size, _, H, W = GT_mask_all.shape

    # 初始化输出张量
    output = torch.zeros(batch_size, 2, H, W, device=GT_mask_all.device, dtype=torch.float32)

    for i in range(batch_size):
        # 取出当前样本的预测和真值mask
        GT_mask = GT_mask_all[i].squeeze(0)      # [H, W]
 
        # 计算前景区域的骨架信号
        fg_skeleton = generateGuidingSignal(GT_mask) if GT_mask.sum() > 0 else torch.zeros_like(GT_mask, dtype=torch.float32, device=GT_mask.device)  # 如果fg为全0，创建一个全零张量

        # 合并前景和背景骨架信号到输出
        output[i, 0] = fg_skeleton  # 前景骨架信号

    return output


##########################################################################################################



# Define the dataset class
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, images_dir, masks_dir, signal_dir,  filenames):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.signal_dir = signal_dir
        # self.suppixel_dir = suppixel_dir
        self.filenames = filenames

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        # 根据索引获取当前文件名
        filename = self.filenames[idx]
        
        # 加载图像、掩模和超像素文件
        image_path = os.path.join(self.images_dir, filename)
        mask_path = os.path.join(self.masks_dir, filename)
        signal_path = os.path.join(self.signal_dir, filename)
        # suppixel_path = os.path.join(self.suppixel_dir, filename)

        image = np.load(image_path)
        mask = np.load(mask_path)
        signal = np.load(signal_path)
        # suppixel = np.load(suppixel_path)

        # 转换为 PyTorch 张量
        image = torch.tensor(image.transpose(2, 0, 1), dtype=torch.float32)  # (channels, height, width)
        mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)  # (1, height, width)
        signal = torch.tensor(signal, dtype=torch.float32)  # (channels, height, width)
        # suppixel = torch.tensor(suppixel.transpose(2, 0, 1), dtype=torch.float32)  # (1, height, width)

        return image, mask, signal
    
# 获取文件夹中的文件名
def get_filenames_from_folder(folder_path):
    return [filename for filename in os.listdir(folder_path) if filename.endswith('.npy')]    
    


# 设置训练集和验证集的文件夹路径
# train_images_dir = "/data_nas/gjs/ISF_pixel_level_data/BCSS_x10_reinhard_cut/Train_val_step256_no_filling/train/tumor/image_npy"
# train_masks_dir = "/data_nas/gjs/ISF_pixel_level_data/BCSS_x10_reinhard_cut/Train_val_step256_no_filling/train/tumor/mask_npy"
# train_signal_dir = '/data_nas/gjs/ISF_pixel_level_data/BCSS_x10_reinhard_cut/Train_val_step256_no_filling/train/tumor/signal_max_point_npy'

# val_images_dir = "/data_nas/gjs/ISF_pixel_level_data/BCSS_x10_reinhard_cut/Train_val_step256_no_filling/val/tumor/image_npy"
# val_masks_dir = "/data_nas/gjs/ISF_pixel_level_data/BCSS_x10_reinhard_cut/Train_val_step256_no_filling/val/tumor/mask_npy"
# val_signal_dir = '/data_nas/gjs/ISF_pixel_level_data/BCSS_x10_reinhard_cut/Train_val_step256_no_filling/val/tumor/signal_max_point_npy'

# ИЗМЕНЕННЫЕ ПУТИ ДЛЯ ТЕСТИРОВАНИЯ НА МАЛОМ ДАТАСЕТЕ (ПАТЧИ 512x512)
path = "/Users/olesyaindychko/Documents/phd/code/ProGIS/data/patches"
i = 1
cls = 'tumor'  # изменено с 'all_class' на 'tumor'

# Убран цикл for i in range(1,4) - тестируем только на fold_1
if True:  # оставлено для сохранения отступов

    train_images_dir = f"{path}/fold_{i}/train/{cls}/image_npy"
    train_masks_dir = f"{path}/fold_{i}/train/{cls}/mask_npy"
    train_signal_dir = f'{path}/fold_{i}/train/{cls}/signal_all_line_npy'
    val_images_dir = f"{path}/fold_{i}/val/{cls}/image_npy"
    val_masks_dir = f"{path}/fold_{i}/val/{cls}/mask_npy"
    val_signal_dir = f'{path}/fold_{i}/val/{cls}/signal_all_line_npy'

    # cls = 'others'
    # train_images_dir = f"/data_nas2/gjs/ISF_pixel_level_data/BCSS_x10_reinhard_cut/150/train/{cls}/image_npy"
    # train_masks_dir = f"/data_nas2/gjs/ISF_pixel_level_data/BCSS_x10_reinhard_cut/150/train/{cls}/mask_npy"
    # train_signal_dir = f'/data_nas2/gjs/ISF_pixel_level_data/BCSS_x10_reinhard_cut/150/train/{cls}/signal_max_point_npy'
    # val_images_dir = f"/data_nas2/gjs/ISF_pixel_level_data/BCSS_x10_reinhard_cut/150/val/{cls}/image_npy"
    # val_masks_dir = f"/data_nas2/gjs/ISF_pixel_level_data/BCSS_x10_reinhard_cut/150/val/{cls}/mask_npy"
    # val_signal_dir = f'/data_nas2/gjs/ISF_pixel_level_data/BCSS_x10_reinhard_cut/150/val/{cls}/signal_max_point_npy'

    # 获取训练集和验证集的文件名
    train_filenames = get_filenames_from_folder(train_images_dir)
    val_filenames = get_filenames_from_folder(val_images_dir)
    # 创建自定义数据集类的实例
    train_dataset = CustomDataset(train_images_dir, train_masks_dir, train_signal_dir, train_filenames)
    val_dataset = CustomDataset(val_images_dir, val_masks_dir, val_signal_dir, val_filenames)
    # ИЗМЕНЕНО для теста: batch_size=4, num_workers=0 (для macOS)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=0)
    # 创建模型实例
    # model = get_efficientunet_b0(out_channels=1, concat_input=True, pretrained=False).cuda()
    model = MultiScaleResUnet(in_channels=5, num_classes=1)
    # model = UNet(n_channels=3, n_classes=1)
    # model.load_state_dict(torch.load('/home/gjs/ISF_nuclick/check_points_BCSS/nuclick_512_ckpt/nuclick_512_all_class_100.pth'))
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=4e-4)
    loss_fn = nn.BCELoss()

    # Training function
    def train_model(model, train_loader, val_loader, loss_fn, optimizer, epochs=50):
        best_dice = 0.0
        # ИЗМЕНЕНО: device='cpu' для macOS (или 'cuda:0' если есть GPU)
        device = 'cpu'  # изменено с 'cuda:1'
        model.to(device)

        for epoch in range(epochs):
            model.train()
            train_loss = 0.0
            train_dice_score = 0.0
            train_accuracy = 0.0  # 用于累积准确率
            train_true_positive_ratio, train_false_positive_ratio = 0.0 , 0.0
            
            train_loader_tqdm = tqdm(train_loader, total=len(train_loader), desc=f'Epoch {epoch+1}/{epochs} [train]', unit='batch')
            
                    
            for images, masks, aux_inputs in train_loader_tqdm:
                
                
                images, aux_inputs, masks= images.to(device), aux_inputs.to(device), masks.to(device)
                # aux_input_new = processMasks_signal(masks)
                
                input = torch.cat([images, aux_inputs], dim=1)
                
                optimizer.zero_grad()

                # outputs, all_superpixel_features = model(images, aux_inputs, superpixels)
                outputs = model(input)
            
                
                loss = dice_loss(outputs, masks)
                # loss = calc_dc_loss_sp(masks, all_pixel_features, superpixels , fg_proto_features)
                
                # loss = dice_loss(first_seg, masks)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item() * images.size(0)
                
                
                outputs = (outputs >= 0.5).int()
                
                train_dice_score += dice_coeff(outputs, masks).item() * images.size(0)
                
                # 计算准确率
                _, batch_accuracy = calculate_binary_segmentation_accuracy(outputs, masks)
                train_accuracy += batch_accuracy * images.size(0)  # 将每个批次的准确率加权累积
                
                # true_positive_ratio_1 = calculate_spp(outputs, masks , suppixels)* images.size(0)
                # train_true_positive_ratio += true_positive_ratio_1 
                
                
                # true_positive_ratio_1, false_positive_ratio_1 = calculate_metrics(outputs, masks)
                # train_true_positive_ratio += true_positive_ratio_1* images.size(0)
                # train_false_positive_ratio += false_positive_ratio_1* images.size(0)
                
                
            train_loss /= len(train_loader.dataset)
            
            train_dice_score /= len(train_loader.dataset)
            
            train_accuracy /= len(train_loader.dataset)  # 计算平均准确率
            
            train_true_positive_ratio /= len(train_loader.dataset)
            
            # train_false_positive_ratio /= len(train_loader.dataset)
            
            ###
            model.eval()
            val_loss = 0.0
            dice_score = 0.0
            val_accuracy = 0.0  # 用于累积准确率
            val_true_positive_ratio, val_false_positive_ratio = 0.0 , 0.0
            

            with torch.no_grad():
                iou_scores = []
                
                val_loader_tqdm = tqdm(val_loader, total=len(val_loader), desc=f'Epoch {epoch+1}/{epochs} [val]', unit='batch')
                for images, masks, aux_inputs in val_loader_tqdm:
                    images, aux_inputs, masks = images.to(device), aux_inputs.to(device), masks.to(device)
                    # aux_input_new = processMasks_signal(masks)
                    input = torch.cat([images, aux_inputs], dim=1)
                    
                    outputs = model(input)
                    
                    # first_seg_val = model(images, aux_inputs, superpixels)
                    
                    # loss = dice_loss(outputs, masks) + (1-signal_protofg_similar.mean()) + signal_protobg_similar.mean()
                    #loss = criterion(outputs, masks)

                    
                    loss = dice_loss(outputs , masks)
                    # # loss = calc_dc_loss_sp(masks, all_pixel_features, superpixels , fg_proto_features)
                    # val_loss += loss.item() * images.size(0)
                    
                    
                    outputs = (outputs >= 0.5).int()
                    
                    dice_score += dice_coeff(outputs, masks).item() * images.size(0)
                    
                    # 计算准确率
                    _, batch_accuracy = calculate_binary_segmentation_accuracy(outputs, masks)
                    val_accuracy += batch_accuracy * images.size(0)  # 将每个批次的准确率加权累积
                    
                    
                    # true_positive_ratio_1 = calculate_spp(outputs, masks , suppixels)
                    # val_true_positive_ratio += true_positive_ratio_1 * images.size(0)
                    
                    # true_positive_ratio_1, false_positive_ratio_1 = calculate_metrics(outputs, masks)
                    # val_true_positive_ratio += true_positive_ratio_1* images.size(0)
                    # val_false_positive_ratio += false_positive_ratio_1* images.size(0)
                    
                    outputs = outputs.squeeze(1).cpu().numpy()
                    preds = (outputs >= 0.5).astype(int)
                    
                    masks = masks.cpu().numpy()
                    for pred, mask in zip(preds, masks):
                        miou = compute_miou_binary(pred, mask)
                        if not np.isnan(miou):
                            iou_scores.append(miou)
                mean_iou = np.mean(iou_scores)
                    

            val_loss /= len(val_loader.dataset)
            dice_score /= len(val_loader.dataset)
            val_accuracy /= len(val_loader.dataset)  # 计算平均准确率
            val_true_positive_ratio /= len(val_loader.dataset)
            
            # val_false_positive_ratio /= len(val_loader.dataset)

            print(f'Epoch {epoch+1}/{epochs}, Train Loss (CombinedLoss): {train_loss:.4f}, Train Dice: {train_dice_score:.4f}, Train Acc: {train_accuracy:.4f}, Val Loss : {val_loss:.4f}, Val Dice: {dice_score:.4f}, Val_Mean IOU: {mean_iou:.4f}, Val_Acc: {val_accuracy:.4f}')
            # print(f'Epoch {epoch+1}/{epochs}, Train_true_positive_ratio: {train_true_positive_ratio:.4f}, Train_false_positive_ratio:{train_false_positive_ratio:.4f},  Val_true_positive_ratio: {val_true_positive_ratio:.4f}, Val_false_positive_ratio:{val_false_positive_ratio:.4f}')
            # print(f'Epoch {epoch+1}/{epochs}, Train_true_positive_ratio: {train_true_positive_ratio:.4f}, Val_true_positive_ratio: {val_true_positive_ratio:.4f}')
            # scheduler.step()
            
            # # Save the best model
            if dice_score > best_dice:
                best_dice = dice_score
                print("best dice result: ",best_dice)
                # ИЗМЕНЕНО: сохранение в локальную директорию
                checkpoint_dir = f'{path}/checkpoints/fold_{i}'
                os.makedirs(checkpoint_dir, exist_ok=True)
                torch.save(model.state_dict(), f'{checkpoint_dir}/best_model_dice_{best_dice:.4f}_epoch_{epoch+1}.pth')
                print(f'✓ Модель сохранена: {checkpoint_dir}/best_model_dice_{best_dice:.4f}_epoch_{epoch+1}.pth')

    print(f"fold_{i}_{cls}.")
    # ИЗМЕНЕНО: epochs=3 для быстрого теста (вместо 100)
    train_model(model, train_loader, val_loader, loss_fn, optimizer, epochs=3)
    # print("模型：", model.__class__.__name__)




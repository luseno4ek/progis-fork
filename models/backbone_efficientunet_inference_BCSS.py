#分割模型和相似度模型计算分成俩模型
#不对特征上采样，最后对预测的mask进行上采样
from efficientunet import *
import torch.optim.lr_scheduler as lr_scheduler
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BatchNorm2d
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F1
from torchvision import models
#from resnet import resnet50

#from config import config
# from models.losses import getLoss, dice_coef

import torch.optim as optim
from sklearn.model_selection import train_test_split

from scipy.ndimage import distance_transform_edt
import scipy.ndimage as ndi
from skimage.morphology import skeletonize
from scipy.ndimage import label

from skimage.measure import label as label_1
from skimage.measure import regionprops

import cv2

from skimage import io, img_as_float
from skimage.segmentation import slic, mark_boundaries

import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from skimage.draw import disk
from skimage.color import rgb2lab
import pickle
from PIL import Image



from thop import profile  ## 测量模型 Flops  params



multiGPU = False
learningRate = 4e-4
img_chls = 3
weight_decay = 5e-5
RandomizeGuidingSignalType='Skeleton'

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
        skel = skeletonize(newMask.cpu().numpy())
        skel = torch.tensor(skel, dtype=torch.float32, device=binaryMask.device)
    else:
        skel = torch.zeros_like(binaryMask, dtype=torch.float32, device=binaryMask.device)

    return skel

def processMasks(pred_mask_all, GT_mask_all):
    """
    批量处理GT_mask和pred_mask，计算每个样本的前景和背景骨架信号。
    参数:
        pred_masks: 预测的mask，形状为 [batch_size, 1, H, W]。
        GT_masks: 真值mask，形状为 [batch_size, 1, H, W]。
    返回:
        输出张量，形状为 [batch_size, 2, H, W]。
        每个样本的第0通道为前景区域骨架信号，第1通道为背景区域骨架信号。
    """
    pred_mask_all = (pred_mask_all > 0.5).float()

    batch_size, _, H, W = pred_mask_all.shape

    # 初始化输出张量
    output = torch.zeros(batch_size, 2, H, W, device=pred_mask_all.device, dtype=torch.float32)
    centers = []  # 存储每个样本的最大错误连通域中心坐标

    for i in range(batch_size):
        # 取出当前样本的预测和真值mask
        pred_mask = pred_mask_all[i].squeeze(0)  # [H, W]
        GT_mask = GT_mask_all[i].squeeze(0)      # [H, W]

        # 计算前景区域 (GT_mask为1且pred_mask为0)
        fg = (GT_mask == 1) & (pred_mask == 0)
        fg = fg.to(torch.float32)  # [H, W]

        # 计算背景区域 (GT_mask为0且pred_mask为1)
        bg = (GT_mask == 0) & (pred_mask == 1)
        bg = bg.to(torch.float32)  # [H, W]
        
        # 找出前景的最大连通域
        if fg.sum() > 0:
            labeled_fg = label_1(fg.cpu().numpy(), connectivity=1)
            regions_fg = regionprops(labeled_fg)
            if regions_fg:
                largest_region_fg = max(regions_fg, key=lambda r: r.area)
                fg_largest = (labeled_fg == largest_region_fg.label)
                fg_largest = torch.from_numpy(fg_largest).to(fg.device, dtype=torch.float32)
                fg_center = largest_region_fg.centroid
                fg_center = (round(fg_center[0]), round(fg_center[1]))  # 四舍五入
            else:
                fg_largest = torch.zeros_like(fg)
                fg_center = None
        else:
            fg_largest = torch.zeros_like(fg)
            fg_center = None

        # 找出背景的最大连通域
        if bg.sum() > 0:
            labeled_bg = label_1(bg.cpu().numpy(), connectivity=1)
            regions_bg = regionprops(labeled_bg)
            if regions_bg:
                largest_region_bg = max(regions_bg, key=lambda r: r.area)
                bg_largest = (labeled_bg == largest_region_bg.label)
                bg_largest = torch.from_numpy(bg_largest).to(bg.device, dtype=torch.float32)
                bg_center = largest_region_bg.centroid  # 获取背景最大连通域的中心坐标 (y, x)
                bg_center = (round(bg_center[0]), round(bg_center[1]))  # 四舍五入
            else:
                bg_largest = torch.zeros_like(bg)
                bg_center = None
        else:
            bg_largest = torch.zeros_like(bg)
            bg_center = None
        
        # 比较前景和背景的最大连通域面积
        fg_area = fg_largest.sum().item()
        bg_area = bg_largest.sum().item()

        if fg_area >= bg_area:
            largest_connected = fg_largest
            # 计算前景区域的骨架信号
            fg_skeleton = generateGuidingSignal(largest_connected) if largest_connected.sum() > 0 else torch.zeros_like(pred_mask, dtype=torch.float32, device=pred_mask.device)  # 如果fg为全0，创建一个全零张量
            output[i, 0] = fg_skeleton  # 前景骨架信号
            centers.append(fg_center)  # 保存中心坐标
        else:
            largest_connected = bg_largest
            # 计算背景区域的骨架信号
            bg_skeleton = generateGuidingSignal(largest_connected) if largest_connected.sum() > 0 else torch.zeros_like(pred_mask, dtype=torch.float32, device=pred_mask.device)  # 如果fg为全0，创建一个全零张量
            output[i, 1] = bg_skeleton  # 背景骨架信号
            centers.append(bg_center)  # 保存中心坐标
            
        
        # # 计算前景区域的骨架信号
        # fg_skeleton = generateGuidingSignal(fg) if fg.sum() > 0 else torch.zeros_like(pred_mask, dtype=torch.float32, device=pred_mask.device)  # 如果fg为全0，创建一个全零张量
        
        # # 计算背景区域的骨架信号
        # bg_skeleton = generateGuidingSignal(bg) if bg.sum() > 0 else torch.zeros_like(pred_mask, dtype=torch.float32, device=pred_mask.device)  # 如果bg为全0，创建一个全零张量

        # # 合并前景和背景骨架信号到输出
        # output[i, 0] = fg_skeleton  # 前景骨架信号
        # output[i, 1] = bg_skeleton  # 背景骨架信号
         
    return output, centers


# ── GPU-accelerated processMasks with centroid output (replaces CPU version in inference) ──

def _edt_skeleton_gpu(mask, max_steps=80):
    """EDT-ridge skeleton, fully on GPU, zero CPU-GPU syncs inside loop."""
    B = mask.shape[0]
    device = mask.device
    edt = torch.zeros_like(mask)
    current = mask.clone()
    for step in range(1, max_steps + 1):
        next_c = -F.max_pool2d(-current, kernel_size=3, stride=1, padding=1)
        newly_removed = (current > 0) & (next_c == 0)
        edt = edt + newly_removed.float() * step
        current = next_c
    edt = edt + (current > 0).float() * (max_steps + 1)
    mask_bool = mask > 0
    edt_in_mask = edt * mask
    n_pixels = mask.flatten(1).sum(1).clamp(min=1)
    edt_sum = edt_in_mask.flatten(1).sum(1)
    edt_sq  = edt_in_mask.pow(2).flatten(1).sum(1)
    means = edt_sum / n_pixels
    stds  = (edt_sq / n_pixels - means.pow(2)).clamp(min=0).sqrt()
    rand   = torch.rand(B, 1, 1, 1, device=device)
    thresh = (means.view(B,1,1,1) - stds.view(B,1,1,1)
              + rand * 2 * stds.view(B,1,1,1)).clamp(min=0)
    core = (edt > thresh) & mask_bool
    need_fallback = (core.float().flatten(1).sum(1) == 0) & (mask.flatten(1).sum(1) > 0)
    max_edt = (edt * mask).flatten(1).max(1).values.view(B, 1, 1, 1)
    fallback_core = (edt >= max_edt - 1e-6) & mask_bool
    core = torch.where(need_fallback.view(B, 1, 1, 1), fallback_core, core)
    core_edt  = edt * core.float()
    local_max = F.max_pool2d(core_edt, kernel_size=3, stride=1, padding=1)
    ridge = (core_edt >= local_max - 1e-6) & core
    empty_ridge = (ridge.float().flatten(1).sum(1) == 0) & (core.float().flatten(1).sum(1) > 0)
    ridge = torch.where(empty_ridge.view(B, 1, 1, 1), core, ridge)
    return ridge.float()


def processMasks_gpu(pred_mask_all, GT_mask_all, max_edt_steps=80):
    """
    GPU drop-in for processMasks. Returns (signal [B,2,H,W], centers list).
    centers[b] = (cy, cx) int tuple, or None if no error region for sample b.
    Replaces CPU processMasks in the inference loop — eliminates scipy/skimage bottleneck.
    """
    device = pred_mask_all.device
    B, _, H, W = pred_mask_all.shape

    pred_bin = (pred_mask_all > 0.5).float()
    fg = ((GT_mask_all == 1) & (pred_bin == 0)).float()
    bg = ((GT_mask_all == 0) & (pred_bin == 1)).float()

    fg_area = fg.flatten(1).sum(1)
    bg_area = bg.flatten(1).sum(1)
    use_fg  = (fg_area >= bg_area).float().view(B, 1, 1, 1)

    selected = fg * use_fg + bg * (1 - use_fg)          # [B, 1, H, W]
    skel     = _edt_skeleton_gpu(selected, max_edt_steps)

    output = torch.zeros(B, 2, H, W, device=device)
    output[:, 0:1] = skel * use_fg
    output[:, 1:2] = skel * (1 - use_fg)

    # Compute centroids of selected error regions (vectorised, 3 tiny syncs total)
    ys = torch.arange(H, device=device).float().view(1, 1, H, 1)
    xs = torch.arange(W, device=device).float().view(1, 1, 1, W)
    n  = selected.flatten(1).sum(1).clamp(min=1)
    cy = (selected * ys).flatten(1).sum(1) / n
    cx = (selected * xs).flatten(1).sum(1) / n
    has_region = selected.flatten(1).sum(1) > 0
    cy_list  = cy.round().long().tolist()
    cx_list  = cx.round().long().tolist()
    has_list = has_region.tolist()
    centers = [(cy_list[b], cx_list[b]) if has_list[b] else None for b in range(B)]

    return output, centers


##########################################################################################################



### 相似度计算
    
class EfficientUNet_proto(nn.Module):
    def __init__(self, freeze=False, pretrained=True):
        super().__init__()
        self.freeze = freeze
        self.EfficientUNet_backbone = get_efficientunet_b0(out_channels=1, concat_input=True, pretrained=True, backbone=True)
        self.segment_part = get_efficientunet_b0(out_channels=1, concat_input=True, pretrained=False, backbone=False)
        

        
    def forward(self, roi_input , roi_aux_input , roi_suppixel, input , aux_input, superpixels , mask_box, threod):
        # 生成与 masks 形状相同的全零张量
        pred_mask = torch.zeros_like(roi_suppixel)
        roiseg_input = torch.cat((roi_input, pred_mask, roi_aux_input), dim=1)
        
        seg_orginal = self.segment_part(roiseg_input)
        
        x = self.EfficientUNet_backbone(input)
        
        '''
        #########################################################################################
        superpixel = []
        for b in range(x.shape[0]):
            image = input[b].cpu().numpy().transpose(1, 2, 0)  # 转换为 [H, W, 3]
            image = image.astype(np.uint8)
            image = img_as_float(image)  # 确保图像是浮动类型
            
            
            # 假设你要基于第一个图像的特征进行SLIC分割
            feature_map = x[b].cpu().numpy()  # 提取特征图并转换为NumPy数组，形状[32, H, W]
            # 将特征图调整为 [H, W, 32]，即每个像素的 32 个特征
            feature_map = feature_map.transpose(1, 2, 0)  # 转换为 [H, W, channels]
            # 假设 feature_map 形状是 [H, W, 32]，即图像的高、宽和特征通道数
            feature_map_min = np.min(feature_map, axis=(0, 1), keepdims=True)  # 在 H 和 W 维度上找最小值
            feature_map_max = np.max(feature_map, axis=(0, 1), keepdims=True)  # 在 H 和 W 维度上找最大值

            # 对每个通道进行归一化，使其值在 [0, 1] 范围内
            normalized_feature_map = (feature_map - feature_map_min) / (feature_map_max - feature_map_min)

            # 对特征图进行SLIC分割
            segments = slic(normalized_feature_map, n_segments=600, compactness=0.8, sigma=1, multichannel=True, convert2lab=False, enforce_connectivity=True, start_label=0, slic_zero=False, mask=None, channel_axis=-1)
            # Plot and save the segmented image with boundaries
            # fig = plt.figure("Superpixels -- %d segments" % (100))
            # ax = fig.add_subplot(1, 1, 1)
            # ax.imshow(mark_boundaries(image, segments))
            # plt.axis("off")

            # # Save the output image
            # output_path = os.path.join("/home/gjs/ISF_nuclick/keshihua_125wsi/val_slic_feat", filenames[b])
            # output_path = output_path.replace(".npy", ".png")
            # plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
            # plt.close(fig)  # Close the figure to avoid memory issues
            
            # 将生成的超像素标签从 [H, W] 转换为张量，并调整形状为 [1, H, W]
            sp_tensor = torch.tensor(segments, dtype=torch.long).unsqueeze(0)  # 形状变为 [1, H, W]

            # 将其添加到超像素标签列表中
            superpixel.append(sp_tensor)
        # 将所有超像素标签合并成一个张量，形状为 [batch_size, 1, H, W]
        superpixel = torch.cat(superpixel, dim=0)
        superpixel = superpixel.unsqueeze(1)
        # 确保张量在正确的设备上（CPU 或 GPU）
        superpixel = superpixel.to(input.device)       

        #############################################################################################
        '''
        
        
        # 对sigmoid_output进行阈值处理
        # bg_sigmoid_output = seg_orginal.clone()  # 复制一份，防止原始数据被修改
        fg_sigmoid_output = seg_orginal.clone()  # 复制一份，防止原始数据被修改
        
        # 将大于 0.2 的元素置零
        # 创建布尔掩码
        mask_greater_than_0_6 = fg_sigmoid_output > 0.95
        mask_less_equal_0_6 = fg_sigmoid_output <= 0.95

        # 使用掩码设置值
        fg_sigmoid_output[mask_greater_than_0_6] = 1
        fg_sigmoid_output[mask_less_equal_0_6] = 0
        
        #########################################################################################################
        # 设置固定裁剪尺寸
        crop_size = 256

        # 去掉 mask_with_box 的 channel 维度
        mask_box = mask_box.squeeze(1)  # [batch_size, H, W]

        # 使用广播机制将 mask_with_box 应用到 x 上
        x_cropped = x * mask_box.unsqueeze(1)  # [batch_size, 32, H, W]

        # 初始化存储裁剪后的特征图的列表
        cropped_regions = []

        for i in range(x.shape[0]):
            # 获取当前批次的 mask 和特征图
            mask = mask_box[i]  # [H, W]
            feature_map = x_cropped[i]  # [32, H, W]
            
            # 找到 mask 为 1 的位置
            nonzero_coords = torch.nonzero(mask, as_tuple=True)  # (y_indices, x_indices)
            
            if len(nonzero_coords[0]) > 0:  # 如果有非零区域
                # 计算非零区域的中心坐标
                center_y = (nonzero_coords[0].min() + nonzero_coords[0].max()) // 2
                center_x = (nonzero_coords[1].min() + nonzero_coords[1].max()) // 2

                # 确定裁剪区域的边界
                start_y = max(0, center_y - crop_size // 2)
                start_x = max(0, center_x - crop_size // 2)
                end_y = start_y + crop_size
                end_x = start_x + crop_size

                # 确保裁剪区域不超过图片边界
                start_y = min(start_y, feature_map.shape[1] - crop_size)
                start_x = min(start_x, feature_map.shape[2] - crop_size)
                end_y = start_y + crop_size
                end_x = start_x + crop_size

                # 裁剪特征图
                cropped = feature_map[:, start_y:end_y, start_x:end_x]  # [32, crop_size, crop_size]
            else:
                # 如果 mask 全为 0，则返回全零特征
                cropped = torch.zeros((x.shape[1], crop_size, crop_size), dtype=x.dtype, device=x.device)
            
            cropped_regions.append(cropped)

        # 堆叠裁剪后的特征图
        output_tensor = torch.stack(cropped_regions, dim=0)  # [batch_size, 32, crop_size, crop_size]

        # 使用掩膜提取前景特征
        foreground_features = output_tensor * fg_sigmoid_output  # [batch_size, 32, H, W]

        # 计算每个通道的前景特征平均值
        # 为避免除以零的情况，计算前景像素的数量
        foreground_pixel_count = fg_sigmoid_output.sum(dim=(2, 3), keepdim=True)  # [batch_size, 32, 1, 1]
        foreground_pixel_count = torch.clamp(foreground_pixel_count, min=1)  # 防止除以0

        # 前景平均特征池化
        fg_avg_features = foreground_features.sum(dim=(2, 3), keepdim=True) / foreground_pixel_count  # [batch_size, 32, 1, 1]
        fg_avg_features = fg_avg_features.squeeze(3)
        fg_avg_features = fg_avg_features.squeeze(2)
        
        
        # 对整个 batch 的 x 进行归一化
        x_normalized = F.normalize(x, dim=1)  # 对 C 维度归一化, 结果形状: [batch_size, C, H, W]
        # 对整个 batch 的 prototype（前景平均特征）进行归一化
        prototype_normalized = F.normalize(fg_avg_features, dim=1)  # 对 C 维度归一化, 结果形状: [batch_size, C]
        # 使用广播机制计算相似度（余弦相似度）
        # torch.einsum('bchw,bc->bhw', x_normalized, prototype_normalized)
        out_mask = (torch.einsum('bchw,bc->bhw', x_normalized, prototype_normalized))**2
        
        
        
        '''
        ############################################################################################################
        
        #################################### 通过avgpooling,获得所有超像素块的特征 #####################################
        
        # 初始化一个张量来存储每个超像素块的特征表示
        batch_size, channels, H, W = x.shape
        all_suppixel_labels = torch.unique(superpixels) 
        num_all_labels = torch.unique(superpixels).numel()
        # 获取 roi 区域中存在的超像素标签值，并将其转换为列表
        roi_suppixel_labels = torch.unique(roi_suppixel)

        all_superpixel_features = torch.zeros((batch_size, channels, num_all_labels), device=x.device)
        all_superpixel_counts = torch.zeros((batch_size, num_all_labels), device=x.device)
        
        for b in range(batch_size):
            for sp in all_suppixel_labels:
                sp = int(sp.item())  # 将张量转换为整数，便于后续处理
                mask = (superpixels[b] == sp).float()  # 将mask转换为float类型，用于乘法运算
                
                # 对应的x中的特征加到superpixel_features中
                selected_features = x[b] * mask  # x[b] 形状为 (channels, 512, 512)
                all_superpixel_features[b, :, sp] += selected_features.sum(dim=(1, 2))  # 对 (H, W) 维度求和
                all_superpixel_counts[b, sp] += mask.sum()  # 统计当前超像素块中的像素个数


        # 对每个超像素块进行平均池化
        all_superpixel_features /= (all_superpixel_counts.unsqueeze(1) + 1e-6)  # 防止除零错误
        
        ################################################################################################################
        
        ###################################################### spp 计算相似度 ###########################################       
        # 初始化一个列表来存储所有batch的similarity_mask
        all_similarity_masks = []
        # fg_proto_spp_features = []
        
        epsilon = 1e-6
        # 对 all_superpixel_features 的每个特征维度进行归一化
        all_superpixel_features_normalized = all_superpixel_features / (
            torch.norm(all_superpixel_features, dim=1, keepdim=True) + epsilon
        )
        # 对 fg_avg_features 的每个特征维度进行归一化
        fg_avg_features_normalized = fg_avg_features / (
            torch.norm(fg_avg_features, dim=1, keepdim=True) + epsilon
        )
        
        for b in range(batch_size):
            
            ################ 计算原型相似度#############
            if torch.sum(fg_sigmoid_output) > 0:
                # print("superpixels number: ",foreground_superpixel_count)    
                # average_foreground_feature = foreground_superpixel_features_sum / foreground_superpixel_count  # 计算平均特征

                # cosine_similarities = torch.zeros(num_all_labels, device=x.device)

                # 计算当前 batch 的所有超像素块的相似度
                batch_cosine_similarities = F.cosine_similarity(
                    all_superpixel_features_normalized[b],
                    fg_avg_features_normalized[b].unsqueeze(1),
                    dim=0
                )  # shape: [num_all_labels, 1]
                 # 创建一个与输入图像相同尺寸的 mask
                similarity_mask = torch.zeros((H, W), device=x.device)

                # 将每个超像素块的相似度值赋给对应超像素块的每个像素
                for sp in all_suppixel_labels:
                    sp = int(sp.item())  # 将张量转换为整数，便于后续处理
                    similarity_mask[superpixels[b, 0] == sp] = batch_cosine_similarities[sp]
                
                    

                # 将生成的 similarity_mask 存储在列表中
                all_similarity_masks.append(similarity_mask.unsqueeze(0))  # 在第0维增加一个维度，方便之后的拼接
                # fg_proto_spp_features.append(average_foreground_feature.unsqueeze(0))

            else:
                print(f"第 {b} 个 batch 中ROI没有前景像素")
                average_foreground_feature = torch.zeros((channels), device=x.device) # 计算平均特征
                similarity_mask = torch.zeros((H, W), device=x.device)
                all_similarity_masks.append(similarity_mask.unsqueeze(0))
                # fg_proto_spp_features.append(average_foreground_feature.unsqueeze(0))    
                
              
        
        # 使用 torch.cat 将所有 batch 的 similarity_mask 拼接在一起
        out_mask = torch.cat(all_similarity_masks, dim=0)
        # fg_proto_feature = torch.cat(fg_proto_spp_features, dim=0)
        
        ###############################################################################################################
        '''
        
        out_mask = out_mask.unsqueeze(1)
        
        
        # 计算out_mask的最小值和最大值
        min_val = out_mask.min()
        max_val = out_mask.max()
        # 最大最小归一化： (out_mask - min) / (max - min)
        out_mask_normalized = (out_mask - min_val) / (max_val - min_val + 1e-6)  # 避免除以0
        
        # # 计算每张图片的最小值和最大值
        # min_val = out_mask.view(out_mask.shape[0], -1).min(dim=1)[0].view(out_mask.shape[0], 1, 1, 1)
        # max_val = out_mask.view(out_mask.shape[0], -1).max(dim=1)[0].view(out_mask.shape[0], 1, 1, 1)
        # # 最大最小归一化
        # out_mask_normalized = (out_mask - min_val) / (max_val - min_val + 1e-6)
        
        # out_mask_normalized = torch.sigmoid(out_mask)

        out_mask_thresh = out_mask_normalized.clone()  # 创建一个 out_mask 的副本
        out_mask_thresh[out_mask_thresh <= threod] = 0
        out_mask_thresh[out_mask_thresh > threod] = 1
        
        
        
        
        
        # roi_pre_masks = torch.zeros_like(fg_sigmoid_output).to(input.device)
        # for i in range(x.shape[0]):
        #     # 获取当前批次的 mask 和特征图
        #     mask = mask_box[i]  # [H, W]

        #     # 找到 mask 为 1 的位置
        #     nonzero_coords = torch.nonzero(mask, as_tuple=True)  # (y_indices, x_indices)
            
        #     if len(nonzero_coords[0]) > 0:  # 如果有非零区域
        #         # 计算非零区域的中心坐标
        #         center_y = (nonzero_coords[0].min() + nonzero_coords[0].max()) // 2
        #         center_x = (nonzero_coords[1].min() + nonzero_coords[1].max()) // 2

        #         # 确定裁剪区域的边界
        #         start_y = max(0, center_y - crop_size // 2)
        #         start_x = max(0, center_x - crop_size // 2)
        #         end_y = start_y + crop_size
        #         end_x = start_x + crop_size

        #         # 确保裁剪区域不超过图片边界
        #         start_y = min(start_y, feature_map.shape[1] - crop_size)
        #         start_x = min(start_x, feature_map.shape[2] - crop_size)
        #         end_y = start_y + crop_size
        #         end_x = start_x + crop_size

        #         # 裁剪特征图
        #         out_mask_normalized[i, :, start_y:end_y, start_x:end_x] = seg_orginal.clone()[i] # [32, crop_size, crop_size]
        #         roi_pre_masks[i] = out_mask_thresh[i, :, start_y:end_y, start_x:end_x].clone()
        #         out_mask_thresh[i, :, start_y:end_y, start_x:end_x] = fg_sigmoid_output[i]


        
        # 定义阈值列表，例如：[0.1, 0.3, 0.5, 0.7, 0.9]
        # thresholds = [0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]



        return x, out_mask_thresh, fg_sigmoid_output ,x ,out_mask_normalized , superpixels


        # return all_superpixel_features, out_mask_thresh, nuclick_out, fg_proto_feature

        # return  x , out_mask , seg_orginal , fg_proto_feature 
    
    
def ROI_crop(input, aux_input, superpixel, mask):
    # 假设 input 和 aux_input 的形状分别为 (batch_size, 3, H, W) 和 (batch_size, 2, H, W)
    batch_size, _, H, W = input.shape
    
    # 创建空列表以存储裁剪后的 ROI
    roi_inputs = []
    roi_aux_inputs = []
    roi_suppixels = []
    roi_masks = []

    # 遍历 batch 中的每个样本
    for b in range(batch_size):
        # 找到 aux_input 中像素值为 1 的所有位置
        indices = (aux_input[b, 0] == 1).nonzero(as_tuple=True)

        # 如果找到指导信号点
        if indices[0].numel() > 0:
            # 计算中心点（x 和 y 的平均值）
            center_y = indices[0].float().mean().round().long().item()
            center_x = indices[1].float().mean().round().long().item()

            # 计算裁剪的边界
            start_y = max(center_y - 128, 0)
            start_x = max(center_x - 128, 0)
            end_y = min(start_y + 256, H)
            end_x = min(start_x + 256, W)

            # 调整起始位置，以保持裁剪区域的大小为 256x256
            if end_y - start_y < 256:
                start_y = max(end_y - 256, 0)
            if end_x - start_x < 256:
                start_x = max(end_x - 256, 0)
        else:
            # print("signal 为空！")
            # 随机生成裁剪的起始点，确保不会超出边界
            start_y = torch.randint(0, max(H - 256, 1), (1,)).item()
            start_x = torch.randint(0, max(W - 256, 1), (1,)).item()
            end_y = start_y + 256
            end_x = start_x + 256
            
        # 裁剪 input 和 aux_input
        roi_input = input[b, :, start_y:end_y, start_x:end_x]
        roi_aux_input = aux_input[b, :, start_y:end_y, start_x:end_x]
        roi_suppixel = superpixel[b, :, start_y:end_y, start_x:end_x]
        roi_mask = mask[b, :, start_y:end_y, start_x:end_x]

        roi_inputs.append(roi_input)
        roi_aux_inputs.append(roi_aux_input)
        roi_suppixels.append(roi_suppixel)
        roi_masks.append(roi_mask)
        

    # 将裁剪后的列表转换为张量
    roi_inputs = torch.stack(roi_inputs) if roi_inputs else None
    roi_aux_inputs = torch.stack(roi_aux_inputs) if roi_aux_inputs else None
    roi_suppixels = torch.stack(roi_suppixels) if roi_suppixels else None
    roi_masks = torch.stack(roi_masks) if roi_masks else None
    
    
    return roi_inputs, roi_aux_inputs, roi_suppixels, roi_masks

##########################################################################################
import numpy as np
import torch
from scipy.ndimage import label, find_objects

def get_largest_connected_component(roi_mask):
    """
    选择并返回 roi_mask 中前景（值为 1）中最大的连通区域，其余区域置为 0。
    
    参数:
    roi_mask (torch.Tensor): 二值化掩码，形状为 [1, H, W]，前景像素值为 1，背景像素值为 0。

    返回:
    torch.Tensor: 只包含最大连通域的掩码，其他区域为 0，形状为 [1, H, W]。
    """
    # 将 PyTorch 张量转为 NumPy 数组以便使用 scipy
    roi_mask_np = roi_mask.squeeze(0).cpu().numpy()  # 去掉 batch 维度并移到 CPU

    # 1. 使用 scipy 的 label 函数寻找连通区域
    structure = np.ones((3, 3), dtype=int)  # 邻接结构，可以设置为8邻域或4邻域
    labeled_array, num_features = label(roi_mask_np, structure)  # labeled_array 存储标记结果，num_features 是连通域数量

    # 如果没有前景区域（即没有连通域），直接返回全零的掩码
    if num_features == 0:
        return torch.zeros_like(roi_mask)

    # 2. 找到每个连通域的面积（像素数）
    regions = find_objects(labeled_array)  # 每个连通区域的边界框

    region_sizes = []
    for region in regions:
        region_mask = labeled_array[region] == labeled_array[region][0, 0]  # 提取当前连通域
        region_sizes.append(np.sum(region_mask))  # 计算当前连通域的像素数量

    # 3. 找到最大连通域的索引
    max_region_index = np.argmax(region_sizes)

    # 4. 创建一个新的 roi_mask，其中只保留最大连通域的像素
    new_roi_mask_np = np.zeros_like(roi_mask_np)  # 创建一个全 0 的新掩码

    # 使用找到的最大连通域的标签，保留该区域
    new_roi_mask_np[labeled_array == max_region_index + 1] = 1  # 连通域标签从 1 开始，因此加 1

    # 5. 将结果转回 PyTorch 张量，并恢复原始形状 [1, H, W]
    new_roi_mask = torch.tensor(new_roi_mask_np, dtype=torch.float32).unsqueeze(0)  # 添加 batch 维度

    return new_roi_mask


def generateGuidingSignal_1(mask, RandomizeGuidingSignalType):
    mask = mask.squeeze(0)  # Remove the batch dimension if it's (1, H, W)
    
    if RandomizeGuidingSignalType == 'Skeleton':
        # Create binary mask
        binaryMask = (mask > (0.5 * mask.max())).to(torch.uint8)
        
        if binaryMask.sum() > 1:
            # Compute distance transform (move to CPU for NumPy operations)
            distance_map = distance_transform_edt(binaryMask.cpu().numpy())
            distance_map = torch.tensor(distance_map, dtype=torch.float32, device=mask.device)
            
            # Calculate mean and std (ensure they are on CPU before NumPy operations)
            tempMean = distance_map.mean().cpu().numpy()
            tempStd = distance_map.std().cpu().numpy()
            
            # Random threshold based on mean and std
            tempThresh = np.random.uniform(tempMean - tempStd, tempMean + tempStd)
            tempThresh = torch.tensor(tempThresh, device=mask.device)
            
            if tempThresh < 0:
                tempThresh = np.random.uniform(tempMean / 2, tempMean + tempStd / 2)
                tempThresh = torch.tensor(tempThresh, device=mask.device)
            
            # Apply threshold to get new mask
            newMask = distance_map > tempThresh
            if newMask.sum() == 0:
                newMask = distance_map > (tempThresh / 2)
            
            if newMask.sum() == 0:
                newMask = binaryMask

            # Skeletonize (use skimage and convert back to tensor)
            skel = skeletonize(newMask.cpu().numpy())
            skel = torch.tensor(skel, dtype=torch.float32, device=mask.device)
        else:
            skel = torch.zeros_like(binaryMask, dtype=torch.float32).unsqueeze(-1)

        return skel

def ROI_crop_signal_line(input, aux_input, superpixel, mask):
    # 假设 input 和 aux_input 的形状分别为 (batch_size, 3, H, W) 和 (batch_size, 2, H, W)
    batch_size, _, H, W = input.shape
    
    # 创建空列表以存储裁剪后的 ROI
    roi_inputs = []
    roi_aux_inputs = []
    roi_suppixels = []
    roi_masks = []
    mask_box = torch.zeros_like(mask)
    
    all_aux_inputs = aux_input.clone()
    
    centers = []



    # 遍历 batch 中的每个样本
    for b in range(batch_size):
        # 找到 aux_input 中像素值为 1 的所有位置
        indices = (aux_input[b, 0] == 1).nonzero(as_tuple=True)

        # 如果找到指导信号点
        if indices[0].numel() > 0:
            # 计算中心点（x 和 y 的平均值）
            center_y = indices[0].float().mean().round().long().item()
            center_x = indices[1].float().mean().round().long().item()

            # 计算裁剪的边界
            start_y = max(center_y - 128, 0)
            start_x = max(center_x - 128, 0)
            end_y = min(start_y + 256, H)
            end_x = min(start_x + 256, W)

            # 调整起始位置，以保持裁剪区域的大小为 256x256
            if end_y - start_y < 256:
                start_y = max(end_y - 256, 0)
            if end_x - start_x < 256:
                start_x = max(end_x - 256, 0)
        else:
            # print("signal 为空！")
            # 随机生成裁剪的起始点，确保不会超出边界
            start_y = torch.randint(0, max(H - 256, 1), (1,)).item()
            start_x = torch.randint(0, max(W - 256, 1), (1,)).item()
            end_y = start_y + 256
            end_x = start_x + 256
            
        # 裁剪 input 和 aux_input
        roi_input = input[b, :, start_y:end_y, start_x:end_x]
        # roi_aux_input = aux_input[b, :, start_y:end_y, start_x:end_x]
        roi_suppixel = superpixel[b, :, start_y:end_y, start_x:end_x]
        roi_mask = mask[b, : ,start_y:end_y, start_x:end_x]
        roi_mask_new = get_largest_connected_component(roi_mask)
    
        
        mask_box[b, :, start_y:end_y, start_x:end_x] = 1
        
        RandomizeGuidingSignalType = 'Skeleton'  # Set this according to your needs
        guidingSignal = generateGuidingSignal_1(roi_mask_new, RandomizeGuidingSignalType)
        guidingSignal = guidingSignal.unsqueeze(0)

        # Expanding the guiding signal tensor and ensuring the correct shape
        guidingSignal_expanded = torch.cat([guidingSignal, torch.zeros_like(guidingSignal)], dim=0)
        if guidingSignal_expanded.dim() == 4 :
            guidingSignal_expanded = guidingSignal_expanded.squeeze(-1)
        # guidingSignal_expanded = guidingSignal_expanded.permute(2, 0, 1)
        
        guidingSignal_expanded = guidingSignal_expanded.to(input.device)

        roi_inputs.append(roi_input)
        roi_aux_inputs.append(guidingSignal_expanded)
        roi_suppixels.append(roi_suppixel)
        roi_masks.append(roi_mask)
        
        centers.append([center_x, center_y])
        

    # 将裁剪后的列表转换为张量
    roi_inputs = torch.stack(roi_inputs) if roi_inputs else None
    roi_aux_inputs = torch.stack(roi_aux_inputs) if roi_aux_inputs else None
    roi_suppixels = torch.stack(roi_suppixels) if roi_suppixels else None
    roi_masks = torch.stack(roi_masks) if roi_masks else None
    
    roi_aux_inputs = roi_aux_inputs.to(input.device)
    all_aux_inputs = all_aux_inputs.to(input.device)
    
    return roi_inputs, roi_aux_inputs, roi_suppixels , roi_masks, mask_box , all_aux_inputs, centers





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
    pred = pred.cpu().numpy()
    target = target.cpu().numpy()
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


def calculate_metrics(output, target):
    """
    Calculates dice, accuracy, and IoU scores.

    Args:
        output (torch.Tensor): Predicted segmentation mask (logits or probabilities). Shape: [batch_size, 1, H, W]
        target (torch.Tensor): Ground truth segmentation mask. Shape: [batch_size, 1, H, W]

    Returns:
        tuple: A tuple containing the average dice, accuracy, and IoU scores across the batch.
    """
    batch_size = output.size(0)
    output = torch.sigmoid(output)  # Apply sigmoid if output is logits
    output = (output > 0.5).float()  # Convert to binary mask

    dice_scores = []
    iou_scores = []
    acc_scores = []

    for i in range(batch_size):
        pred = output[i, 0]  # Get the prediction for the current batch item
        true = target[i, 0]    # Get the target for the current batch item

        intersection = torch.sum(pred * true)
        union = torch.sum(pred) + torch.sum(true)
        dice = (2 * intersection) / (union + 1e-7) # Add smooth term to prevent division by zero
        dice_scores.append(dice.item())


        iou = intersection / (union - intersection + 1e-7)
        iou_scores.append(iou.item())


        acc = torch.sum(pred == true).float() / true.numel()
        acc_scores.append(acc.item())

    avg_dice = sum(dice_scores) / batch_size
    avg_iou = sum(iou_scores) / batch_size
    avg_acc = sum(acc_scores) / batch_size

    return avg_dice, avg_acc, avg_iou


def NoI(pre_mask_list, masks, filenames):
    batch_size, _, _, _ = masks.shape
    iou_batch_list = []
    dice_batch_list = []
    acc_batch_list = []
    # saved_filenames = set()  # 使用集合来高效地跟踪已保存的文件名
    # output_txt_path = "/home/gjs/ISF_nuclick/X_ISF/Visualization_bu/fold1_BCSS_saved_files.txt"
    # num_list = [1, 5, 10, 15, 20]

    for b in range(batch_size):
        iou_single_list = []
        dice_single_list = []
        acc_single_list = []
        for i, pre_mask in enumerate(pre_mask_list):
            iou_single_list.append(compute_miou_binary(pre_mask[b], masks[b]))
            acc_single_list.append(calculate_binary_segmentation_accuracy(pre_mask[b], masks[b]))
            dice_single_list.append(dice_coeff(pre_mask[b], masks[b]))

            ## 可视化
            # Visualization_iou = compute_miou_binary(pre_mask[b], masks[b])
            
            # signal_file_str = filenames[b]  # 从元组中提取出字符串 提前提取，避免重复计算
            # filename = signal_file_str.split('/')[-1].replace(".npy", "")
            
            # if Visualization_iou > 0.85 and filename not in saved_filenames:
            # if ((i+1) in num_list):
                
            #     IoU = compute_miou_binary(pre_mask[b], masks[b])
                
            #     saved_filenames.add(filename)  # 将文件名添加到已保存的集合中

            #     # 将filename和i+1写入txt文件
            #     try:
            #         with open(output_txt_path, "a") as f:  # 使用 "a" 模式追加写入
            #             f.write(f"{i+1}_{filename} : {IoU}\n")
            #     except FileNotFoundError:  # 如果文件不存在，则创建它
            #         with open(output_txt_path, "w") as f:
            #             f.write(f"{i+1}_{filename} : {IoU}\n")

                # out_put_1 = pre_mask[b].squeeze(0)

                # 使用PIL保存图片，避免matplotlib的资源问题
                # image_array = out_put_1.detach().cpu().numpy()
                # image = Image.fromarray((image_array * 255).astype('uint8')) # 转换为uint8，确保正确显示
                # image.save(f'/home/gjs/ISF_nuclick/X_ISF/Visualization_bu/BCSS_fold1_ProGIS/{i+1}_{filename}.png')
                    
        iou_batch_list.append(iou_single_list)
        acc_batch_list.append(acc_single_list)
        dice_batch_list.append(dice_single_list)
                
    return iou_batch_list, acc_batch_list, dice_batch_list
        
    




###################################################

# Define the dataset class
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, images_dir, signal_dir, masks_dir, suppixel_dir,  filenames):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.signal_dir = signal_dir
        self.suppixel_dir = suppixel_dir
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
        suppixel_path = os.path.join(self.suppixel_dir, filename)

        image = np.load(image_path)
        mask = np.load(mask_path)
        signal = np.load(signal_path)
        suppixel = np.load(suppixel_path)

        # 转换为 PyTorch 张量
        image = torch.tensor(image.transpose(2, 0, 1), dtype=torch.float32)  # (channels, height, width)
        mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)  # (1, height, width)
        
        signal = torch.tensor(signal.transpose(2, 0, 1), dtype=torch.float32)  # (channels, height, width)
        # suppixel = torch.tensor(suppixel, dtype=torch.float32).unsqueeze(0)
        suppixel = torch.tensor(suppixel.transpose(2, 0, 1), dtype=torch.float32)  # (1, height, width)

        return image, signal, mask, suppixel, filename
    
# 获取文件夹中的文件名
def get_filenames_from_folder(folder_path):
    return [filename for filename in os.listdir(folder_path) if filename.endswith('.npy')]   


# Training function
def train_model(model, val_loader, epochs=50, threod=0.4, fold=1 , cls_num="1"):
    best_dice = 0.0
    
    model.to(device)

    for epoch in range(epochs):
   
        ###
        model.eval()
        
        number = 21
        
        iou_NOI_list = []
        acc_NOI_list = []
        dice_NOI_list = []
        
        val_dice_score = [0.0 for _ in range(number)]
        nuclick_val_dice = [0.0 for _ in range(number)]
        val_accuracy = [0.0 for _ in range(number)]  
        nuclick_val_acc = [0.0 for _ in range(number)]
        mean_IoU = [0.0 for _ in range(number)]
        mean_IoU_list = [[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]
        

        with torch.no_grad():

            for images, aux_inputs, masks, superpixels, filenames in tqdm(val_loader, desc=f'threod={threod:.2f}', leave=True):
                images, aux_inputs, masks, superpixels= images.to(device), aux_inputs.to(device), masks.to(device), superpixels.to(device)
                # outputs, all_superpixel_features = model(images, aux_inputs, superpixels)
                roi_input , roi_aux_input , roi_suppixel , roi_mask , mask_box, all_aux_inputs, centers_1= ROI_crop_signal_line(images , aux_inputs, superpixels, masks)
            

                all_perpixel_features , pre_masks ,nuclick_out , all_pixel_features, out_mask_normalized , superpixels = model(roi_input , roi_aux_input , roi_suppixel, images, aux_inputs, superpixels, mask_box, threod)
                
                #############################################  迭代  ############################################
                # pre_masks = torch.zeros_like(masks).to(device)
                
                
                out_put = pre_masks.clone()
                pre_mask_list = []
                pre_mask_list.append(out_put)
                
                count = 0
            
                    
                signal, centers = processMasks_gpu(out_put, masks)
                union_signal = torch.bitwise_or(signal.to(torch.uint8), aux_inputs.to(torch.uint8))
                
                # union_signal = all_aux_inputs
                # out_mask_normalized = pre_masks.clone()
                
                while count < 19:
                    if count > 0:
                        out_put = pre_masks
                        
                        signal, centers = processMasks_gpu(out_put, masks)
                        union_signal = torch.bitwise_or(signal.to(torch.uint8), union_signal.to(torch.uint8))
                    
                    # 假设 outputs 和 masks 的形状都是 (batch_size, 1, H, W)
                    batch_size, _, H, W = out_put.shape
 
                    # val_accuracy += batch_accuracy * images.size(0)  # 将每个批次的准确率加权累积
                    # print("矫正前的准确率：",batch_accuracy)
                    # print("文件名：", filenames)
                        
                    roi_inputs = []
                    roi_aux_inputs = []
                    roi_pred_masks = []

                    for b in range(batch_size):
                        # 计算裁剪的边界，中心坐标为正方形框的中心
                        # centers[b] is None when prediction is perfect (no error region) -> use image center
                        center_b = centers[b] if centers[b] is not None else (H // 2, W // 2)
                        start_y = max(center_b[0] - 128, 0)
                        start_x = max(center_b[1] - 128, 0)
                        end_y = min(start_y + 256, H)
                        end_x = min(start_x + 256, W)

                        # 调整起始位置，以保持裁剪区域的大小为 100x100
                        if end_y - start_y < 256:
                            start_y = int(max(end_y - 256, 0))
                        if end_x - start_x < 256:
                            start_x = int(max(end_x - 256, 0))
                            
                        mask_box = torch.zeros_like(masks)
            
                        # 裁剪 input 和 aux_input
                        roi_input = images[b, :, start_y:end_y, start_x:end_x]
                        roi_aux_input = union_signal[b, :, start_y:end_y, start_x:end_x]
                        # pred_mask = out_mask_normalized[b, :, start_y:end_y, start_x:end_x]
                        pred_mask = out_put[b, :, start_y:end_y, start_x:end_x]


                        roi_inputs.append(roi_input)
                        roi_aux_inputs.append(roi_aux_input)
                        roi_pred_masks.append(pred_mask)
                    

                    # 将裁剪后的列表转换为张量
                    roi_inputs = torch.stack(roi_inputs) if roi_inputs else None
                    roi_aux_inputs = torch.stack(roi_aux_inputs) if roi_aux_inputs else None
                    roi_pred_masks = torch.stack(roi_pred_masks) if roi_pred_masks else None
                    
                    
                    
                    
                    # 生成与 masks 形状相同的全零张量

                    roiseg_input = torch.cat((roi_inputs, roi_pred_masks, roi_aux_inputs), dim=1)
                    
                    seg_orginal = model.segment_part(roiseg_input)

                    # 对sigmoid_output进行阈值处理
                    # bg_sigmoid_output = seg_orginal.clone()  # 复制一份，防止原始数据被修改
                    fg_sigmoid_output = seg_orginal.clone()  # 复制一份，防止原始数据被修改
                    
                    # 将大于 0.2 的元素置零
                    # 创建布尔掩码
                    mask_greater_than_0_6 = fg_sigmoid_output > 0.5
                    mask_less_equal_0_6 = fg_sigmoid_output <= 0.5

                    # 使用掩码设置值
                    fg_sigmoid_output[mask_greater_than_0_6] = 1
                    fg_sigmoid_output[mask_less_equal_0_6] = 0
                    
                    for b in range(batch_size):
                        # 计算裁剪的边界，中心坐标为正方形框的中心
                        # centers[b] is None when prediction is perfect (no error region) -> use image center
                        center_b = centers[b] if centers[b] is not None else (H // 2, W // 2)
                        start_y = max(center_b[0] - 128, 0)
                        start_x = max(center_b[1] - 128, 0)
                        end_y = min(start_y + 256, H)
                        end_x = min(start_x + 256, W)

                        # 调整起始位置，以保持裁剪区域的大小为 100x100
                        if end_y - start_y < 256:
                            start_y = int(max(end_y - 256, 0))
                        if end_x - start_x < 256:
                            start_x = int(max(end_x - 256, 0))
                            
                        # average_mask[b, :, start_y:end_y, start_x:end_x] = out_mask_normalized_2[b, :, start_y:end_y, start_x:end_x]
                        pre_masks[b, :, start_y:end_y, start_x:end_x] = fg_sigmoid_output[b]
                        out_mask_normalized[b, :, start_y:end_y, start_x:end_x] = seg_orginal.clone()[b]
                    # average_mask[average_mask > 0.6] = 1
                    # average_mask[average_mask <= 0.6] = 0
                    pre_masks[pre_masks > 0.5 ] = 1
                    pre_masks[pre_masks <= 0.5 ] = 0
                    pre_masks_clone = pre_masks.clone()
                    
                    pre_mask_list.append(pre_masks_clone)
 
                            
                    count +=1 

                    
                # print("矫正结束！")       
                iou_batch_list, acc_batch_list, dice_batch_list = NoI(pre_mask_list, masks, filenames)
                iou_NOI_list.extend(iou_batch_list)
                acc_NOI_list.extend(acc_batch_list)
                dice_NOI_list.extend(dice_batch_list)

                    
                for i in range(len(pre_mask_list)):
                    iou_scores = []
                    val_dice_score[i] += dice_coeff(pre_mask_list[i], masks).item() * images.size(0)
                    nuclick_val_dice[i] += dice_coeff(nuclick_out, roi_mask).item() * images.size(0)
                    
                    # 计算准确率
                    _, batch_accuracy = calculate_binary_segmentation_accuracy(pre_mask_list[i], masks)
                    val_accuracy[i] += batch_accuracy * images.size(0)  # 将每个批次的准确率加权累积
                    
                    _, nuclick_batch_accuracy = calculate_binary_segmentation_accuracy(nuclick_out, roi_mask)
                    nuclick_val_acc[i] += nuclick_batch_accuracy * images.size(0)  # 将每个批次的准确率加权累积
                    
                    for pred, mask in zip(pre_mask_list[i], masks):
                        miou = compute_miou_binary(pred, mask)
                        if not np.isnan(miou):
                            mean_IoU_list[i].append(miou)
                    
                    # mean_IoU_list.append(iou_scores)
            
            fold_dir = f'/data_nas2/gjs/ISF_pixel_level_data/BCSS_x10_reinhard_cut/125WSI/fold_{fold}/efficientUnet/results' 
            os.makedirs(fold_dir, exist_ok=True)    
            filename_iou = f"{fold_dir}/iou_{cls_num}_thr{threod}.pkl" #  指定文件名
            filename_acc = f"{fold_dir}/acc_{cls_num}_thr{threod}.pkl" #  指定文件名
            filename_dice = f"{fold_dir}/dice_{cls_num}_thr{threod}.pkl" #  指定文件名
            
            try:
                with open(filename_iou, 'wb') as f:
                    pickle.dump(iou_NOI_list, f)
                print(f"NOI list saved to {filename_iou}")
                
                with open(filename_acc, 'wb') as f:
                    pickle.dump(acc_NOI_list, f)
                print(f"NOI list saved to {filename_acc}")
                
                with open(filename_dice, 'wb') as f:
                    pickle.dump(dice_NOI_list, f)
                print(f"NOI list saved to {filename_dice}")
                
                
            except Exception as e:
                print(f"Error saving NOI list: {e}")
                    
            # 对每个列表中的每个元素除以 len(train_loader.dataset)
            dataset_len = len(val_loader.dataset)

            for i in range(len(val_dice_score)):
                val_dice_score[i] /= dataset_len
                nuclick_val_dice[i] /= dataset_len
                val_accuracy[i] /= dataset_len
                nuclick_val_acc[i] /= dataset_len
    
                mean_IoU[i] = np.mean(mean_IoU_list[i])
                
        
        
        for i in range(21):
            print(f'Thresholds:{i+1:.4f} , Val Dice: {val_dice_score[i]:.4f}, Val_Mean IOU: {mean_IoU[i]:.4f}, Val_Acc: {val_accuracy[i]:.4f}, nuclick_Val_Dice: {nuclick_val_dice[i]:.4f},  nuclick_Val_Acc: {nuclick_val_acc[i]:.4f}')
        
        # print(f"Execution time: {elapsed_time:.4f} seconds")
        
        return val_dice_score[19], mean_IoU[19], val_accuracy[19]

 
# thresholds =  [0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
# cls = '6'
# choices=['tumor', 'stroma', 'inflammatory_infiltration','necrosis', 'others']

# ── Config ────────────────────────────────────────────────────────────────────
FOLD        = 1
CLS         = 'all'
PATCHES_DIR = '../data/patches'
SPLITS_JSON = '../data/processed/fold_splits.json'
# Checkpoint: trained ROI-Seg model — update path to your best checkpoint
# e.g. ../data/patches/fold_1/ROI_ckpt/BCSS_effi-Unet_roi_best_dice0.XXXX_epochN.pth
ROI_CKPT    = '../data/patches/fold_1/ROI_ckpt/BCSS_effi-Unet_roi_best_dice0.4354_epoch1.pth'
# ─────────────────────────────────────────────────────────────────────────────

from dataset import RoISegDataset

class InferenceDataset(torch.utils.data.Dataset):
    """Wraps RoISegDataset to match inference loader format:
       (image, aux_input, mask, suppixel_dummy, filename)
       suppixel is zeros — used only as shape reference inside model.forward.
    """
    def __init__(self, base_dataset):
        self.base = base_dataset

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        image, mask, signal = self.base[idx]
        fname, _ = self.base.items[idx]
        suppixel = torch.zeros(1, image.shape[1], image.shape[2], dtype=torch.float32)
        return image, signal, mask, suppixel, fname


device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Model: ImageNet pretrained backbone (pretrained=True set in __init__)
#        + our trained ROI-Seg (segment_part)
model = EfficientUNet_proto()
model.segment_part.load_state_dict(torch.load(ROI_CKPT, map_location='cpu'))
print(f"Loaded ROI-Seg checkpoint: {ROI_CKPT}")

for param in model.parameters():
    param.requires_grad = False

if multiGPU:
    model = nn.DataParallel(model, device_ids=[0, 1])

# Cosine-similarity threshold for backbone prototype matching
# Test several values — optimal depends on ImageNet feature quality
threod_sim_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85]

val_base    = RoISegDataset(PATCHES_DIR, SPLITS_JSON, fold=FOLD, split='val', cls=CLS)
val_dataset = InferenceDataset(val_base)
val_loader  = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=8, pin_memory=True)

for threod_sim in threod_sim_list:
    dice, iou, acc = train_model(model, val_loader, epochs=1, threod=threod_sim,
                                 fold=FOLD, cls_num=CLS)
    print(f"threod={threod_sim:.2f} | Dice@20={dice:.4f}  mIoU@20={iou:.4f}  Acc={acc:.4f}")

torch.cuda.empty_cache()

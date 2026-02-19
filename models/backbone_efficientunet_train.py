#分割模型和相似度模型计算分成俩模型
#不对特征上采样，最后对预测的mask进行上采样
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
import torch.nn.functional as F1
from torchvision import models
#from resnet import resnet50

#from config import config
# from models.losses import getLoss, dice_coef

import torch.optim as optim
from sklearn.model_selection import train_test_split
from tqdm import tqdm


import scipy.ndimage as ndi


import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import torch.distributed as dist
import torch.multiprocessing as mp

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from efficientunet import *
from UNet import *

# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

# device_id =  0 # Specify which GPU to use, e.g., GPU 0
# torch.cuda.set_device(device_id)

# Set the main device to GPU 0
# device_ids = 1   #[0, 1]
# device = torch.device('cuda')

multiGPU = False
learningRate = 4e-4
img_chls = 3
weight_decay = 5e-5

# pre-training loss for the proposed superpixel-level contrastive learning
def calc_dc_loss_sp( masks, all_perpixel_features, superpixels):
    
    # all_superpixel_features.size = (b , c , h , w)
    # superpixels.size = (b , 1 , h , w)
    # masks.size = (b , 1 , h , w)
    # print(f"Loss is calculated on device: {all_perpixel_features.device}")
    if not all_perpixel_features.is_cuda:
        pass # Or handle specific CPU logic if necessary

    device = all_perpixel_features.device # This is the key: get the device from an input tenso

    b, c, h, w = all_perpixel_features.shape
    loss_dc = 0
    loss_px = 0
    temperature = 0.3

    def get_sps_features(all_perpixel_features, superpixels, sps_list):
        pos_list = []
        target_pos_list = []
        one_sps = superpixels[bi,0, ...] # w, h
        one_outputs = all_perpixel_features[bi, ...] # c, w, h
        one_sps_feature_list = []
        for spi in sps_list:
            sp_index = torch.where(one_sps==spi)
            sp_feature = one_outputs[:, sp_index[0], sp_index[1]]
            sp_feature = torch.mean(sp_feature, -1)
            one_sps_feature_list.append(sp_feature)
        sps_feature_list = torch.stack(one_sps_feature_list)
        return sps_feature_list

    def get_sp_contrast_loss(fg_sps_labels_list, bg_sps_labels_list, affinity_matrix_ff, affinity_matrix_fb):
        loss = 0
        loss_ff = 0
        loss_fb = 0

        for i in range(len(fg_sps_labels_list)):

            cos_smi_sp_ff =  affinity_matrix_ff[i, :]
            cos_all_ff = torch.sum(cos_smi_sp_ff)     
            cos_avg_ff = cos_all_ff/len(fg_sps_labels_list)
            # loss_ff += -torch.log(cos_avg_ff/30)
            
            cos_smi_sp_fb =  affinity_matrix_fb[i, :]
            cos_all_fb = torch.sum(cos_smi_sp_fb)     
            cos_avg_fb = cos_all_fb/len(bg_sps_labels_list)
            # loss_fb += -torch.log(cos_avg_fb/30)
            
            loss += -torch.log(cos_avg_ff/(cos_avg_ff + cos_avg_fb + 1e-8) )
            
            
        return loss/len(fg_sps_labels_list)
    
    def get_loss_per_pixels_with_fg_bg_sps(all_perpixel_features, superpixels, fg_sps_labels_list, bg_sps_labels_list, fg_sps_feature_list1_norm, bg_sps_feature_list1_norm, affinity_matrix ):
        fg_total_loss = 0
        bg_total_loss = 0
        total_loss = 0 
        
        one_sps = superpixels[bi, 0, ...]  # w, h
        one_image_feature = all_perpixel_features[bi, ...]  # c, w, h
        affinity_matrix_fg_sum = torch.sum(affinity_matrix, 1)
        affinity_matrix_bg_sum = torch.sum(affinity_matrix, 0)
        
        
            
        # 前景像素与前景超像素块特征相似性增加
        for i in range(len(fg_sps_labels_list)):
            sp_index_fg = torch.where(one_sps == fg_sps_labels_list[i])
            pixels_feature_fg = one_image_feature[:, sp_index_fg[0], sp_index_fg[1]].T
            # pixels_feature_fg_norm = pixels_feature_fg / pixels_feature_fg.norm(dim=1)[:, None]
            # 避免除以 0 的情况
            pixels_norm = pixels_feature_fg.norm(dim=1).clone()
            pixels_norm[pixels_norm == 0] = 1e-8
            pixels_feature_fg_norm = pixels_feature_fg / pixels_norm[:, None]
            
            # 计算像素与当前前景超像素块特征的相似性
            sp_feature_fg_norm = fg_sps_feature_list1_norm[i:i+1]  # 当前前景超像素块的特征
            affinity_matrix_pixels = torch.mm(sp_feature_fg_norm, pixels_feature_fg_norm.t())
            affinity_matrix_pixels = torch.exp(affinity_matrix_pixels / temperature)
            e_sim_fg = affinity_matrix_pixels.mean()
            e_dis_fg = affinity_matrix_fg_sum[i]
            # 计算总损失
            fg_total_loss += -torch.log(e_sim_fg /(e_sim_fg + e_dis_fg + 1e-8))
            
        # print("fg_total_loss:", fg_total_loss)
           
        # # 背景像素与背景超像素块特征相似性增加
        # for i in range(len(bg_sps_labels_list)):
        #     sp_index_bg = torch.where(one_sps == bg_sps_labels_list[i])
        #     pixels_feature_bg = one_image_feature[:, sp_index_bg[0], sp_index_bg[1]].T
        #     # pixels_feature_bg_norm = pixels_feature_bg / pixels_feature_bg.norm(dim=1)[:, None]
        #     # 避免除以 0 的情况
        #     pixels_norm = pixels_feature_bg.norm(dim=1).clone()
        #     pixels_norm[pixels_norm == 0] = 1e-8
        #     pixels_feature_bg_norm = pixels_feature_bg / pixels_norm[:, None]
            
        #     # 计算像素与当前前景超像素块特征的相似性
        #     sp_feature_bg_norm = bg_sps_feature_list1_norm[i:i+1]  # 当前前景超像素块的特征
        #     affinity_matrix_pixels = torch.mm(sp_feature_bg_norm, pixels_feature_bg_norm.t())
        #     affinity_matrix_pixels = torch.exp(affinity_matrix_pixels / temperature)
        #     e_sim_bg = affinity_matrix_pixels.mean()
        #     e_dis_bg = affinity_matrix_bg_sum[i]
        #     # 计算总损失
        #     bg_total_loss += -torch.log(e_sim_bg /(e_sim_bg + e_dis_bg + 1e-8))
         
        # print("bg_total_loss:", bg_total_loss) 
        total_loss = fg_total_loss / len(fg_sps_labels_list) #+ bg_total_loss / len(bg_sps_labels_list)

           
            
            
        return total_loss
    '''    
    # def Prototypical_loss(fg_sps_labels_list, bg_sps_labels_list , sps_feature_list):
    def Prototypical_loss(fg_sps_labels_list, bg_sps_labels_list, sps_feature_list):
        # Step 1: 生成标签列表，前景设为1，背景设为0
        labels = torch.zeros(sps_feature_list.shape[0], dtype=torch.long).cuda()
        labels[fg_sps_labels_list] = 1
        labels[bg_sps_labels_list] = 0
        
        fg_sps_labels_list = fg_sps_labels_list.tolist()
        
        # Step 2: 随机抽取1/3的前景超像素块
        fg_selected = random.sample(fg_sps_labels_list, len(fg_sps_labels_list) // 3)
        
        # Step 3: 平均池化获得前景原型特征
        fg_prototype = torch.mean(sps_feature_list[fg_selected], dim=0, keepdim=True)
        
        # Step 4: 计算原型特征与所有超像素块的相似度（余弦相似度）
        similarity = F.cosine_similarity(sps_feature_list, fg_prototype)
        
        # Step 5: 最大最小归一化，将相似度映射到 [0, 1]
        sim_min, sim_max = similarity.min(), similarity.max()
        normalized_similarity = (similarity - sim_min) / (sim_max - sim_min + 1e-8)  # 加一个小值防止除0

        # Step 6: 计算交叉熵损失
        loss = F.binary_cross_entropy(normalized_similarity, labels.float())
        
        return loss
    '''   

        
        
    for bi in range(b): # for each case
        
        num_batch = b

        target_sps = superpixels[bi,0,...] # w, h
        
        sps_labels_list = torch.unique(target_sps)


        sps_feature_list = get_sps_features(all_perpixel_features, superpixels, sps_labels_list).to(device) # size = [[],[]...],  所有超像素块的特征
        
        
        loss_img = 0
        
        
        # 将 masks 和 superpixels 展平
        mask_flat = masks[bi].view(-1)  # shape: (H*W,)
        superpixels_flat = superpixels[bi].view(-1)  # shape: (H*W,)
        '''
        fg_sps_labels_list = []
        bg_sps_labels_list = []

        for label in sps_labels_list:
            # 获取该超像素块中所有像素的掩码值
            mask_for_superpixel = mask_flat[superpixels_flat == label]

            # 计算前景和背景像素的数量
            num_fg_pixels = torch.sum(mask_for_superpixel)
            num_bg_pixels = mask_for_superpixel.numel() - num_fg_pixels

            if num_fg_pixels > num_bg_pixels:
                fg_sps_labels_list.append(label)
            else:
                bg_sps_labels_list.append(label)
        '''    
                
        # 将 masks 和 superpixels 展平
        mask_flat = masks[bi].view(-1)  # shape: (H*W,)
        superpixels_flat = superpixels[bi].view(-1)  # shape: (H*W,)

        # 获取除背景外的所有类别，假设背景类为0
        unique_labels = mask_flat.unique()
        fg_labels = unique_labels[unique_labels > 0]  # 去除背景
        
        fg_labels_num = len(fg_labels)

        # 遍历每个前景类别，依次将一个类别作为前景，其余类别作为背景
        for fg_label in fg_labels:
            fg_sps_labels_list = []
            bg_sps_labels_list = []

            # 当前类别作为前景，其他类别作为背景
            # mask_for_fg = (mask_flat == fg_label).float()  # 当前前景掩码
            # mask_for_bg = (mask_flat != fg_label).float()  # 其余类别作为背景

            # 遍历超像素块标签列表
            for label in sps_labels_list:
                # 获取该超像素块中所有像素的掩码值
                mask_for_superpixel = mask_flat[superpixels_flat == label]

                # 当前前景和背景像素的数量
                num_fg_pixels = torch.sum(mask_for_superpixel == fg_label)
                num_bg_pixels = torch.sum(mask_for_superpixel != fg_label)

                # 根据前景和背景像素数量进行分类
                if num_fg_pixels > num_bg_pixels:
                    fg_sps_labels_list.append(label)
                else:
                    bg_sps_labels_list.append(label)
                

            fg_sps_labels_list = torch.tensor(fg_sps_labels_list).long().to(device)
            bg_sps_labels_list = torch.tensor(bg_sps_labels_list).long().to(device)
            
            fg_sps_feature_list = [sps_feature_list[label] for label in fg_sps_labels_list]   #前景超像素块的特征
            bg_sps_feature_list = [sps_feature_list[label] for label in bg_sps_labels_list]   #背景超像素块的特征
       
                    
            if len(fg_sps_feature_list) == 0 or len(bg_sps_feature_list)== 0:
                fg_labels_num -= 1
                continue    
                        
            fg_sps_feature_list1 = torch.stack(fg_sps_feature_list)
            bg_sps_feature_list1 = torch.stack(bg_sps_feature_list)



            fg_sps_feature_list1_norm = fg_sps_feature_list1 / fg_sps_feature_list1.norm(dim=1)[:, None]  #归一化超像素特征
            bg_sps_feature_list1_norm = bg_sps_feature_list1 / bg_sps_feature_list1.norm(dim=1)[:, None]  #归一化超像素特征
            

            affinity_matrix_ff = torch.mm(fg_sps_feature_list1_norm, fg_sps_feature_list1_norm.t())
            affinity_matrix_ff = torch.exp(affinity_matrix_ff / temperature)
            
            affinity_matrix_fb = torch.mm(fg_sps_feature_list1_norm, bg_sps_feature_list1_norm.t())
            affinity_matrix_fb = torch.exp(affinity_matrix_fb / temperature)
            
            loss_sp1= get_sp_contrast_loss(fg_sps_labels_list, fg_sps_labels_list, affinity_matrix_ff , affinity_matrix_fb)
                            
            loss_px = get_loss_per_pixels_with_fg_bg_sps(all_perpixel_features, superpixels, fg_sps_labels_list, bg_sps_labels_list, fg_sps_feature_list1_norm , bg_sps_feature_list1_norm , affinity_matrix_fb) #让前景超像素块中的像素特征与所在超像素块的特征更相似，与背景超像素块的特征远离
            
            # loss_proto = Prototypical_loss(fg_sps_labels_list, bg_sps_labels_list, sps_feature_list)
            
            # print("loss_sp1:", loss_sp1.item(), "loss_px:", loss_px.item())

            # print("loss_sp1:",loss_sp1)
            # print("loss_px:",loss_px)
            
            loss_img += loss_sp1 +loss_px   #+ loss_proto
            
        if fg_labels_num != 0 :    
            loss_dc += loss_img /fg_labels_num
        else :
            num_batch -= 1
            continue
    if num_batch !=0:
        loss_dc = loss_dc / num_batch
    else:
        loss_dc = 1e-8

    return loss_dc




###################################################

# Define the dataset class
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, images_dir, masks_dir, suppixel_dir,  filenames):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        # self.signal_dir = signal_dir
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
        # signal_path = os.path.join(self.signal_dir, filename)
        suppixel_path = os.path.join(self.suppixel_dir, filename)

        image = np.load(image_path)
        mask = np.load(mask_path)
        # signal = np.load(signal_path)
        suppixel = np.load(suppixel_path)

        # 转换为 PyTorch 张量
        image = torch.tensor(image.transpose(2, 0, 1), dtype=torch.float32)  # (channels, height, width)
        mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)  # (1, height, width)
        # signal = torch.tensor(signal.transpose(2, 0, 1), dtype=torch.float32)  # (channels, height, width)
        suppixel = torch.tensor(suppixel, dtype=torch.float32).unsqueeze(0)  # (1, height, width)

        return image, mask, suppixel
    
# 获取文件夹中的文件名
def get_filenames_from_folder(folder_path):
    return [filename for filename in os.listdir(folder_path) if filename.endswith('.npy')]

# Filter out background-only patches (required for contrastive learning)
def get_fg_filenames(images_dir, masks_dir):
    filenames = get_filenames_from_folder(images_dir)
    fg_filenames = []
    for f in filenames:
        mask = np.load(os.path.join(masks_dir, f))
        if mask.sum() > 0:  # keep only patches with foreground
            fg_filenames.append(f)
    return fg_filenames   


# 设置训练集和验证集的文件夹路径
# train_images_dir = "/data_nas/gjs/ISF_pixel_level_data/BCSS_x10_reinhard_cut/150/train/Contrast_learning/image_npy"
# train_masks_dir = "/data_nas/gjs/ISF_pixel_level_data/BCSS_x10_reinhard_cut/150/train/Contrast_learning/mask_npy"
# train_superpixel_dir = '/data_nas/gjs/ISF_pixel_level_data/BCSS_x10_reinhard_cut/150/train/Contrast_learning/image_SLIC_600'

# val_images_dir = "/data_nas/gjs/ISF_pixel_level_data/BCSS_x10_reinhard_cut/150/val/Contrast_learning/image_npy"
# val_masks_dir = "/data_nas/gjs/ISF_pixel_level_data/BCSS_x10_reinhard_cut/150/val/Contrast_learning/mask_npy"
# val_superpixel_dir = '/data_nas/gjs/ISF_pixel_level_data/BCSS_x10_reinhard_cut/150/val/Contrast_learning/image_SLIC_600'

# ── Configuration ────────────────────────────────────────────────────────────
FOLD        = 1
PATCHES_DIR = "../data/patches"
SPLITS_JSON = "../data/patches/fold_splits.json"
CLS         = 'tumor'       # class used for contrastive learning
N_SEGMENTS  = 500
# ─────────────────────────────────────────────────────────────────────────────

from dataset import ContrastDataset

train_dataset = ContrastDataset(PATCHES_DIR, SPLITS_JSON, fold=FOLD, split='train',
                                cls=CLS, n_segments=N_SEGMENTS)
val_dataset   = ContrastDataset(PATCHES_DIR, SPLITS_JSON, fold=FOLD, split='val',
                                cls=CLS, n_segments=N_SEGMENTS)
print(f"Train patches: {len(train_dataset)}  Val patches: {len(val_dataset)}")



#####################################################


# 创建模型实例
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = get_efficientunet_b0(out_channels=1, concat_input=True, pretrained=False).to(device)

try:
    from torchinfo import summary
    summary(model, input_size=(1, 3, 512, 512), device=device,
            col_names=["input_size", "output_size", "num_params", "trainable"])
except ImportError:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model params: total={total:,}  trainable={trainable:,}")


if multiGPU :
    # 初始化分布式训练
    # 初始化分布式训练
    def init_distributed_mode():
        if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
            rank = int(os.environ['RANK'])
            world_size = int(os.environ['WORLD_SIZE'])
            dist.init_process_group(backend='nccl', init_method='env://')
            torch.cuda.set_device(rank % torch.cuda.device_count())  # 设置当前进程的GPU
            print(f"Initialized process group with rank {rank} and world size {world_size}")
        else:
            print("Environment variables RANK and WORLD_SIZE are not set.")
        return rank

    rank = init_distributed_mode()
    
    
    model = model.cuda()
    model = DDP(model, device_ids=[rank], output_device=rank)
    train_sampler = DistributedSampler(train_dataset)
    train_loader = DataLoader(train_dataset, batch_size=20, sampler=train_sampler, num_workers=4)
    val_sampler = DistributedSampler(val_dataset)
    val_loader = DataLoader(val_dataset, batch_size=20, sampler=val_sampler, num_workers=4)
    # map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
    # model.load_state_dict(torch.load('/home/gjs/ISF_nuclick/checkpoints_new/contrast_learing/resnet18TCGABR-Rdnct3_spp_ss_dc_ALL.pth' , map_location=map_location), strict=False)
else:
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
    


optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=4e-4)

loss_fn = 1


# Training function
def train_model(model, train_loader, val_loader, optimizer, epochs=50):

    best_val_loss = 10.0
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())
    epoch_pbar = tqdm(range(epochs), desc="Overall Training Progress", unit="epoch")

    for epoch in epoch_pbar:
        model.train()
        train_loss = 0.0

        train_batch_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Training]", leave=False, unit="batch")

        for images, masks, superpixels in train_batch_pbar:
            images, masks, superpixels = images.to(device), masks.to(device), superpixels.to(device)
            optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                all_pixel_features = model(images)

            # loss computed in fp32 (contrastive loss has exp/log — needs fp32 precision)
            loss = calc_dc_loss_sp(masks, all_pixel_features.float(), superpixels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item() * images.size(0)
            train_batch_pbar.set_postfix(batch_loss=f"{loss.item():.4f}")
            
        train_loss /= len(train_loader.dataset)
        
        ###
        model.eval()
        val_loss = 0.0
    
        
        

        with torch.no_grad():
            iou_scores = []
            val_batch_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Validation]", leave=False, unit="batch")

            for images, masks, superpixels in val_batch_pbar:
                images, masks, superpixels = images.to(device), masks.to(device), superpixels.to(device)
                with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                    all_pixel_features = model(images)
                loss = calc_dc_loss_sp(masks, all_pixel_features.float(), superpixels)
                val_loss += loss.item() * images.size(0)
                
                val_batch_pbar.set_postfix(batch_loss=f"{loss.item():.4f}")
            
            val_loss /= len(val_loader.dataset)
            epoch_pbar.set_postfix(train_loss=f"{train_loss:.4f}", val_loss=f"{val_loss:.4f}")
            print(f'Epoch {epoch+1}/{epochs}, Train Loss (CombinedLoss): {train_loss:.4f}, Val Loss : {val_loss:.4f}')
            
            
        # 保存最优模型和最新一个epoch的模型
            if val_loss < best_val_loss or (epoch + 1) > 10:
                # 更新最佳验证损失并保存最佳模型
                if val_loss < best_val_loss :
                    best_val_loss = val_loss
                    checkpoint_dir = f'{PATCHES_DIR}/fold_{FOLD}/efficientUnet'
                    os.makedirs(checkpoint_dir, exist_ok=True)
                    model_filename = f'{checkpoint_dir}/efficientUnet_{epoch+1}_loss{val_loss:.4f}_best.pth'
                    print(f"Epoch {epoch+1}: Model saved with lowest Val Loss: {val_loss:.4f}")
                    torch.save(model.state_dict(), model_filename)
                else:
                    checkpoint_dir = f'{PATCHES_DIR}/fold_{FOLD}/efficientUnet'
                    os.makedirs(checkpoint_dir, exist_ok=True)
                    model_filename = f'{checkpoint_dir}/efficientUnet_{epoch+1}_loss{val_loss:.4f}.pth'
                    print(f"Epoch {epoch+1}: Model saved with Val Loss: {val_loss:.4f}")
                    torch.save(model.state_dict(), model_filename)

train_model(model, train_loader, val_loader, optimizer,  epochs=2)


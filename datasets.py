# !/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@file : datasets.py
@time : 2024/5/9 15:38
@author : zhanglingling
@contact : none
@desc : none
"""
import torch.utils.data
import torchvision.transforms.functional
from PIL import Image
from torch import nn
from pathlib import Path

import os 
import glob
import numpy as np  

from audio import OnnxWenetModel, KWav
import cv2 
import random
import json





# 训练munet的数据集 

class MUNetDataset(torch.utils.data.Dataset):

    def __init__(self, data_folder: Path, wenet_model_path: Path, split='train'):
        self.data_folder = data_folder 
        self.video_folder_list = [] # video文件名和文件夹名相同
        with open(os.path.join(data_folder, f'filelist_{split}.txt')) as f:
            for line in f:
                line = line.strip()
                if ' ' in line: 
                    line = line.split()[0]
                self.video_folder_list.append(os.path.join(data_folder, line))
        # 遍历得到每个video的图片数量 
        self.imgs_num_list = [len(glob.glob(os.path.join(video_folder, 'full_body_img', '*.jpg'))) for video_folder in self.video_folder_list]
        self.audio_fea_map = dict()
        self.model_path = wenet_model_path

    def find_vid_by_idx(self, idx):
        cumulative_sum = 0
        for i, num in enumerate(self.imgs_num_list):
            if cumulative_sum <= idx < cumulative_sum + num:
                relative_index = idx - cumulative_sum
                return i, relative_index
            cumulative_sum += num
        return -1, -1
        

    @classmethod
    def from_config(cls, cfg):
        return {
            'data_folder': Path(cfg.dataset.data_path),
            'wenet_model_path': Path(cfg.dataset.wenet_path),
            'split': cfg.dataset.split,
        }
    
    def __len__(self):
        return np.sum(self.imgs_num_list)
    
    def extract_audio_feature(self, audio_path, img_ind):
        if audio_path in self.audio_fea_map.keys():
            kwav = self.audio_fea_map[audio_path]
        else:
            kwav = KWav(audio_path=audio_path)
            #model_path = r'D:\download\baseConfig\wo.onnx'
            model = OnnxWenetModel(self.model_path)
            model.calcall(kwav, mfcc_spec=None)
            self.audio_fea_map[audio_path] = kwav
        return kwav.inxBuf(img_ind) 
    
    def process_image(self, img_groud, bbox_groud, img_ex, bbox_ex):
        cur_img = img_groud
        x1, x2, y1, y2 = bbox_groud
        bbox_width = x2 - x1
        bbox_height = y2 - y1
        # 从源图像中提取roi
        crop_img = img_groud[y1:y2, x1:x2]
        crop_img = cv2.resize(crop_img, (168, 168), cv2.INTER_AREA)
        img_real = crop_img[4:164, 4:164].copy()
        img_real_ori = img_real.copy()
        img_masked = cv2.rectangle(img_real,(5,5,150,145),(0,0,0),-1)
        
        crop_img = img_ex[y1:y2, x1:x2]
        crop_img = cv2.resize(crop_img, (168, 168), cv2.INTER_AREA)
        img_real_ex = crop_img[4:164, 4:164].copy()
      
        img_real_ori = img_real_ori.transpose(2,0,1).astype(np.float32) # （通道数，高，宽）
        img_masked = img_masked.transpose(2,0,1).astype(np.float32)
        img_real_ex = img_real_ex.transpose(2,0,1).astype(np.float32)
        
        img_real_ex_T = torch.from_numpy(img_real_ex / 255.0)  # 转换到0-1范围内
        img_real_T = torch.from_numpy(img_real_ori / 255.0)
        img_masked_T = torch.from_numpy(img_masked / 255.0)
        img_concat_T = torch.cat([img_real_ex_T, img_masked_T], axis=0)

        return img_concat_T, img_real_T



    def __getitem__(self, idx):
        # 找到属于第几个视频 
        video_ind, img_ind = self.find_vid_by_idx(idx)
        assert video_ind != -1 and img_ind != -1 
        video_folder = self.video_folder_list[video_ind]
        image_files = glob.glob(os.path.join(video_folder, 'full_body_img', '*.jpg'))
        image_files.sort(key=lambda x: int(os.path.basename(x).split('.')[0]))  # 排序
      
        #计算audio特征 
        audio_path = os.path.join(video_folder, 'audio.wav')
        feat = self.extract_audio_feature(audio_path, img_ind)
        # 读取json的bbox 
        bbox_json_path = os.path.join(video_folder, 'bbox.json')
        with open(bbox_json_path, 'r') as f:
            bbox_list = json.load(f)
        try:
            img_groud = cv2.imread(image_files[img_ind])  # 作为groud-truth 
        except IOError:
            print(f"Error: Cannot open or read image file. {image_files[img_ind]}")
        bbox_key = os.path.basename(image_files[img_ind]).split('.')[0]
        bbox_groud = bbox_list[bbox_key]
        # 再选一个其他作为网络输入 
        ex_int = random.randint(0, len(image_files) - 1)
        try:
            img_ex = cv2.imread(image_files[ex_int])
        except IOError:
            print(f"Error: Cannot open or read image file. {image_files[ex_int]}")
        bbox_key_ex = os.path.basename(image_files[ex_int]).split('.')[0]
        bbox_ex= bbox_list[bbox_key_ex]
        img_contact_T, img_real_T = self.process_image(img_groud, bbox_groud, img_ex, bbox_ex)
        feat_T = torch.from_numpy(feat)
        return img_contact_T, img_real_T, feat_T


if __name__ == '__main__':
    print(f'dataset')
    data_folder = Path(r'F:\AI\Ultralight-Digital-Human\test_data')
    wenet_folder = Path(r'D:\download\baseConfig\wo.onnx')

    dataset = MUNetDataset(data_folder, wenet_folder, 'train')
    from torch.utils.data import DataLoader
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    img_concat_T, img_real_T, feat_T = next(iter(dataloader))
    print(f'img_concat_T: {img_concat_T.shape}')
    print(f'img_rea_T: {img_real_T.shape}')
    print(f'feat_T: {feat_T.shape}')

        



        


        



       


   
        





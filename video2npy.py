# -*- coding: UTF-8 -*-
# Local modules
import os
import sys
import argparse
# 3rd-Party Modules
import numpy as np
import pickle as pk
from tqdm import tqdm

# PyTorch Modules
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import torch.optim as optim

from transformers import Wav2Vec2Processor, Wav2Vec2Model

import cv2
import utils
import net


def main(args):
    model_path = args.model_path
    modelWrapper = net.ModelWrapper(args) # Change this to use custom model
    modelWrapper.init_model()
    modelWrapper.load_model(model_path, 'test')
    modelWrapper.set_eval()

    video_dir = "data/VideoFlash"
    save_dir = "data/VideoFlashnpy2"

    vid_mean = np.array([[[[94.9899023]],
                               [[130.94191618]],
                               [[58.42676447]]]])
    vid_std = np.array([[[[44.00460252]],
                              [[28.73382746]],
                              [[22.14879893]]]])

    for file_name in tqdm(os.listdir(video_dir)):
        if file_name.endswith(".flv"):
            video_path = os.path.join(video_dir, file_name)
            cap = cv2.VideoCapture(video_path)
            frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                # 将每一帧转换为 RGB 格式
                frame = cv2.resize(frame, (224, 224))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
            cap.release()

            video = np.stack(frames, axis=0)
            video = torch.from_numpy(video).permute(0, 3, 1, 2).float()
            vid_feature = (video - vid_mean) / (vid_std + 0.000001)

            with torch.no_grad():
                arr = np.zeros((1, 66733))

                wav_feature = torch.from_numpy(arr)
                wav_feature = wav_feature.cuda(non_blocking=True).float()
                vid_feature = torch.unsqueeze(vid_feature, dim=0)
                vid_feature = vid_feature.cuda(non_blocking=True).float()
                output = modelWrapper.feed_forward(wav_feature, vid_feature, attention_mask=wav_feature)

            output = output.cpu().numpy()
            video_array = np.stack(output, axis=0)
            folder_name = os.path.splitext(file_name)[0]
            output_folder_path = os.path.join(save_dir, folder_name)
            os.makedirs(output_folder_path, exist_ok=True)
            for i, array in enumerate(video_array):
                # 为每个帧的数组生成一个对应的 .npy 文件
                for j in range(array.shape[0]):
                    filename = f"exp_{j:03d}.npy"
                    np.save(os.path.join(output_folder_path, filename), array[j])

            # 清空帧序列
            frames = []

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == "__main__":
    # Inputs for the main function
    parser = argparse.ArgumentParser()

    # Experiment Arguments
    parser.add_argument(
        '--device',
        choices=['cuda', 'cpu'],
        default='cuda',
        type=str)
    parser.add_argument(
        '--seed',
        default=0,
        type=int)
    parser.add_argument(
        '--conf_path',
        default="config/conf.json",
        type=str)

    # Data Arguments
    parser.add_argument(
        '--corpus_type',
        default="podcast_v1.7",
        type=str)
    parser.add_argument(
        '--model_type',
        default="wav2vec2",
        type=str)
    parser.add_argument(
        '--label_type',
        choices=['dimensional', 'categorical'],
        default='categorical',
        type=str)

    # Chunk Arguments
    parser.add_argument(
        '--use_chunk',
        default=False,
        type=str2bool)
    parser.add_argument(
        '--chunk_hidden_dim',
        default=256,
        type=int)
    parser.add_argument(
        '--chunk_window',
        default=50,
        type=int)
    parser.add_argument(
        '--chunk_num',
        default=11,
        type=int)

    # Model Arguments
    parser.add_argument(
        '--model_path',
        default=None,
        type=str)
    parser.add_argument(
        '--output_num',
        default=4,
        type=int)
    parser.add_argument(
        '--batch_size',
        default=128,
        type=int)
    parser.add_argument(
        '--hidden_dim',
        default=256,
        type=int)
    parser.add_argument(
        '--num_layers',
        default=3,
        type=int)
    parser.add_argument(
        '--epochs',
        default=100,
        type=int)
    parser.add_argument(
        '--lr',
        default=1e-5,
        type=float)

     # Label Learning Arguments
    parser.add_argument(
        '--label_learning',
        default="multi-label",
        type=str)

    parser.add_argument(
        '--corpus',
        default="USC-IEMOCAP",
        type=str)
    parser.add_argument(
        '--num_classes',
        default="four",
        type=str)
    parser.add_argument(
        '--label_rule',
        default="M",
        type=str)
    parser.add_argument(
        '--partition_number',
        default="1",
        type=str)
    parser.add_argument(
        '--data_mode',
        default="primary",
        type=str)

    parser.add_argument(
        '--output_dim',
        default=6,
        type=int)

    parser.add_argument(
        '--output_dim_2',
        default=1,
        type=int)

    # Transformers Arguments
    parser.add_argument(
        '--attn_dropout', type=float, default=0.1,
        help='attention dropout')
    parser.add_argument(
        '--relu_dropout', type=float, default=0.1,
        help='relu dropout')
    parser.add_argument(
        '--embed_dropout', type=float, default=0.25,
        help='embedding dropout')
    parser.add_argument(
        '--res_dropout', type=float, default=0.1,
        help='residual block dropout')
    parser.add_argument(
        '--out_dropout', type=float, default=0.2,
        help='output layer dropout (default: 0.2')
    parser.add_argument(
        '--layers', type=int, default = 5,
        help='number of layers in the network (default: 5)')
    parser.add_argument(
        '--num_heads', type=int, default = 10,
        help='number of heads for multi-head attention layers(default: 10)')
    parser.add_argument(
        '--attn_mask', action='store_false',
        help='use attention mask for transformer (default: true)')
    parser.add_argument(
        '--clip', type = float, default = 0.8,
        help='gradient clip value (default: 0.8)')
    parser.add_argument(
        '--optim', type = str, default = 'Adam',
        help='optimizer to use (default: Adam)')
    parser.add_argument(
        '--decay', type = int, default = 6,
        help='When to decay learning rate (default: 5)')


    args = parser.parse_args()

    # Call main function
    main(args)


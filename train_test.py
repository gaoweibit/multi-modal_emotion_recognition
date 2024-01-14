# -*- coding: UTF-8 -*-
# Local modules
import os
import sys
import argparse
# 3rd-Party Modules
import numpy as np
import pickle as pk
import pandas as pd
from tqdm import tqdm
from datetime import datetime

# PyTorch Modules
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast

import torch.distributed as dist
dist.init_process_group('gloo', init_method='file:///tmp/somefile', rank=0, world_size=1)

# Self-Written Modules
sys.path.append(os.getcwd())
import utils
from utils.conloss import SupConLoss, Focal_Loss
import net

args = None
train_dataloader = None
test_dataloader = None
# from net import ser, chunk


def train_test(args):
    print(args)
    print(datetime.now())

    utils.set_deterministic(args.seed)
    utils.print_config_description(args.conf_path)  # 输出description

    config_dict = utils.load_env(args.conf_path)
    assert config_dict.get("config_root",
                           None) != None, "No config_root in config/conf.json"
    config_path = os.path.join(
        config_dict["config_root"],
        config_dict[args.corpus_type])  # 通过拼接得到读取到了six.json的路径
    utils.print_config_description(config_path)  # 输出description

    # Make model directory
    model_path = args.model_path  # 用以保存模型的路径
    os.makedirs(model_path, exist_ok=True)  # 创建改路径的目录

    # Initialize dataset
    DataManager = utils.DataManager(config_path)
    lab_type = args.label_type

    if args.label_type == "dimensional":
        assert args.output_num == 6

    if args.label_type == "categorical":
        emo_num = DataManager.get_categorical_emo_num()  # 获得情绪类别个数
        print(emo_num)
        assert args.output_num == emo_num

    audio_path, video_path, label_path = utils.load_audio_and_label_file_paths(
        args)  # 获取路径
    print(audio_path, video_path, label_path)
    """
    将audio与video对比
    得到一一对应的视频文件和语音文件  存在fnames_aud 和fname_vid内
    audio得到的是一个个文件，video得到的是一个个目录    即一个音频文件对应一个视频目录
    """
    fnames_aud, fnames_vid = [], []  # 列表
    v_fnames = os.listdir(video_path)  # 获取目录下所有的目录和文件，保存在列表v_fnames内
    for fname_aud in os.listdir(audio_path):
        if fname_aud.replace('.wav', '') in v_fnames:
            fnames_aud.append(fname_aud)
            fnames_vid.append(fname_aud.replace('.wav', ''))
    fnames_aud.sort()
    fnames_vid.sort()
    """
    分别将audio和video与label对比
    得到有label的每一个文件的具体地址 label_path/filename.wav  存放在train_wav_path和train_vid_path内
    """
    snum = 10000000000000000
    train_wav_path = DataManager.get_wav_path(split_type="train",
                                            wav_loc=audio_path,
                                            fnames=fnames_aud,
                                            lbl_loc=label_path)[:snum]

    train_vid_path = DataManager.get_vid_path(split_type="train",
                                            vid_loc=video_path,
                                            fnames=fnames_vid,
                                            lbl_loc=label_path)[:snum]

    train_utts = [fname.split('/')[-1] for fname in train_wav_path]  # 得到文件名

    # 返回数组，每一行是一个情绪打分序列
    train_labs = DataManager.get_msp_labels(train_utts,
                                            lab_type=lab_type,
                                            lbl_loc=label_path)

    # 得到所有文件的列表
    train_wavs = utils.WavExtractor(train_wav_path).extract()

    train_vids = utils.VidExtractor(train_vid_path).extract()

    train_set = utils.AudVidSet(
        train_wavs,
        train_vids,
        train_labs,
        train_utts,
        print_dur=True,
        lab_type=lab_type,
        label_config=DataManager.get_label_config(lab_type))


    # 将训练集的音频和视频的标准化统计信息保存到文件中，一遍在测试和预测时使用
    train_set.save_norm_stat(model_path + "/train_norm_stat.pkl")

    train_dataloader = DataLoader(train_set,num_workers=args.batch_size,
                batch_size=args.batch_size,
                collate_fn=utils.collate_fn_padd,
                shuffle=True)

    # test dataloader
    test_wav_path = DataManager.get_wav_path(split_type="test",wav_loc=audio_path, fnames=fnames_aud, lbl_loc=label_path)
    test_vid_path = DataManager.get_vid_path(split_type="test",vid_loc=video_path, fnames=fnames_vid, lbl_loc=label_path)
    test_wav_path.sort()

    test_utts = [fname.split('/')[-1] for fname in test_wav_path]
    test_utts.sort()
    test_labs = DataManager.get_msp_labels(test_utts, lab_type=lab_type,lbl_loc=label_path)

    test_wavs = utils.WavExtractor(test_wav_path).extract()
    test_vids = utils.VidExtractor(test_vid_path).extract()
    ###################################################################################################
    with open(args.model_path+"/train_norm_stat.pkl", 'rb') as f:
        wav_mean, wav_std, vid_mean, vid_std = pk.load(f)

    test_set = utils.AudVidSet(test_wavs, test_vids, test_labs, test_utts, 
        print_dur=True, lab_type=lab_type,
        wav_mean = wav_mean, wav_std = wav_std,
        vid_mean = vid_mean, vid_std = vid_std, 
        label_config = DataManager.get_label_config(lab_type)
    )
    test_loader = DataLoader(test_set, batch_size=1, collate_fn=utils.collate_fn_padd, shuffle=False)

    lm = utils.LogManager()
    if args.label_type == "dimensional":
        lm.alloc_stat_type_list(["test_aro", "test_dom", "test_val"])
    elif args.label_type == "categorical":
        lm.alloc_stat_type_list(["test_loss", "test_acc"])


    modelWrapper = net.ConModelWrapper(args)
    modelWrapper.init_model()
    modelWrapper.init_optimizer()
    modelWrapper.load_model("wav2vec2-large-robust-finetunned/model/wav2vec2",
                            'train')

    # 添加：
    focal_loss = Focal_Loss()

    # Initialize loss function
    if args.label_type == "dimensional":
        lm.alloc_stat_type_list([
            "train_aro", "train_dom", "train_val", "dev_aro", "dev_dom",
            "dev_val"
        ])
    elif args.label_type == "categorical":
        lm.alloc_stat_type_list(["train_loss", "train_acc"])

    epochs = args.epochs
    losses_train = []
    test_acc_lst = []
    acc = 0.
    low_count = 0
    max_low_count = 3
    for epoch in range(epochs):
        print(datetime.now())
        print("Epoch:", epoch)
        lm.init_stat()
        modelWrapper.set_train()  # 训练模式
        for xy_pair in tqdm(train_dataloader):
            xa = xy_pair[0]  
            xv = xy_pair[1]
            y = xy_pair[2]
            mask = xy_pair[3]

            # randomly shutting off modalities
            """
            0.2的几率关闭audio模态，0.2的几率关闭video模态，且两模态不会同时关闭
            """
            p1 = 0.2
            p2 = 0.2

            # 生成0~1之前的随机数
            randn = torch.rand(1)

            if randn < p1:
                xa *= 0

            elif randn > p1 and randn < p1 + p2:
                xv *= 0

            # randomly shutting off modalities

            xa = xa.cuda(non_blocking=True).float()
            xv = xv.cuda(non_blocking=True).float()
            y = y.cuda(non_blocking=True).float()
            mask = mask.cuda(non_blocking=True).float()

            # 自动类型转换 混合精度训练
            with autocast():
                preds_va, loss1, loss2 = modelWrapper.feed_forward(
                    xa, xv, attention_mask=mask)

                if args.label_type == "categorical":
                    if args.label_learning == "hard-label":
                        """
                        loss
                        """
                        lossva = focal_loss(F.softmax(preds_va, dim=-1), y)
                        w1, wva, w_avm = 0.5*(1-args.loss_w), args.loss_w, 0.5*(1-args.loss_w)
                        loss = w1 * loss1 + w_avm * loss2 + wva * lossva
                    pred = preds_va
                    acc = utils.calc_acc(pred, y)

            # Backpropagation
            modelWrapper.backprop(loss)
            ccc = []
            if args.label_type == "dimensional":
                lm.add_torch_stat("train_aro", ccc[0])
                lm.add_torch_stat("train_dom", ccc[1])
                lm.add_torch_stat("train_val", ccc[2])
            elif args.label_type == "categorical":
                lm.add_torch_stat("train_loss", loss)
                lm.add_torch_stat("train_acc", acc)

        # eval
        modelWrapper.set_eval()
        with torch.no_grad():
            total_pred = [] 
            total_y = []
            for xy_pair in tqdm(test_loader):
                xa = xy_pair[0]
                xv = xy_pair[1]
                y = xy_pair[2]
                mask = xy_pair[3]

                xa=xa.cuda(non_blocking=True).float()
                xv=xv.cuda(non_blocking=True).float()
                y=y.cuda(non_blocking=True).float()
                mask=mask.cuda(non_blocking=True).float()

                pred, loss1, loss2 = modelWrapper.feed_forward(
                        xa, xv, attention_mask=mask)

                total_pred.append(pred)
                total_y.append(y)
                
            total_pred = torch.cat(total_pred)
            total_y = torch.cat(total_y)
        

        if args.label_type == "categorical":
            if args.label_learning == "hard-label":
                loss = focal_loss(F.softmax(total_pred, dim=-1), total_y)                

            acc = utils.calc_acc(total_pred, total_y)
            lm.add_torch_stat("test_loss", loss)
            lm.add_torch_stat("test_acc", acc)
        
        total_pred_np = total_pred.detach().cpu().numpy()
        total_y_np = total_y.detach().cpu().numpy()

        total_pred_pd = pd.DataFrame(total_pred_np)
        total_y_pd = pd.DataFrame(total_y_np)

        info_list = model_path.split("/")
        
        save_path = model_path + '/predictions/test/'

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        total_pred_pd.to_csv(save_path+'y_pred.csv')
        total_y_pd.to_csv(save_path+'y_true.csv')    

        wa, ua = utils.scores(model_path + '/predictions/test/')
        if np.isnan(wa):
            wa = 0.
        if np.isnan(ua):
            us = 0.

        lm.print_stat()
        trial_score = 0.5*(wa+ua)
        
        if len(test_acc_lst) == 0:
            test_acc_lst.append(trial_score)
        elif trial_score <= max(test_acc_lst):
            low_count += 1
            test_acc_lst.append(trial_score)
        else:
            low_count = 0
            test_acc_lst.append(trial_score)     
        
        lm.print_stat()
        if args.label_type == "dimensional":
            dev_loss = 3.0 - lm.get_stat("dev_aro") - lm.get_stat(
                "dev_dom") - lm.get_stat("dev_val")
        elif args.label_type == "categorical":
            tr_loss = lm.get_stat("train_loss")
            losses_train.append(tr_loss)

        if low_count >= max_low_count:
            break

    with open(model_path + '/train_loss.txt', 'w') as f:
        for item in losses_train:
            f.write("%s\n" % item)


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
    parser.add_argument('--device',
                        choices=['cuda', 'cpu'],
                        default='cuda',
                        type=str)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--conf_path', default="config/conf.json", type=str)

    # Data Arguments
    parser.add_argument('--corpus_type', default="podcast_v1.7", type=str)
    parser.add_argument('--model_type', default="wav2vec2", type=str)
    parser.add_argument('--label_type',
                        choices=['dimensional', 'categorical'],
                        default='categorical',
                        type=str)

    # Chunk Arguments
    parser.add_argument('--use_chunk', default=False, type=str2bool)
    parser.add_argument('--chunk_hidden_dim', default=256, type=int)
    parser.add_argument('--chunk_window', default=50, type=int)
    parser.add_argument('--chunk_num', default=11, type=int)

    # Model Arguments
    parser.add_argument('--model_path', default=None, type=str)

    parser.add_argument('--output_num', default=4, type=int)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--hidden_dim', default=256, type=int)
    parser.add_argument('--num_layers', default=3, type=int)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--lr', default=1e-5, type=float)
    parser.add_argument('--loss_w', default=0.7, type=float)

    # Label Learning Arguments
    # 需要调整
    parser.add_argument('--label_learning', default="multi-label", type=str)

    parser.add_argument('--corpus', default="USC-IEMOCAP", type=str)
    parser.add_argument('--num_classes', default="four", type=str)
    parser.add_argument('--label_rule', default="M", type=str)
    parser.add_argument('--partition_number', default="1", type=str)
    parser.add_argument('--data_mode', default="primary", type=str)
    parser.add_argument('--v_dim', default=50, type=int)
    parser.add_argument('--fold', default='1', type=str)
    parser.add_argument('--output_dim', default=6, type=int)

    parser.add_argument('--output_dim_2', default=1, type=int)

    # Transformers Arguments
    parser.add_argument('--attn_dropout',
                        type=float,
                        default=0.1,
                        help='attention dropout')
    parser.add_argument('--relu_dropout',
                        type=float,
                        default=0.1,
                        help='relu dropout')
    parser.add_argument('--embed_dropout',
                        type=float,
                        default=0.25,
                        help='embedding dropout')
    parser.add_argument('--res_dropout',
                        type=float,
                        default=0.1,
                        help='residual block dropout')
    parser.add_argument('--out_dropout',
                        type=float,
                        default=0.2,
                        help='output layer dropout (default: 0.2')
    parser.add_argument('--layers',
                        type=int,
                        default=5,
                        help='number of layers in the network (default: 5)')
    parser.add_argument(
        '--num_heads',
        type=int,
        default=10,
        help='number of heads for multi-head attention layers(default: 10)')
    parser.add_argument(
        '--attn_mask',
        action='store_false',
        help='use attention mask for transformer (default: true)')
    parser.add_argument('--clip',
                        type=float,
                        default=0.8,
                        help='gradient clip value (default: 0.8)')
    parser.add_argument('--optim',
                        type=str,
                        default='Adam',
                        help='optimizer to use (default: Adam)')
    parser.add_argument('--decay',
                        type=int,
                        default=6,
                        help='When to decay learning rate (default: 5)')

    args = parser.parse_args()

    # Call main function
    train_test(args)

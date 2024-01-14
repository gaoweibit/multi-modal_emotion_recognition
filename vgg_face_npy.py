import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchfile
from torchvision import transforms
from PIL import Image
from torch.utils.data import DataLoader

class ImgDataset(torch.utils.data.Dataset):
    def __init__(self, img_path):
        super().__init__()
        self.transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
        ])
        self.root = img_path
        self.img_dir_lst = os.listdir(img_path)
        self.img_lst = []
        for img_dir in self.img_dir_lst:
            img_file_lst = os.listdir(os.path.join(self.root, img_dir))
            for img_file in img_file_lst:
                self.img_lst.append(os.path.join(self.root, img_dir, img_file))

    def __getitem__(self, idx):
        img = self.transform(Image.open(self.img_lst[idx]))
        img_name = os.path.basename(self.img_lst[idx])
        img_dir = os.path.basename(os.path.dirname(self.img_lst[idx]))
        return img, img_dir, img_name

    def __len__(self):
        return len(self.img_lst)

class VGG_16(nn.Module):
    """
    Main Class
    """

    def __init__(self):
        """
        Constructor
        """
        super().__init__()
        self.block_size = [2, 2, 3, 3, 3]
        self.conv_1_1 = nn.Conv2d(3, 64, 3, stride=1, padding=1)
        self.conv_1_2 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.conv_2_1 = nn.Conv2d(64, 128, 3, stride=1, padding=1)
        self.conv_2_2 = nn.Conv2d(128, 128, 3, stride=1, padding=1)
        self.conv_3_1 = nn.Conv2d(128, 256, 3, stride=1, padding=1)
        self.conv_3_2 = nn.Conv2d(256, 256, 3, stride=1, padding=1)
        self.conv_3_3 = nn.Conv2d(256, 256, 3, stride=1, padding=1)
        self.conv_4_1 = nn.Conv2d(256, 512, 3, stride=1, padding=1)
        self.conv_4_2 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.conv_4_3 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.conv_5_1 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.conv_5_2 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.conv_5_3 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.fc6 = nn.Linear(512 * 7 * 7, 4096)
        self.fc7 = nn.Linear(4096, 4096)
        self.fc8 = nn.Linear(4096, 2622)


    def load_weights(self, path="/root/workspace/code/pretrained/VGG_FACE.t7"):
        """ Function to load luatorch pretrained

        Args:
            path: path for the luatorch pretrained
        """
        model = torchfile.load(path)
        counter = 1
        block = 1
        for i, layer in enumerate(model.modules):
            if layer.weight is not None:
                if block <= 5:
                    self_layer = getattr(self, "conv_%d_%d" % (block, counter))
                    counter += 1
                    if counter > self.block_size[block - 1]:
                        counter = 1
                        block += 1
                    self_layer.weight.data[...] = torch.tensor(layer.weight).view_as(self_layer.weight)[...]
                    self_layer.bias.data[...] = torch.tensor(layer.bias).view_as(self_layer.bias)[...]
                else:
                    self_layer = getattr(self, "fc%d" % (block))
                    block += 1
                    self_layer.weight.data[...] = torch.tensor(layer.weight).view_as(self_layer.weight)[...]
                    self_layer.bias.data[...] = torch.tensor(layer.bias).view_as(self_layer.bias)[...]

    def forward(self, x):
        """ Pytorch forward

        Args:
            x: input image (224x224)

        Returns: class logits

        """
        x = F.relu(self.conv_1_1(x))
        x = F.relu(self.conv_1_2(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv_2_1(x))
        x = F.relu(self.conv_2_2(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv_3_1(x))
        x = F.relu(self.conv_3_2(x))
        x = F.relu(self.conv_3_3(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv_4_1(x))
        x = F.relu(self.conv_4_2(x))
        x = F.relu(self.conv_4_3(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv_5_1(x))
        x = F.relu(self.conv_5_2(x))
        x = F.relu(self.conv_5_3(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc6(x))
        x = F.dropout(x, 0.5, self.training)
        return self.fc7(x)

if __name__ == "__main__":
    # 初始化模型并加载权重
    model = VGG_16()
    model.load_weights()
    model = model.to('cuda')

    # 输入文件夹和输出文件夹的路径
    input_folder = '/root/autodl-tmp/data/multi_modal/iemocap_processed/IEMOCAP/png'
    output_folder = '/root/autodl-tmp/data/multi_modal/iemocap_processed/IEMOCAP/npy'

    dataset = ImgDataset(input_folder)
    dataloader = DataLoader(dataset, batch_size=32, num_workers=32, drop_last=False)

    model.eval()
    for idx, data in enumerate(dataloader):
        img = data[0].to('cuda')
        img_dir = data[1]
        img_name = data[2]
        feature_vector = model(img)

        for i in range(len(img_dir)):
            # 保存特征向量到文件
            output_subfolder = os.path.join(output_folder, img_dir[i])
            os.makedirs(output_subfolder, exist_ok=True)
            feature_file_name = os.path.splitext(img_name[i])[0] + ".npy"
            feature_file_path = os.path.join(output_subfolder, feature_file_name)
            np.save(feature_file_path, feature_vector[i].detach().cpu().numpy())

            print(f"Saved feature vector for {img_name[i]} in {feature_file_path}")


                    

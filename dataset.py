import os
import cv2
import torch
import random
import numpy as np
import PIL.Image as Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from data_util import RandomCrop, RandomRotation, RandomResizedCrop, RandomHorizontallyFlip, RandomVerticallyFlip


class TrainDataset(Dataset):
    def __init__(self, opt):
        super().__init__()
        self.image_size = opt.image_size

        self.dataset = os.path.join(opt.train_dataset, 'train.txt')
        self.image_path = opt.train_dataset
        self.mat_files = open(self.dataset, 'r').readlines()
        self.file_num = len(self.mat_files)
        transform_list = []

        transform_list += [transforms.ToTensor(),
                           transforms.Normalize((0.5, 0.5, 0.5),
                                                (0.5, 0.5, 0.5))]
        self.transform = transforms.Compose(transform_list)

    def __len__(self):
        return self.file_num

    def __getitem__(self, idx):
        file_name = self.mat_files[idx % self.file_num]

        img_file = file_name.split(' ')[0]
        gt_file = file_name.split(' ')[1].strip()

        label = img_file.split('/')[-1][:-4]
        label = np.array([np.float32(i) for i in label])

        in_img = cv2.imread(self.image_path+img_file)
        gt_img = cv2.imread(self.image_path+gt_file)

        inp_img = Image.fromarray(in_img)
        tar_img = Image.fromarray(gt_img)

        ps = self.image_size
        w, h = tar_img.size
        padw = ps - w if w < ps else 0
        padh = ps - h if h < ps else 0

        # Reflect Pad in case image is smaller than patch_size
        if padw != 0 or padh != 0:
            inp_img = TF.pad(inp_img, (0, 0, padw, padh), padding_mode='reflect')
            tar_img = TF.pad(tar_img, (0, 0, padw, padh), padding_mode='reflect')

        inp_img = self.transform(inp_img)
        tar_img = self.transform(tar_img)

        hh, ww = tar_img.shape[1], tar_img.shape[2]

        rr = random.randint(0, hh - ps)
        cc = random.randint(0, ww - ps)
        aug = random.randint(0, 8)

        # Crop patch
        inp_img = inp_img[:, rr:rr + ps, cc:cc + ps]
        tar_img = tar_img[:, rr:rr + ps, cc:cc + ps]

        # Data Augmentations
        if aug == 1:
            inp_img = inp_img.flip(1)
            tar_img = tar_img.flip(1)
        elif aug == 2:
            inp_img = inp_img.flip(2)
            tar_img = tar_img.flip(2)
        elif aug == 3:
            inp_img = torch.rot90(inp_img, dims=(1, 2))
            tar_img = torch.rot90(tar_img, dims=(1, 2))
        elif aug == 4:
            inp_img = torch.rot90(inp_img, dims=(1, 2), k=2)
            tar_img = torch.rot90(tar_img, dims=(1, 2), k=2)
        elif aug == 5:
            inp_img = torch.rot90(inp_img, dims=(1, 2), k=3)
            tar_img = torch.rot90(tar_img, dims=(1, 2), k=3)
        elif aug == 6:
            inp_img = torch.rot90(inp_img.flip(1), dims=(1, 2))
            tar_img = torch.rot90(tar_img.flip(1), dims=(1, 2))
        elif aug == 7:
            inp_img = torch.rot90(inp_img.flip(2), dims=(1, 2))
            tar_img = torch.rot90(tar_img.flip(2), dims=(1, 2))


        sample = {'in_img': inp_img, 'gt_img': tar_img, 'label': label}
        return sample


class ValDataset(Dataset):
    def __init__(self, opt):
        super(ValDataset, self).__init__()

        self.dataset = os.path.join(opt.val_dataset, 'test.txt')
        self.image_path = opt.val_dataset

        self.mat_files = open(self.dataset, 'r').readlines()
        self.file_num = len(self.mat_files)

        self.ps = opt.image_size

        transform_list = []

        transform_list += [transforms.ToTensor(),
                           transforms.Normalize((0.5, 0.5, 0.5),
                                                (0.5, 0.5, 0.5))]
        self.transform = transforms.Compose(transform_list)

    def __len__(self):
        return self.file_num

    def __getitem__(self, idx):
        file_name = self.mat_files[idx % self.file_num]

        img_file = file_name.split(' ')[0]
        gt_file = file_name.split(' ')[1].strip()

        in_img = cv2.imread(self.image_path + img_file)
        gt_img = cv2.imread(self.image_path + gt_file)

        in_img = Image.fromarray(in_img)
        gt_img = Image.fromarray(gt_img)

        ps = self.ps
        in_img = TF.center_crop(in_img, (ps,ps))
        gt_img = TF.center_crop(gt_img, (ps,ps))
        in_img = self.transform(in_img)
        gt_img = self.transform(gt_img)


        sample = {'in_img': in_img, 'gt_img': gt_img}
        return sample

class TestDataset(Dataset):
    def __init__(self, opt):
        super().__init__()
        self.dataset = os.path.join(opt.test_dataset, 'test.txt')
        self.image_path = opt.dataset_path

        self.mat_files = open(self.dataset, 'r').readlines()
        self.file_num = len(self.mat_files)
        transform_list = []

        transform_list += [transforms.ToTensor(),
                           transforms.Normalize((0.5, 0.5, 0.5),
                                                (0.5, 0.5, 0.5))]
        self.transform = transforms.Compose(transform_list)

    def __len__(self):
        return self.file_num

    def __getitem__(self, idx):
        file_name = self.mat_files[idx % self.file_num]

        img_file = file_name.split(' ')[0]
        gt_file = file_name.split(' ')[1].strip()

        in_img = cv2.imread(self.image_path + img_file)
        gt_img = cv2.imread(self.image_path + gt_file)

        in_img = Image.fromarray(in_img)
        gt_img = Image.fromarray(gt_img)

        in_img = self.transform(in_img)
        gt_img = self.transform(gt_img)

        img_path = img_file.split('/')
        sample = {'in_img': in_img, 'gt_img': gt_img, 'image_name':img_path[-1][:-4]}

        return sample

class TestRealDataset(Dataset):
    def __init__(self, opt):
        super().__init__()
        self.root_dir = './datasets/Real'  #unpaired real world degraded images
        self.mat_files = os.listdir(self.root_dir)

        self.file_num = len(self.mat_files)
        transform_list = []

        transform_list += [transforms.ToTensor(),
                           transforms.Normalize((0.5, 0.5, 0.5),
                                                (0.5, 0.5, 0.5))]
        self.transform = transforms.Compose(transform_list)

    def __len__(self):
        return self.file_num

    def __getitem__(self, idx):
        file_name = self.mat_files[idx % self.file_num]

        img_file = os.path.join(self.root_dir, file_name)
        in_img = cv2.imread(img_file)

        gt_img = in_img

        in_img = Image.fromarray(in_img)
        gt_img = Image.fromarray(gt_img)

        in_img = self.transform(in_img)
        gt_img = self.transform(gt_img)

        img_path = file_name.split('.')
        sample = {'in_img': in_img, 'gt_img': gt_img,
                  'image_name': img_path[0] + '_restored'}

        return sample
from math import floor, ceil
from PIL import Image

import torchvision.transforms as transforms

import random
import torch
import utils
import os


class DatasetKfold(torch.utils.data.Dataset):
    # Works only if the size(img_dir) >= size(mask_dir)
    def __init__(self, img_dir, mask_dir, actual_fold, dataset="train", folds=5):
        super().__init__()
        self.imgs = img_dir
        self.masks = mask_dir
        self.transform = transforms.ToTensor()
        self.folds = folds

        self.files = [file for file in os.listdir(self.masks) if file in os.listdir(self.imgs)]
        self.files = list(utils.split(self.files, self.folds))

        self.dataset = []
        if dataset == "train":
            for i in range(self.folds):
                if i != actual_fold:
                    self.dataset = self.dataset + self.files[i]
            print("Training set length: {}".format(len(self.dataset)))
        elif dataset == "validate":
            self.dataset = self.files[actual_fold]
            print("Validation set length: {}".format(len(self.dataset)))
        random.shuffle(self.dataset)

    def __len__(self):
        """Returns the length of the dataset"""
        return len(self.dataset)

    def __getitem__(self, index):
        file_name = self.dataset[index]
        # Retreiving images and masks
        img = Image.open(os.path.join(self.imgs, file_name))
        mask = Image.open(os.path.join(self.masks, file_name))
        image_tensor = self.transform(img)
        img.close()
        mask_tensor = self.transform(mask)
        mask.close()

        # Removing alpha value, if any;
        image_tensor = image_tensor[:3, :, :]

        # Transforming images and masks to size 512x512
        if image_tensor.shape[1] == image_tensor.shape[2]:
            # Images are square, only resizing
            tr = transforms.Resize(size=(512, 512))
            return tr(image_tensor), tr(mask_tensor), file_name

        # Images are not square, first resizing with a max
        # length of 512 then adding padding
        tr = transforms.Resize(size=511, max_size=512)
        image_tensor = tr(image_tensor)
        mask_tensor = tr(mask_tensor)
        if image_tensor.shape[1] > image_tensor.shape[2]:
            padding = (image_tensor.shape[1] - image_tensor.shape[2]) / 2
            tr = transforms.Pad((ceil(padding), 0, floor(padding), 0))
        else:
            padding = (image_tensor.shape[2] - image_tensor.shape[1]) / 2
            tr = transforms.Pad((0, ceil(padding), 0, floor(padding)))
        return tr(image_tensor), tr(mask_tensor), file_name

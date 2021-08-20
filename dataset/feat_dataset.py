import math
import os
import random

import numpy as np
import pandas as pd
import torch
import torch.utils.data as data
import utils
from PIL import Image


class FeatDataset(data.Dataset):
    def __init__(
        self,
        mode="train",
        excel_dir="data/training/information.xlsx",
        feat_model="vgg",
        config=None,
    ):
        self.feat_model = feat_model
        df = pd.read_excel(excel_dir)
        df = df[df["mode"] == mode]
        self.files = df.filename.values
        self._config = config
        self._mode = mode

    def __getitem__(self, idx):
        path = (
            "./data/training/feature_ext/"
            + self.feat_model
            + "/"
            + self.files[idx][:-4]
            + ".pth"
        )

        feature = torch.load(path)
        labelpath = (
            "./data/training/feature_ext/"
            + self.feat_model
            + "/"
            + self.files[idx][:-4]
            + ".txt"
        )
        with open(labelpath, mode="r") as f:
            lines = f.read().splitlines()
        labels = [utils.label_to_id(i) for i in lines]
        labels = torch.tensor(labels)
        if self._mode == "test":
            return feature.unsqueeze(0), labels.unsqueeze(0)
        if self._config.head == "lstm" or self._config.head == "tcn":
            return lstm_slice_dataset(feature, labels, self._config.batch_size)
        bd, bl = batch_maker(
            feature,
            labels,
            shuffle=self._config.shuffle,
            drop_last=True,
            batch_size=self._config.batch_size,
            n_sample=self._config.n_sample,
        )
        return bd, bl

    def __len__(self):
        return len(self.files)


class ImgDataset(data.Dataset):
    def __init__(
        self,
        mode="train",
        excel_dir="./data/training/information.xlsx",
        feat_model="vgg",
        transform=None,
    ):
        self.feat_model = feat_model
        df = pd.read_excel(excel_dir)
        df = df[df["mode"] == mode]
        self.files = df.filename.values
        self.transform = transform

    def __getitem__(self, idx):
        path = os.path.join(
            "./data/training/tmp_images/",
            self.files[idx][:-4] + "_resized",
        )
        files = os.listdir(path)
        images = [f for f in files if f[-3:] == "jpg"]
        texts = [f for f in files if f[-3:] == "txt"]
        images.sort()
        texts.sort()
        num = min(len(images), len(texts))
        if len(images) > len(texts):
            images = images[:num]
        elif len(images) < len(texts):
            texts = texts[:num]

        if len(images) != len(texts):
            assert "image size and texts size not match"
            exit()

        labels = []
        for i in texts:
            with open(os.path.join(path, i), mode="r") as f:
                labels.append(f.read())
        labels = [utils.label_to_id(i) for i in labels]
        labels = torch.tensor(labels)
        imgs = []
        for i in images:
            im = Image.open(os.path.join(path, i))
            im = self.transform(im)
            imgs.append(im.unsqueeze(0))
        imgs = torch.cat(imgs, axis=0)
        return imgs, labels

    def __len__(self):
        return len(self.files)


def batch_maker(data, label, shuffle=True, drop_last=True, batch_size=8, n_sample=10):
    if shuffle:
        p = torch.randperm(data.size()[0])
        data = data[p]
        label = label[p]
    n = len(data) // batch_size

    batched_data = []
    batched_label = []
    if not drop_last:
        n += 1
    for i in range(n):
        if n_sample == i:
            break
        b = data[i * batch_size : i * batch_size + batch_size, :, :, :]
        lb = label[i * batch_size : i * batch_size + batch_size]
        batched_data.append(b.unsqueeze(0))
        batched_label.append(lb.unsqueeze(0))
    batched_data = torch.cat(batched_data, 0)
    batched_label = torch.cat(batched_label, 0)
    return batched_data, batched_label


def lstm_slice_dataset(data, label, batch_size):
    n = len(data) - batch_size
    index = random.randint(0, n)
    data = data[index : index + batch_size]
    label = label[index : index + batch_size]
    return data.unsqueeze(0), label.unsqueeze(0)


if __name__ == "__main__":
    # d = featDataset()
    d = imgDataset()

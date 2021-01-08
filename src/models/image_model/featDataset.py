import math
import os

import numpy as np
import pandas as pd
import torch
import torch.utils.data as data
from PIL import Image

import utils


class featDataset(data.Dataset):
    def __init__(
        self,
        mode="train",
        excel_dir="../../../data/training/information.xlsx",
        feat_model="vgg",
    ):
        self.feat_model = feat_model
        df = pd.read_excel(excel_dir)
        df = df[df["mode"] == mode]
        self.files = df.filename.values

    def __getitem__(self, idx):
        path = (
            "../../../data/training/feature_ext/"
            + self.feat_model
            + "/"
            + self.files[idx][:-4]
            + ".pth"
        )
        # path = "../../../data/feature_ext/" + self.feat_model + "/" + "r25" + ".pth"
        feature = torch.load(path)
        labelpath = (
            "../../../data/training/feature_ext/"
            + self.feat_model
            + "/"
            + self.files[idx][:-4]
            + ".txt"
        )
        with open(labelpath, mode="r") as f:
            lines = f.read().splitlines()
        labels = [utils.label_to_id(i) for i in lines]
        labels = torch.tensor(labels)
        return feature, labels

    def __len__(self):
        return len(self.files)


class imgDataset(data.Dataset):
    def __init__(
        self,
        mode="train",
        excel_dir="../../../data/training/information.xlsx",
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
            "../../../data/training/tmp_images/",
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
        # print(len(images), len(texts))

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


if __name__ == "__main__":
    # d = featDataset()
    d = imgDataset()

    d[0]

#

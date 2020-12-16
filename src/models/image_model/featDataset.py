import os
import pandas as pd
import torch
import torch.utils.data as data
from PIL import Image
import math
import numpy as np

import utils


class featDataset(data.Dataset):
    def __init__(
        self, mode="train", excel_dir="../../../data/information.xlsx", feat_model="vgg"
    ):
        self.feat_model = "vgg"
        df = pd.read_excel(excel_dir)
        df = df[df["mode"] == mode]
        self.files = df.filename.values

    def __getitem__(self, idx):
        path = (
            "../../../data/feature_ext/"
            + self.feat_model
            + "/"
            + self.files[idx][:-4]
            + ".pth"
        )
        # path = "../../../data/feature_ext/" + self.feat_model + "/" + "r25" + ".pth"
        # print(path)
        feature = torch.load(path)
        # print(feature.shape)
        labelpath = (
            "../../../data/feature_ext/"
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


if __name__ == "__main__":
    d = featDataset()
    d[0]

#

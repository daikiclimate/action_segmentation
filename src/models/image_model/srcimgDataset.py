import math
import os

import numpy as np
import pandas as pd
import torch
import torch.utils.data as data
from PIL import Image


class Dataset(data.Dataset):
    def __init__(self, excel_dir="../../../data/information.xlsx", feat_model="vgg"):
        self.feat_model = "vgg"
        df = pd.read_excel(excel_dir)
        df = df[df["mode"] == "train"]
        self.files = df.filename.values

    def __getitem__(self, idx):
        path = (
            "../../../data/feature_ext/"
            + self.feat_model
            + "/"
            + self.files[idx][:-4]
            + ".pth"
        )
        path = "../../../data/feature_ext/" + self.feat_model + "/" + "r25" + ".pth"
        # print(path)
        feature = torch.load(path)
        print(feature.shape)

        return feature

    def __len__(self):
        return len(self.files)


if __name__ == "__main__":
    d = Dataset()
    d[1]
#

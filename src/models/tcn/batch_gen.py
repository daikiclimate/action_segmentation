import os
import random
import sys

import numpy as np
import pandas as pd
import torch

sys.path.append("../")
import utils


class BatchGenerator(object):
    def __init__(self, num_classes, actions_dict, gt_path, features_path):
        # def __init__(self, num_classes, actions_dict, gt_path, features_path, sample_rate):
        self.list_of_examples = list()
        self.index = 0
        self.num_classes = num_classes
        self.actions_dict = actions_dict
        self.gt_path = gt_path
        self.features_path = features_path
        self.feat_model = "vgg"
        # self.sample_rate = sample_rate

    def reset(self):
        self.index = 0
        random.shuffle(self.list_of_examples)

    def has_next(self):
        if self.index < len(self.list_of_examples):
            return True
        return False

    def read_data(
        self, mode="train", excel_dir="../../../data/training/information.xlsx"
    ):
        # self.feat_model = feat_model
        df = pd.read_excel(excel_dir)
        df = df[df["mode"] == mode]
        self.list_of_examples = df.filename.values

        # file_ptr = open(vid_list_file, 'r')
        # self.list_of_examples = file_ptr.read().split('\n')[:-1]
        # file_ptr.close()
        random.shuffle(self.list_of_examples)

    def next_batch(self, batch_size):
        batch = self.list_of_examples[self.index : self.index + batch_size]
        self.index += batch_size

        batch_input = []
        batch_target = []
        for vid in batch:
            if vid == "r23mp4":
                vid = "r23.mp4"
            features = torch.load(
                os.path.join(self.features_path, vid.split(".")[0] + ".pth")
            )
            labelpath = (
                "../../../data/training/feature_ext/"
                + self.feat_model
                + "/"
                + vid[:-4]
                + ".txt"
            )
            with open(labelpath, mode="r") as f:
                lines = f.read().splitlines()
            labels = [utils.label_to_id(i) for i in lines]
            labels = torch.tensor(labels)

            # file_ptr = open(self.gt_path + vid, "r")
            # content = file_ptr.read().split("\n")[:-1]
            # classes = np.zeros(min(np.shape(features)[1], len(content)))
            # for i in range(len(classes)):
            # classes[i] = self.actions_dict[content[i]]
            batch_input.append(features)
            batch_target.append(labels)
            # batch_target.append(classes[:: self.sample_rate])
        return features.unsqueeze(0), labels.unsqueeze(0)
        # # return torch.tensor(np.array(batch_input)), torch.tensor(np.array(batch_target))
        # length_of_sequences = map(len, batch_target)
        # ml = max(length_of_sequences)
        #
        # batch_input_tensor = torch.zeros(
        #     len(batch_input),
        #     np.shape(batch_input[0])[0],
        #     # max(length_of_sequences),
        #     ml,
        #     dtype=torch.float,
        # )
        # batch_target_tensor = torch.ones(
        #     len(batch_input),
        #     # max(length_of_sequences),
        #     ml,
        #     dtype=torch.long
        # ) * (-100)
        # mask = torch.zeros(
        #     len(batch_input),
        #     self.num_classes,
        #     # max(length_of_sequences),
        #     ml,
        #     dtype=torch.float,
        # )
        # print(batch_input_tensor.shape)
        # print(batch_input[0].shape)
        # for i in range(len(batch_input)):
        #     batch_input_tensor[i,  : np.shape(batch_input[i])[1]] = batch_input[i]
        #     # batch_input_tensor[i, :, : np.shape(batch_input[i])[1]] = batch_input[i]
        #     # torch.from_numpy(
        #     #     batch_input[i]
        #     # )
        #     batch_target_tensor[i, : np.shape(batch_target[i])[0]] = batch_target[i]
        #     # torch.from_numpy(
        #     #     batch_target[i]
        #     # )
        #     mask[i, :, : np.shape(batch_target[i])[0]] = torch.ones(
        #         self.num_classes, np.shape(batch_target[i])[0]
        #     )
        #
        # return batch_input_tensor, batch_target_tensor, mask

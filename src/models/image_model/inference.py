from featDataset import featDataset, imgDataset
from featModel import featModel, imgModel
import torch
import tqdm
import sys

sys.path.append("../")
from evaluater import evaluater
import numpy as np


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_dataset = featDataset(mode="test", feat_model="vgg")
    model = featModel()
    model = model.to(device)
    model.load_state_dict(torch.load("model.pth"))
    model.eval()

    for feature, label, f_name in test_dataset:
        print(f_name)
        eval = evaluater()
        labels = []
        preds = []
        for i in tqdm.tqdm(range(feature.shape[0])):
            f, l = feature[i].unsqueeze(0).to(device), label[i]
            output = model(f)
            output = torch.argmax(output, axis=1)
            # print(np.array(l))
            # exit()
            labels.extend([np.array(l)])
            preds.extend(output.cpu().detach().numpy())
        # print(labels[:5])
        # print(len(labels))
        # exit()
        # print(preds[:5])
        eval.set_data(labels, preds)
        count_labels(labels)
        # eval.print_eval(["accuracy"])


def count_labels(labels):
    num_labels = np.zeros(11)
    for l in labels:
        num_labels[l] += 1
    print(num_labels.sum())
    print(num_labels)


if __name__ == "__main__":
    main()

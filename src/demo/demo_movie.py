import sys

import cv2

sys.path.append("../../")
import argparse
import os

import pandas as pd
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import tqdm

from src.models.image_model import utils
from src.models.image_model.featModel import featModel


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file")
    parser.add_argument("output_name", defalt="sample.mp4")
    return parser.parse_args()


def main():
    args = get_args()
    # video_path = "../../data/videos/r21.mp4"
    # output_name = "../../data/demo/r21.mp4"
    video_path = os.path.join("../../data/videos/", args.input_file)
    output_name = os.path.join("../../demo", args.output_name)
    build_labeled_video(video_path, output_name)


def build_labeled_video(video_path, output_name):
    cap = cv2.VideoCapture(video_path)
    model = build_model()
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    all_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    out = cv2.VideoWriter(output_name, fourcc, fps, (W, H))

    for i in tqdm.tqdm(range(all_frames)):
        ret, frame = cap.read()
        if ret == False:
            break
        label = image_to_label(frame)
        edit_img = label_edit(frame, label)
        edit_img = cv2.cvtColor(edit_img, cv2.COLOR_RGB2BGR)
        out.write(edit_img)

    cap.release()
    cv2.destroyAllWindows()
    print("done")


def label_edit(img, label):
    label = utils.id_to_label(label[0])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.rectangle(img, (0, 1280), (750, 900), (255, 182, 193), thickness=-1)
    img = cv2.putText(
        img, label, (20, 1025), cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 0, 0), thickness=5
    )
    return img


def image_to_label(img):
    transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    input_tensor = transform(img).unsqueeze(0)
    model = build_model()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    input_tensor = input_tensor.to(device)
    output = model(input_tensor).argmax(axis=1).cpu().detach().numpy()
    return output


def build_model():
    feat_model = featModel()
    model_path = "../../models/vgg"
    model_name = "vgg_feat_06493908650124272.pth"

    base_model = models.vgg16(pretrained=True)
    feat_model.load_state_dict(torch.load(os.path.join(model_path, model_name)))
    total_model = nn.Sequential(base_model.features, feat_model)
    total_model.eval()
    return total_model


if __name__ == "__main__":
    main()

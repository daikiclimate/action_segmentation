import os
import time

import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

# should be config
source_path = "../../data/training/tmp_images/"
model = "vgg"
model = "mobilenet"


def get_video_names():
    return os.listdir("../../data/training/videos")


def get_resized_image_list(video_name):
    video_path = source_path + video_name[:-4] + "_resized/"
    # videos = sorted(os.listdir(video_path))
    videos = sorted([f for f in os.listdir(video_path) if f[-4:] == ".txt"])
    videos = [video_path + i for i in videos]
    return videos


def get_image():
    pass


def predict():
    v = get_video_names()
    transform = transforms.Compose(
        [
            # transforms.ToPILImage(),
            # transforms.Resize(255),
            # transforms.CenterCrop(224),
            transforms.ToTensor(),
            # transforms.RandomHorizontalFlip(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # model = "mobilenet"
    if model == "vgg":
        net = models.vgg16(pretrained=True)
    else:
        net = models.mobilenet_v2(pretrained=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = net.to(device)
    print(device)
    # exit()
    for vid in v:
        labels = []
        # if os.path.exists(f):
        #     continue
        # exit()
        # print("\r", vid, end = "")
        print(vid)
        t0 = time.time()
        img_list = get_resized_image_list(vid)
        feature_list = []
        for img in img_list:
            if not (os.path.exists(img) and os.path.exists(img[:-4] + ".jpg")):
                break

            with open(img, mode="r") as f:
                labels.append(f.read())
            # image process
            print("\r", img, end="")
            image = Image.open(img[:-4] + ".jpg")
            image = transform(image)
            image = image.reshape(1, image.shape[0], image.shape[1], image.shape[2])
            image = image.to(device)

            feature = net.features(image)
            feature_list.append(feature[0].cpu().detach().numpy())
            # exit()

        t1 = time.time()
        print("\n", round(t1 - t0), "sec")
        f = "../../data/training/feature_ext/" + model + "/" + vid[:-4] + ".pth"
        feat = torch.tensor(feature_list)
        # print(f)
        torch.save(feat, f)

        with open(f[:-4] + ".txt", mode="w") as f:
            f.writelines("\n".join(labels))
        continue


if __name__ == "__main__":
    # print(get_video_names())
    predict()

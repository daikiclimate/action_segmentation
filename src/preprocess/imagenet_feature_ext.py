import os
import torch
import torchvision.models as models
import torchvision.transforms as transforms

from PIL import Image

source_path = "../../data/tmp_images/"
def get_video_names():
    return os.listdir("../../data/videos")

def get_resized_image_list(video_name):
    video_path = source_path + video_name[:-4] + "_resized/"
    # videos = sorted(os.listdir(video_path))
    videos = sorted([f for f in os.listdir(video_path) if f[-4:]=='.txt'])
    videos = [video_path + i for i in videos]
    return videos


def get_image():
    pass

def predict():
    v = get_video_names()
    transform = transforms.Compose([
    # transforms.ToPILImage(),
    # transforms.Resize(255),
    # transforms.CenterCrop(224),
    transforms.ToTensor(),
    # transforms.RandomHorizontalFlip(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

    model = "vgg"
    # model = "mobilenet"
    if model == "vgg":
        net = models.vgg16(pretrained = True)
    else:
        net = models.mobilenet_v2(pretrained=True)

    labels = []
    for vid in v:
        # if os.path.exists(f):
        #     continue
        # exit()
        print("\r", vid, end = "")
        img_list = get_resized_image_list(vid)
        feature_list = []
        for img in img_list:
            if not(os.path.exists(img) and os.path.exists(img[:-4] + ".jpg")):
                break
        # for img in img_list[:5]:
            #image process
            print("\r", img, end = "")
            image =  Image.open(img[:-4]+".jpg")
            image = transform(image)
            image = image.reshape(1, image.shape[0], image.shape[1], image.shape[2])
            # print(image.shape)

            feature = net.features(image)
            feature_list.append(feature[0].detach().numpy())

        # for img in img_list:
            # if os.path.exists(img) and os.path.exists(img[:-4] + ".jpg"):
            #     break

            with open(img, mode = "r") as f:
                labels.append(f.read())
        

        f = "../../data/feature_ext/"+model+"/" + vid[:-4] + ".pth"
        feat = torch.tensor(feature_list)
        print(f)
        torch.save(feat, f)

        with open(f[:-4] + ".txt", mode = "w") as f:
            f.writelines(labels)
        continue


if __name__=="__main__":
    # print(get_video_names())
    predict()


import ffmpeg
import numpy as np
import os 
import cv2
import pandas as pd

def load_excel():
    return 

def video_to_images(video_name, video_path, save_path):
    # 入力
    stream = ffmpeg.input(video_path + video_name)
    # 出力
    r = 20

    if not os.path.exists(save_path + video_name[:-4]):
        os.makedirs(save_path + video_name[:-4])

    stream = ffmpeg.output(stream, save_path + video_name[:-4] + '/%05d.jpg',r=r ,f='image2')
    # 実行
    ffmpeg.run(stream)

def resize_images(video_name, save_path, img_size = 256):
    file = save_path + video_name[:-4]
    images = os.listdir(file)
    print(len(images))
    for img_name in images:
        im = cv2.imread(file +"/"+ img_name)
        im = cv2.resize(im, (img_size, img_size))
        save_dir_resized = save_path + video_name[:-4] + "_resized/"
        if not os.path.exists(save_dir_resized):
            os.makedirs(save_dir_resized)
        cv2.imwrite(save_dir_resized + img_name, im)
    print("image resized")

def load_annotation(annotation_path, annotation_name):
    df = pd.read_table(annotation_path + annotation_name, header = None).values
    start = df[:, 1]
    start = [round(i*100) for i in start]
    finish = df[:, 2]
    finish = [round(i*100) for i in finish]
    labels = df[:, 3]
    time = np.full(finish[-1], "..........")
    for i in range(len(start)):
        time[start[i]:finish[i]] = labels[i] 

    tmp = "opening"
    for i in range(len(time)):
        if time[i] == "..........":
            time[i] = tmp
        tmp = time[i]
    return time

def time_to_label(time, fps = 20):
    t = 1
    labels = []
    while True:
        index = int(1 + (t-1) * 100 * 1/fps)
        if index > time.shape[0]:
            print(index)
            return labels
        labels.append(time[index])
        t += 1

def labels_to_txt(labels, annotation_name, save_path):
    for i in range(1,1+len(labels)):
        print("\r",i, end = "")
        save_dir_resized = save_path + annotation_name[:-4] + "_resized/"
        with open(save_dir_resized + str(i).zfill(5) + ".txt", "w") as f:
            f.write(labels[i-1])

    
def listed_video():
    path = "../../data/"
    videos = os.listdir(path +"videos")
    return videos

def image_process():
    video_path = "../../data/videos/"
    save_path = "../../data/tmp_images/"
    video_list = listed_video()
    for video_name in video_list:
        video_to_images(video_name, video_path, save_path)
        resize_images(video_name, save_path)

def label_process():
    annotation_path = "../../data/annotation/"
    save_path = "../../data/tmp_images/"
    # annotation_name = "r1.txt"
    video_list = listed_video()
    video_list.sort()
    for video_name in video_list:
        print(video_name)
        annotation_name = video_name[:-4] + ".txt"
        time = load_annotation(annotation_path, annotation_name)
        labels = time_to_label(time)
        print(len(time), len(labels))

        labels_to_txt(labels, annotation_name, save_path)
        # exit()
    


if __name__ == "__main__":
    # image_process()
    label_process()
    exit()
    video_path = "../../data/videos/"
    video_name = "r1.mp4"
    save_path = "../../data/tmp_images/"
    # video_to_images(video_name, video_path, save_path)
    # resize_images(video_name, save_path)

    annotation_path = "../../data/"
    annotation_name = "r1.txt"
    time = load_annotation(annotation_path, annotation_name)
    time_to_label(time)
    exit()


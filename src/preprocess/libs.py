import ffmpeg
import os 
import cv2

def load_excel():
    return 

def video_to_images(video_name, video_path, save_path):
    # 入力
    stream = ffmpeg.input(video_path + video_name)
    # 出力
    r = 1

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
    



if __name__ == "__main__":
    video_path = "../../data/"
    video_name = "r1.mp4"
    save_path = "../../data/"
    # video_to_images(video_name, video_path, save_path)
    resize_images(video_name, save_path)


import pandas as pd
from models.image_model import utils
import numpy as np

def get_labels(files):
    label_count = np.zeros(11, dtype = np.int)

    for idx in range(len(files)):
        labelpath = (
            "../data/training/feature_ext/vgg/"
            + files[idx][:-4]
            + ".txt"
        )
        with open(labelpath, mode="r") as f:
            lines = f.read().splitlines()
        for line in lines:
            x = utils.label_to_id(line)
            label_count[x] += 1

    count_sum = np.sum(label_count[1:-1])
    print(count_sum)
    for i in range(1,10):
        print(f"{utils.id_to_label(i)}: [{label_count[i]}][{round(label_count[i]/count_sum,5)}]")


if __name__=="__main__":
    excel_dir="../data/training/information.xlsx"
    df = pd.read_excel(excel_dir)
    # df = df[df["mode"] == ]
    files = df.filename.values
    get_labels(files)

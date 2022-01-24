import pandas as pd
import torch


def main():
    excel_dir = "data/training/information.xlsx"
    df = pd.read_excel(excel_dir)
    print(df.head(30))
    exit()
    files = df.filename.values

    for idx in range(len(files)):

        path = "./data/training/feature_ext/" + "vgg" + "/" + files[idx][:-4] + ".pth"

        feature = torch.load(path)
        print(path, len(feature))


if __name__ == "__main__":
    main()

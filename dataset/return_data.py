from .feat_dataset import FeatDataset, ImgDataset
from .transform import return_img_transform

from torch.utils.data import DataLoader


def return_dataset(config):
    dataset_type = config.type
    if dataset_type == "feat":
        train_dataset = FeatDataset(
            mode="train", feat_model=config.model, config=config
        )
        test_dataset = FeatDataset(mode="test", feat_model=config.model, config=config)
    elif dataset_type == "img":
        transforms = return_img_transform()
        train_dataset = ImgDataset(mode="train", transform=transforms)
        test_dataset = ImgDataset(mode="test", transform=transforms)

    return train_dataset, test_dataset


def return_dataloader(config):
    import time

    train_set, test_set = return_dataset(config)
    t0 = time.time()
    for _ in train_set:
        print(f"time.time()-t0")
    exit()
    t1 = time.time()
    print(f"dataset{t1-t0}")
    train_loader = DataLoader(
        train_set,
        batch_size=1,
        shuffle=True,
        drop_last=True,
        num_workers=0,
        # pin_memory = False,
        # num_workers=8,
    )
    for _ in train_loader:
        pass
    t2 = time.time()
    print(f"loader{t2-t1}")
    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=1,
        drop_last=True,
        num_workers=8,
    )

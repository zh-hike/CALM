from torch.utils.data import DataLoader

from .dataset import HandWritten


def build(name, data_root, test_size=None, random_state=1, batch_size=100000):
    train_dataset = eval(name)(
        data_root=data_root, test_size=test_size, random_state=random_state
    )

    val_dataset = eval(name)(
        data_root=data_root, test_size=test_size, random_state=random_state, train=False
    )

    train_dl = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=True
    )
    val_dl = DataLoader(val_dataset, batch_size=batch_size)
    return train_dl, val_dl

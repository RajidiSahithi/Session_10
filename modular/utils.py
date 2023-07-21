import torch
from pytorch_lightning.trainer.supporters import TrainDataLoaderIter

class CustomTrainDataLoaderIter(Dataset):
    def inputs_labels_from_batch(self, batch):
        batch = iter(train_loader)
        images, labels = next(batch)

def get_lr(optimizer):
    """
    For tracking LR 
    """
    for param_group in optimizer.param_groups:
        return param_group['lr']


def denormalize(img):
    channel_means = (0.4914, 0.4822, 0.4465)
    channel_stdevs = (0.2023, 0.1994, 0.2010)
    img = img.astype(dtype = np.float32)

    for i in range (img.sahpe[2]):
        img[:, :, i] = img [:, :, i] * channel_stdevs [i] + channel_means [i]
    return img
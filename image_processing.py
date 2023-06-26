from datetime import time

import matplotlib.pyplot as plt
import cv2
import torch
import os
import pandas as pd
import random
import numpy as np
from torch import nn, optim, device
from torch.optim import lr_scheduler
from torchvision import transforms, datasets
from torch.utils.data import Dataset
from multiprocessing import freeze_support
from efficientnet_pytorch import EfficientNet
import time


train = pd.read_csv('input/train.csv')
# EXAMINE SAMPLE BATCH
image_size = 256
# transformations
sample_trans = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor()])


train = pd.read_csv('input/train.csv')
train.columns = ['id_code', 'diagnosis']

def seed_everything(seed=23):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed = 23
seed_everything(seed)


def prepare_image(path, sigmaX=10, do_random_crop=False):
    # import image
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # perform smart crops
    image = crop_black(image, tol=10)
    if do_random_crop:
        image = random_crop(image, size=(0.9, 1))

    # resize and color
    image = cv2.resize(image, (int(image_size), int(image_size)))
    image = cv2.addWeighted(image, 4, cv2.GaussianBlur(image, (0, 0), sigmaX), -4, 128)

    # circular crop
    image = circle_crop(image, sigmaX=sigmaX)

    # convert to tensor
    image = torch.tensor(image)
    image = image.permute(2, 1, 0)
    return image


# automatic crop of black areas
def crop_black(img, tol=7):
    if img.ndim == 2:
        mask = img > tol
        return img[np.ix_(mask.any(1), mask.any(0))]

    elif img.ndim == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img > tol
        check_shape = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))].shape[0]

        if check_shape == 0:
            return img
        else:
            img1 = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))]
            img2 = img[:, :, 1][np.ix_(mask.any(1), mask.any(0))]
            img3 = img[:, :, 2][np.ix_(mask.any(1), mask.any(0))]
            img = np.stack([img1, img2, img3], axis=-1)
            return img


# circular crop around center
def circle_crop(img, sigmaX=10):
    height, width, depth = img.shape

    largest_side = np.max((height, width))
    img = cv2.resize(img, (largest_side, largest_side))

    height, width, depth = img.shape

    x = int(width / 2)
    y = int(height / 2)
    r = np.amin((x, y))

    circle_img = np.zeros((height, width), np.uint8)
    cv2.circle(circle_img, (x, y), int(r), 1, thickness=-1)

    img = cv2.bitwise_and(img, img, mask=circle_img)
    return img


# random crop
def random_crop(img, size=(0.9, 1)):
    height, width, depth = img.shape

    cut = 1 - random.uniform(size[0], size[1])

    i = random.randint(0, int(cut * height))
    j = random.randint(0, int(cut * width))
    h = i + int((1 - cut) * height)
    w = j + int((1 - cut) * width)

    img = img[i:h, j:w, :]

    return img


# dataset
class EyeData(Dataset):

    # initialize
    def __init__(self, data, directory, transform=None):
        self.data = data
        self.directory = directory
        self.transform = transform

    # length
    def __len__(self):
        return len(self.data)

    # get items
    def __getitem__(self, idx):
        img_name = os.path.join(self.directory, self.data.loc[idx, 'id_code'] + '.png')
        image = prepare_image(img_name)
        image = self.transform(image)
        label = torch.tensor(self.data.loc[idx, 'diagnosis'])
        return {'image': image, 'label': label}


# dataset
# sample = EyeData(data=train,
#                  directory='input/train_images',
#                  transform=sample_trans)

# initialization function
def init_model(train=True):

    # training mode
    if train == True:
        # load pre-trained model
        model = EfficientNet.from_pretrained('efficientnet-b4', num_classes=5)

    # inference mode
    if train == False:

        # load pre-trained model
        model = EfficientNet.from_name('efficientnet-b4')
        model._fc = nn.Linear(model._fc.in_features, 5)

        # freeze  layers
        for param in model.parameters():
            param.requires_grad = False

    return model



if __name__ == '__main__':
    freeze_support()

    print(train.shape)
    print('-' * 15)
    print(train['diagnosis'].value_counts(normalize=True))

    batch_size = 20
    image_size = 256

    train_trans = transforms.Compose([transforms.ToPILImage(),
                                      transforms.RandomRotation((-360, 360)),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.RandomVerticalFlip(),
                                      transforms.ToTensor()
                                      ])
    train_dataset = EyeData(data=train,
                            directory='input/train_images',
                            transform=train_trans)

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=4)
    model_name = 'enet_b4'

    model = init_model()

    criterion = nn.CrossEntropyLoss()

    max_epochs = 15
    early_stop = 5

    eta = 1e-3

    step = 5
    gamma = 0.5

    optimizer = optim.Adam(model.parameters(), lr=eta)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=step, gamma=gamma)

    model = init_model()
    cuda = torch.cuda.is_available()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = model.to(device)

    trn_losses = []

    cv_start = time.time()

    for epoch in range(max_epochs):

        epoch_start = time.time()

        trn_loss = 0.0

        model.train()

        for batch_i, data in enumerate(train_loader):
            inputs = data['image']
            labels = data['label'].view(-1)
            inputs = inputs.to(device, dtype=torch.float)
            labels = labels.to(device, dtype=torch.long)
            optimizer.zero_grad()

            with torch.set_grad_enabled(True):
                preds = model(inputs)
                loss = criterion(preds, labels)
                loss.backward()
                optimizer.step()

            trn_loss += loss.item() * inputs.size(0)

        scheduler.step()

        trn_losses.append(trn_loss / len(train))

        print('- epoch {}/{} | lr = {} | trn_loss = {:.4f} | {:.2f} min'.format(
            epoch + 1, max_epochs, scheduler.get_lr()[len(scheduler.get_lr()) - 1],
            trn_loss / len(train), (time.time() - epoch_start) / 60))

        if epoch == (max_epochs - 1):
            print('Training complete. Best results: loss = {:.4f} (epoch {})'.format(
                np.min(trn_losses), np.argmin(trn_losses) + 1))
            print('')

    print('Finished in {:.2f} minutes'.format((time.time() - cv_start) / 60))

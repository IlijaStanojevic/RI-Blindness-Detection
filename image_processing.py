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
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold, StratifiedKFold
from tqdm import tqdm_notebook as tqdm
import time

train = pd.read_csv('input/train.csv')
test = pd.read_csv('input/sample_submission.csv')
image_size = 256


# transformations
sample_trans = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor()])
# test_trans = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor()])


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



def init_model(model_name, train=True,
               trn_layers=2,
               ):
    ### training
    if train == True:

        model = EfficientNet.from_pretrained('efficientnet-b4', num_classes=5)

        for child in list(model.children())[:-trn_layers]:
            for param in child.parameters():
                param.requires_grad = False

    ### inference
    if train == False:

        model = EfficientNet.from_pretrained('efficientnet-b4', num_classes=5)

        for param in model.parameters():
            param.requires_grad = False

    return model


if __name__ == '__main__':

    ################################################
    is_train = False
    ################################################

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
    test_trans = transforms.Compose([transforms.ToPILImage(),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.RandomVerticalFlip(),
                                     transforms.ToTensor()])
    train_dataset = EyeData(data=train,
                            directory='input/train_images',
                            transform=train_trans)

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=4)
    test_dataset = EyeData(data=test,
                           directory='input/test_images',
                           transform=test_trans)




    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              num_workers=4)
    model_name = 'RI'

    model = init_model(model_name, is_train, 2)

    cuda = torch.cuda.is_available()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = model.to(device)

    trn_losses = []

    if not (is_train):
        num_folds = 4
        tta_times = 4

        test_preds = np.zeros((len(test), num_folds))
        cv_start = time.time()

        for fold in range(num_folds):
            print("FOLD: ", fold+1, time.time())
            # load model and sent to GPU
            model = init_model(model_name, train=False)
            model.load_state_dict(torch.load('models/model_{}_fold{}.bin'.format(model_name, fold + 1)))
            # model.load_state_dict(torch.load('models/model_{}_fold{}.bin'.format(model_name, fold + 1)))
            model = model.to(device)
            model.eval()

            # placeholder
            fold_preds = np.zeros((len(test), 1))

            # loop through batches
            for _ in range(tta_times):
                for batch_i, data in enumerate(test_loader):
                    inputs = data['image']
                    inputs = inputs.to(device, dtype=torch.float)
                    preds = model(inputs).detach()
                    _, class_preds = preds.topk(1)
                    fold_preds[batch_i * batch_size:(batch_i + 1) * batch_size, :] += class_preds.cpu().numpy()
            fold_preds = fold_preds / tta_times

            # aggregate predictions
            test_preds[:, fold] = fold_preds.reshape(-1)



        # plot densities
        test_preds_df.plot.kde()
        test_preds = test_preds_df.mean(axis=1).values

        # set cutoffs
        coef = [0.5, 1.75, 2.25, 3.5]

        # rounding
        for i, pred in enumerate(test_preds):
            if pred < coef[0]:
                test_preds[i] = 0
            elif pred >= coef[0] and pred < coef[1]:
                test_preds[i] = 1
            elif pred >= coef[1] and pred < coef[2]:
                test_preds[i] = 2
            elif pred >= coef[2] and pred < coef[3]:
                test_preds[i] = 3
            else:
                test_preds[i] = 4
        sub = pd.read_csv('input/sample_submission.csv')
        sub['diagnosis'] = test_preds.astype('int')

        sub.to_csv('input/submission.csv', index=False)
    else:
        num_folds = 4

        # creating splits
        skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=seed)
        splits = list(skf.split(train['id_code'], train['diagnosis']))

        # placeholders
        oof_preds = np.zeros((len(train), 1))

        # timer
        cv_start = time.time()
        criterion = nn.CrossEntropyLoss()

        max_epochs = 15
        early_stop = 5
        eta = 1e-3
        step = 5
        gamma = 0.5
        for fold in range(num_folds):

            ### DATA PREPARATION

            print('=' * 30)
            print("FOLD: ", fold+1, "/", num_folds)
            print('=' * 30)

            # load splits
            data_train = train.iloc[splits[fold][0]].reset_index(drop=True)
            data_valid = train.iloc[splits[fold][1]].reset_index(drop=True)

            # datasets
            train_dataset = EyeData(data=data_train,
                                         directory='input/train_images',
                                         transform=train_trans)
            valid_dataset = EyeData(data=data_valid, directory='input/train_images', transform=train_trans)
            # data loaders
            train_loader = torch.utils.data.DataLoader(train_dataset,
                                                       batch_size=batch_size,
                                                       shuffle=True,
                                                       num_workers=4)
            valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=False,
                                                       num_workers=4)


            val_kappas = []
            val_losses = []
            trn_losses = []
            bad_epochs = 0

            if fold > 0:
                oof_preds = oof_preds_best.copy()


            model = init_model(model_name, train=True)
            model = model.to(device)

            # optimizer
            optimizer = optim.Adam(model._fc.parameters(), lr=eta)
            scheduler = lr_scheduler.StepLR(optimizer, step_size=step, gamma=gamma)

            for epoch in range(max_epochs):


                epoch_start = time.time()

                trn_loss = 0.0
                val_loss = 0.0

                fold_preds = np.zeros((len(data_valid), 1))

                ## TRAINING

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

                ## INFERENCE

                model.eval()

                for batch_i, data in enumerate(valid_loader):
                    inputs = data['image']
                    labels = data['label'].view(-1)
                    inputs = inputs.to(device, dtype=torch.float)
                    labels = labels.to(device, dtype=torch.long)

                    # compute predictions
                    with torch.set_grad_enabled(False):
                        preds = model(inputs).detach()
                        _, class_preds = preds.topk(1)
                        fold_preds[batch_i * batch_size:(batch_i + 1) * batch_size, :] = class_preds.cpu().numpy()

                    loss = criterion(preds, labels)
                    val_loss += loss.item() * inputs.size(0)

                oof_preds[splits[fold][1]] = fold_preds

                scheduler.step()

                ### EVALUATION

                fold_preds_round = fold_preds
                val_kappa = metrics.cohen_kappa_score(data_valid['diagnosis'], fold_preds_round.astype('int'),
                                                      weights='quadratic')

                val_kappas.append(val_kappa)
                val_losses.append(val_loss / len(data_valid))
                trn_losses.append(trn_loss / len(data_train))

                ### EARLY STOP

                print(
                    '- epoch {}/{} | lr = {} | trn_loss = {:.4f} | val_loss = {:.4f} | val_kappa = {:.4f} | {:.2f} min'.format(
                        epoch + 1, max_epochs, scheduler.get_last_lr()[len(scheduler.get_last_lr()) - 1],
                        trn_loss / len(data_train), val_loss / len(data_valid), val_kappa,
                        (time.time() - epoch_start) / 60))


                if epoch > 0:
                    if val_kappas[epoch] < val_kappas[epoch - bad_epochs - 1]:
                        bad_epochs += 1
                    else:
                        bad_epochs = 0

                # save model weights if improvement
                if bad_epochs == 0:
                    oof_preds_best = oof_preds.copy()
                    torch.save(model.state_dict(), 'models/model_{}_fold{}.bin'.format(model_name, fold + 1))

                # early stop
                if bad_epochs == early_stop:
                    print('Early stopping. Best results: loss = {:.4f}, kappa = {:.4f} (epoch {})'.format(
                        np.min(val_losses), val_kappas[np.argmin(val_losses)], np.argmin(val_losses) + 1))
                    print('')
                    break

                # max epochs
                if epoch == (max_epochs - 1):
                    print('Did not meet early stopping. Best results: loss = {:.4f}, kappa = {:.4f} (epoch {})'.format(
                        np.min(val_losses), val_kappas[np.argmin(val_losses)], np.argmin(val_losses) + 1))
                    print('')
                    break

        # load best predictions
        oof_preds = oof_preds_best

        # print performance
        print('')
        print('Finished in {:.2f} minutes'.format((time.time() - cv_start) / 60))
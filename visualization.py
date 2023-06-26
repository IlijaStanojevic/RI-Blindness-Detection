import numpy as np
import pandas as pd

import torch
import torchvision

from torchvision import transforms, datasets
from torch.utils.data import Dataset

from PIL import Image, ImageFile
from tqdm import tqdm_notebook as tqdm
import random
import time
import sys
import os
import math
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
ImageFile.LOAD_TRUNCATED_IMAGES = True
pd.set_option('display.max_columns', None)

warnings.filterwarnings('ignore')

# import data
train = pd.read_csv('input/train.csv')
test = pd.read_csv('input/sample_submission.csv')
print('Train Size = {}'.format(len(train)))
print('Public Test Size = {}'.format(len(test)))
# plot
fig = plt.figure(figsize=(15, 5))
plt.hist(train['diagnosis'])
plt.title('Class Distribution')
plt.ylabel('Number of examples')
plt.xlabel('Diagnosis')
plt.show()


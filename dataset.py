import random
import re
from collections import defaultdict
from glob import glob
from os.path import join

import numpy as np
import pandas as pd
import torch
import torchvision.transforms.functional as TF
from PIL import Image, ImageFilter
from torch.utils.data import Dataset
from torchvision import transforms


def file_iter(dataroot):
    for file in glob(join(dataroot, '*', '*', '*')):
        yield file


def file_match_iter(dataroot):
    pattern = re.compile(r'(?P<file>.*(?P<fold>[a-zA-Z0-9_]+)/'
                         r'(?P<class>hem|all)/'
                         r'UID_(?P<subject>H?\d+)_(?P<image>\d+)_(?P<cell>\d+)_(all|hem).bmp)')
    for file in file_iter(dataroot):
        match = pattern.match(file)
        if match is not None:
            yield file, match


def to_dataframe(dataroot):
    data = defaultdict(list)
    keys = ['file', 'fold', 'subject', 'class', 'image', 'cell']

    # Load data from the three training folds
    for file, match in file_match_iter(dataroot):
        for key in keys:
            data[key].append(match.group(key))

    # Load data from the phase2 validation set
    phase2 = pd.read_csv(join(dataroot, 'phase2.csv'), header=0, names=['file_id', 'file', 'class'])
    pattern = re.compile(r'UID_(?P<subject>H?\d+)_(?P<image>\d+)_(?P<cell>\d+)_(all|hem).bmp')
    for i, row in phase2.iterrows():
        match = pattern.match(row['file_id'])
        data['file'].append(join(dataroot, f'phase2/{i+1}.bmp'))
        data['fold'].append('3')
        data['subject'].append(match.group('subject'))
        data['class'].append('hem' if row['class'] == 0 else 'all')
        data['image'].append(match.group('image'))
        data['cell'].append(match.group('cell'))

    # Convert to dataframe
    df = pd.DataFrame(data)
    df = df.apply(pd.to_numeric, errors='ignore')
    return df


class ISBI2019(Dataset):
    def __init__(self, df, transform=None):
        super().__init__()
        self.transform = transform
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        # Convert tensors to int because pandas screws up otherwise
        index = int(index)
        file, cls = self.df.iloc[index][['file', 'class']]
        img = Image.open(file)
        cls = 0 if cls == 'hem' else 1
        if self.transform is not None:
            img = self.transform(img)
        return img, cls


def get_class_weights(df):
    class_weights = torch.FloatTensor([
        df.loc[df['class'] == 'hem']['file'].count() / len(df),
        df.loc[df['class'] == 'all']['file'].count() / len(df),
    ]).to(dtype=torch.float32)
    return class_weights


def tf_rotation_stack(x, num_rotations=8):
    xs = []
    for i in range(num_rotations):
        angle = 360 * i / num_rotations
        xrot = TF.rotate(x, angle)
        xrot = TF.to_tensor(xrot)
        xs.append(xrot)
    xs = torch.stack(xs)
    return xs


def gaussian_blur(pil_img, max_blur_radius):
    radius = np.random.uniform(0, max_blur_radius)
    return pil_img.filter(ImageFilter.GaussianBlur(radius))


TF_TRAIN = transforms.Compose([
    transforms.CenterCrop(300),
    transforms.RandomVerticalFlip(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomAffine(degrees=360, translate=(0.2, 0.2)),
    transforms.ToTensor(),
])


TF_VALID_ROT = transforms.Compose([
    transforms.CenterCrop(300),
    transforms.Lambda(tf_rotation_stack),
])

TF_VALID_NOROT = transforms.Compose([
    transforms.CenterCrop(300),
    transforms.ToTensor(),
])


def get_dataset(dataroot, folds_train=(0, 1, 2), folds_valid=(3,), tf_train=TF_TRAIN, tf_valid=TF_VALID_ROT):
    df = to_dataframe(dataroot)
    df_trainset = df.loc[df['fold'].isin(folds_train)]
    trainset = ISBI2019(df_trainset, transform=tf_train)
    class_weights = get_class_weights(df_trainset)

    if folds_valid is not None:
        df_validset = df.loc[df['fold'].isin(folds_valid)]
        validset_subjects = df_validset['subject'].values
        validset = ISBI2019(df_validset, transform=tf_valid)
        return trainset, validset, validset_subjects, class_weights
    else:
        return trainset, class_weights


def get_random_subject_split(dataroot, folds, ratio, tf_train=TF_TRAIN, tf_valid=TF_VALID_ROT):
    df = to_dataframe(dataroot)
    df = df.loc[df['fold'].isin(folds)]
    subjects = set(df['subject'].unique())

    # Split into healthy and non-healthy subjects because we want to include an even mix in both splits
    subjects_hem = set(x for x in subjects if x.startswith('H'))
    subjects_all = subjects - subjects_hem

    def random_split(whole_set):
        k = int(round(len(whole_set) * ratio))
        split1 = set(random.sample(whole_set, k))
        split2 = whole_set - split1
        return split1, split2

    hem1, hem2 = random_split(subjects_hem)
    all1, all2 = random_split(subjects_all)
    subj1 = hem1 | all1
    subj2 = hem2 | all2

    # Split dataset according to subj1 and subj2
    df1 = df.loc[df['subject'].isin(subj1)]
    df2 = df.loc[df['subject'].isin(subj2)]

    ds_train = ISBI2019(df1, transform=tf_train)
    ds_valid = ISBI2019(df2, transform=tf_valid)
    class_weights = get_class_weights(df1)

    return ds_train, ds_valid, class_weights

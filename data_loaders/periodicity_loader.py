import os
import numpy as np
import random
import torch

from util import *

MIN_DETECT_FREQ = 60
MAX_DETECT_FREQ = 800
FS = 96000
def target_transform(target):
    return 2 * target
    # freqs = np.maximum(freqs, MIN_DETECT_FREQ)
    # freqs = np.minimum(freqs, MAX_DETECT_FREQ)
    # periods = fs / freqs
    # min_detect_period = fs / MAX_DETECT_FREQ
    # max_detect_period = fs / MIN_DETECT_FREQ
    # periods /= max_detect_period
    # return periods

class AudioPeriodicityData():
    def __init__(self, input_file, target_file, input_dim, batch_size, downsample=1):
        self.train_set = AudioPeriodicityDataset(
            input_file,
            target_file,
            input_dim,
            downsample=downsample,
            target_transform=target_transform,
        )
        self.train_load = torch.utils.data.DataLoader(
            self.train_set,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=False
        )

        self.eval_set = AudioPeriodicityDataset(
            input_file,
            target_file,
            input_dim,
            downsample=downsample,
            target_transform=target_transform,
        )
        self.eval_load = torch.utils.data.DataLoader(
            self.eval_set,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=False
        )

class AudioPeriodicityDataset(torch.utils.data.Dataset):
    def __init__(self, input_file, target_file, input_dim, downsample=1, target_transform=lambda x: x):
        self.input_dim = input_dim
        self.input_data = load_signal(input_file)
        self.target_data = target_transform(load_signal(target_file))
        self.input_data = self.input_data[::downsample]
        self.target_data = self.target_data[::downsample]
        assert len(self.input_data) == len(self.target_data)

        self.n_frames = len(self.input_data)
        self.n_examples = self.n_frames - self.input_dim
        print("n frames:", self.n_frames)
        print("n examples:", self.n_examples)

    def __len__(self):
        return self.n_examples

    def __getitem__(self, idx):
        input = self.input_data[idx:idx + self.input_dim].reshape((1, self.input_dim))
        target = self.target_data[idx]
        return input, target

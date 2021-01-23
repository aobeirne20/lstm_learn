import os
import numpy as np
import random
import torch

from util import *

class AudioPeriodicityData:
    def __init__(self, input_file, target_file, input_dim, batch_size):
        train_set = AudioPeriodicityDataset(os.path.join(input_file, target_file, input_dim, output_dim)
        self.train_load = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False)

        # val_set = AudioPeriodicityDataset(os.path.join(AudioData.DATA_DIR, "val"), input_dim, output_dim)
        # self.val_load = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False)
        #
        # test_set = AudioPeriodicityDataset(os.path.join(AudioData.DATA_DIR, "test"), input_dim, output_dim)
        # self.test_load = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=True, num_workers=0, pin_memory=False)

class AudioPeriodicityDataset(torch.utils.data.Dataset):
    def __init__(self, input_file, target_file, input_dim):
        self.input_dim = input_dim
        self.input_data = load_signal(input_file)
        self.target_data = load_signal(target_file)
        assert len(self.input_data) == len(self.target_data)

        self.n_frames = len(self.input_data)
        self.n_examples = self.n_frames - self.input_dim
        print("n frames:", self.n_frames)
        print("n examples:", self.n_examples)

    def __len__(self):
        return self.n_examples

    def __getitem__(self, idx):
        return self.input_data[idx:idx + self.input_dim], self.target_data[idx]

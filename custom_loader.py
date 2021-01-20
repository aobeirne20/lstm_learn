import os
import numpy as np
import random
import torch


class AudioSnipDataset(torch.utils.data.Dataset):
    def __init__(self, directory, seg_length, ans_length):
        data_sets = []
        temp_segments = []
        temp_answers = []

        for file in os.listdir(directory):
            if file.endswith(".signal"):
                data_sets.append(np.fromfile(os.path.join(directory, file), dtype=float))

        for data_array in data_sets:
            for index in range(0, data_array.size - 1 - seg_length - ans_length):
                temp_segments.append(data_array[index:index + seg_length])
                temp_answers.append(data_array[index + seg_length + 1:index + seg_length + ans_length + 1])

        x_data_np = np.vstack(temp_segments)
        y_data_np = np.vstack(temp_answers)

        self.x_data = torch.tensor(x_data_np, dtype=torch.float32).to("cuda:0")
        self.y_data = torch.tensor(y_data_np, dtype=torch.float32).to("cuda:0")

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        segment = self.x_data[idx, :]
        answer = self.y_data[idx]
        sample = \
            {'input': segment, 'output': answer}
        return sample




import os
import torch
import torchvision.transforms as transforms
import time
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from custom_loader import AudioSnipDataset

# from models.simple_fc_net import *
from models.lstm import *
# from models.dilation_conv_net import *

torch.manual_seed(0)

if __name__ == '__main__':
    #Also the number of frames in the input
    input_dim = 5000
    #Also the number of frames in the output
    output_dim = 50

    batch_size = 60

    embedding_dim = 5000
    hidden_dim = 512
    n_layers = 2
    drop_prob = 0.5

    Audio = AudioData(input_dim, output_dim, batch_size)

    modelx = NetWrapper(input_dim, output_dim, hidden_dim, embedding_dim, n_layers, drop_prob)

    modelx.learn(Audio.train_load, Audio.val_load, batch_size, learning_rate=0.005, n_epochs=5)

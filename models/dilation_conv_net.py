import sys
import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm

sys.path.append("..")
from data_loaders.periodicity_loader import *

class DilationConvNet(torch.nn.Module):
    def __init__(self, input_dim=1024, hidden_dim=64, output_dim=1):
        super(DilationConvNet, self).__init__()
        assert input_dim % 4 == 0

        self.conv1 = torch.nn.Conv1d(
            in_channels=1,
            out_channels=2,
            kernel_size=4,
            stride=4,
            # padding=0,
            # dilation=1,
            # groups=1
        )
        self.conv2 = torch.nn.Conv1d(
            in_channels=2,
            out_channels=4,
            kernel_size=4,
            stride=4,
        )

        self.fc1 = torch.nn.Linear(input_dim // 4, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        batch_size = x.size(0)
        x = self.conv1(x)
        x = self.conv2(x)
        # x = x.flatten()
        # print(x.size())
        x = self.fc1(torch.reshape(x, (batch_size, -1)))
        x = self.fc2(x)
        return x

class DilationConvNetWrapper():
    def __init__(self, input_dim, hidden_dim, output_dim):
        self.net = DilationConvNet(input_dim, hidden_dim, output_dim)
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.net.to(self.device)

    def learn(self, train_load, eval_load=None, n_epochs=1, batch_size=60, learning_rate=0.005):
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(self.net.parameters(), learning_rate)
        print(f'\nLTSM training with {n_epochs} epochs:')
        self.net.train()

        for epoch in range(n_epochs):
            for batch_idx, (inputs, labels) in tqdm.tqdm(enumerate(train_load), total=len(train_load)):
                # print(inputs.size())
                inputs, labels = inputs.float(), labels.float()
                # print(inputs.size())
                # print(labels.size())
                # print(input)
                self.net.zero_grad()
                output = self.net(inputs)
                loss = criterion(output.squeeze(), labels.float())
                loss.backward()
                optimizer.step()
        outputs, targets, eval_losses = self.evaluate(eval_load)
        # plt.plot(outputs); plt.plot(targets); plt.show()
        print("Loss: {:.6f}...".format(loss.item()), "Val Loss: {:.6f}".format(np.mean(eval_losses)))

    def evaluate(self, eval_load, eval_frames=None):
        self.net.eval()
        criterion = torch.nn.MSELoss()
        eval_frames = len(eval_load) - 1 if eval_frames is None else eval_frames
        output_batches, target_batches, eval_loss_batches = [], [], []
        for i, (input, target) in tqdm.tqdm(enumerate(eval_load), total=eval_frames):
            input, target = input.float().to(self.device), target.float().to(self.device)
            output = self.net(input)
            eval_loss = criterion(output.squeeze(), target.float())
            output_batches.append(output.detach().numpy()[:, 0])
            target_batches.append(target.detach().numpy())
            eval_loss_batches.append(eval_loss.detach().numpy())
            if i >= eval_frames:
                break
        self.net.train()
        return np.hstack(output_batches), np.hstack(target_batches), np.hstack(eval_loss_batches)

    def save(self, model_name):
        torch.save(self.net, f'model_states/{model_name}.pth')

    def load(self, model_name):
        path = f'model_states/{model_name}.pth'
        if torch.cuda.is_available():
            self.net = torch.load(path, 'cuda:0')
        else:
            self.net = torch.load(path, 'cpu')

if __name__ == '__main__':
    input_file = os.path.join(os.getcwd(), "..", "new_data", "train", "inputs", "jackson_z.npy")
    # target_file = os.path.join(os.getcwd(), "..", "new_data", "train", "periods", "jackson_z.npy")
    target_file = os.path.join(os.getcwd(), "..", "new_data", "train", "inputs", "jackson_z.npy")
    input_dim = 256
    batch_size = 50

    data = AudioPeriodicityData(input_file, target_file, input_dim, batch_size, downsample=8)
    model = DilationConvNetWrapper(input_dim, hidden_dim=64, output_dim=1)

    # plt.plot(data.train_set.input_data); plt.plot(data.train_set.target_data); plt.show()

    # outputs, _ = model.evaluate(data.val_load, eval_frames=1000)
    # print(outputs)
    # print(model.net(torch.rand((2, 1, 1024))))

    # o, t, e = model.evaluate(data.eval_load)
    # plt.plot(o); plt.plot(t); plt.show()
    o, t, e = model.evaluate(data.eval_load)
    plt.plot(o); plt.plot(t); plt.show()
    model.learn(data.train_load, data.eval_load, n_epochs=1, batch_size=batch_size, learning_rate=0.005)
    # model.load("dilated_conv_downsample=8.pth")
    o, t, e = model.evaluate(data.eval_load)
    plt.plot(o); plt.plot(t); plt.show()

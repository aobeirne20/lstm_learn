import sys
import torch

sys.path.append("..")
from data_loaders.prediction_loader import *

class DilationConvNet(torch.nn.Module):
    def __init__(self):
        super(DilationConvNet, self).__init__()

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

        self.fc1 = torch.nn.Linear(256, 64)
        self.fc2 = torch.nn.Linear(64, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.fc1(x.flatten())
        x = self.fc2(x)
        return x

class DilationConvNetWrapper():
    def __init__(self):
        self.net = DilationConvNet()
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.net.to(self.device)

    def learn(self, train_load, val_load, n_epochs=1, batch_size=60, learning_rate=0.005):
        criterion = torch.nn.BCELoss()
        optimizer = torch.optim.Adam(self.net.parameters(), learning_rate)
        print(f'\nLTSM training with {n_epochs} epochs:')
        self.net.train()

        for epoch in range(n_epochs):
            for batch_idx, (inputs, labels) in enumerate(train_load):
                model.net.zero_grad()
                output = model.net(inputs)
                loss = criterion(output.squeeze(), labels.float())
                loss.backward()
                optimizer.step()

                if batch_idx % 10 == 0:
                    self.net.eval()
                    mean_val_loss = self.evaluate(val_load)
                    print("Epoch: {}/{}...".format(epoch + 1, n_epochs),
                          "Step: {}...".format(batch_idx),
                          "Loss: {:.6f}...".format(loss.item()),
                          "Val Loss: {:.6f}".format(mean_val_loss))
                    self.net.train()

    def evaluate(self, val_load):
        val_losses = []
        for input, target in val_load:
            input, target = input.to(self.device), target.to(self.device)
            output = self.net(x)
            val_loss = criterion(output.squeeze(), target.float())
            val_losses.append(val_loss.item())

        return np.mean(val_losses)

if __name__ == '__main__':
    input_file = os.path.join(os.getcwd(), "new_input", "train", "signalFiles")
    target_file = os.path.join(os.getcwd(), "new_input", "train", "periods")
    input_dim = 1024
    batch_size = 64

    Audio = AudioPeriodicityData(input_file, target_file, input_dim, batch_size)

    modelx = DilationConvNetWrapper()

    modelx.learn(Audio.train_load, Audio.val_load, batch_size, learning_rate=0.005, n_epochs=5)

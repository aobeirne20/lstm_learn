import torch

class Net(torch.nn.Module):
    def __init__(self, input_dim=1024, output_dim=1):
        super(DilationConvNet, self).__init__()
        self.input_dim = output_dim
        self.output_dim = output_dim

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
        # self.conv3 = torch.nn.Conv1d(
        #     in_channels=self.conv2_dim,
        #     out_channels=self.conv3_dim,
        #     kernel_size=2,
        #     stride=1,
        #     padding=0,
        #     dilation=1,
        #     groups=1
        # )
        #
        self.fc1 = torch.nn.Linear(256, 64)
        self.fc2 = torch.nn.Linear(64, 1)
        # self.fc2 = torch.nn.Linear(self.conv3_dim, self.output_dim)
        # self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.fc1(x.flatten())
        x = self.fc2(x)
        return x

net = Net()
# (batch, channel, sample)
x = torch.rand(1, 1, 1024)
print(x.size())
print(net(x).size())

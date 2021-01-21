import torch
import torchvision.transforms as transforms
import time
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from custom_loader import AudioSnipDataset

torch.manual_seed(0)


class LSTM(torch.nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, embedding_dim, n_layers, drop_prob):
        super(LSTM, self).__init__()
        self.output_dim = output_dim
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim

        self.lstm = torch.nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            dropout=drop_prob,
            batch_first=True)

        self.embedding = torch.nn.Embedding(input_dim, embedding_dim)
        self.dropout = torch.nn.Dropout(drop_prob)
        self.fc = torch.nn.Linear(hidden_dim, output_dim)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x, hidden):
        batch_size = x.size(0)
        x = x.long()
        embeds = self.embedding(x)
        lstm_out, hidden = self.lstm(embeds, hidden)
        lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)

        out = self.dropout(lstm_out)
        out = self.fc(out)
        out = self.sigmoid(out)

        out = out.view(64, -1)
        out = out[:, -1]
        return out, hidden

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to("cuda:0"),
                  weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to("cuda:0"))
        return hidden


class AudioData:
    def __init__(self, input_dim, output_dim, batch_size):
        train_set = AudioSnipDataset("C:/Users/augus/PycharmProjects/lstm_learn/new_data/train/signalFiles", input_dim, output_dim, type="train")
        self.train_load = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False)

        val_set = AudioSnipDataset("C:/Users/augus/PycharmProjects/lstm_learn/new_data/train/signalFiles", input_dim, output_dim, type="val")
        self.val_load = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False)

        test_set = AudioSnipDataset("C:/Users/augus/PycharmProjects/lstm_learn/new_data/train/signalFiles", input_dim, output_dim, type="test")
        self.test_load = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=True, num_workers=0, pin_memory=False)


class NetWrapper:
    def __init__(self, input_dim, output_dim, hidden_dim, embedding_dim, n_layers, drop_prob):
        self.net = LSTM(input_dim, output_dim, hidden_dim, embedding_dim, n_layers, drop_prob)
        if torch.cuda.is_available():
            self.net.to('cuda:0')

    def learn(self, train_load, val_load, batch_size, learning_rate, n_epochs):
        criterion = torch.nn.BCELoss()
        optimizer = torch.optim.Adam(self.net.parameters(), learning_rate)
        print(f'\nLTSM training with {n_epochs} epochs:')

        self.net.train()
        clip = 5

        for epoch in range(n_epochs):
            s_time = time.time()
            loss_total = 0
            hidden = model.net.init_hidden(batch_size)
            for batch_idx, (inputs, labels) in enumerate(train_load):
                hidden = tuple([e.data for e in hidden])
                model.net.zero_grad()
                output, hidden = model.net(inputs, hidden)
                loss = criterion(output.squeeze(), labels.float())
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                optimizer.step()

                if batch_idx % 10 == 0:
                    val_h = model.init_hidden(batch_size)
                    val_losses = []
                    model.eval()
                    for inp, lab in val_load:
                        val_h = tuple([each.data for each in val_h])
                        inp, lab = inp.to('cuda:0'), lab.to('cuda:0')
                        out, val_h = model(inp, val_h)
                        val_loss = criterion(out.squeeze(), lab.float())
                        val_losses.append(val_loss.item())

                    model.train()
                    print("Epoch: {}/{}...".format(epoch + 1, n_epochs),
                          "Step: {}...".format(batch_idx),
                          "Loss: {:.6f}...".format(loss.item()),
                          "Val Loss: {:.6f}".format(np.mean(val_losses)))
                    if np.mean(val_losses) <= valid_loss_min:
                        torch.save(model.state_dict(), './state_dict.pt')
                        print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min,
                                                                                                        np.mean(
                                                                                                            val_losses)))
                        valid_loss_min = np.mean(val_losses)










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

    model = NetWrapper(input_dim, output_dim, hidden_dim, embedding_dim, n_layers, drop_prob)

    model.learn(Audio.train_load, Audio.val_load, batch_size, learning_rate=0.005, n_epochs=5)





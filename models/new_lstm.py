import torch
import numpy as np
import time


class AudioLSTM(torch.nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, n_layers, device):
        super(AudioLSTM, self).__init__()
        self.hidden_size = hidden_dim
        self.n_layers = n_layers
        self.lstm = torch.nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True)
        self.fc = torch.nn.Linear(hidden_dim, output_dim)
        self.device = device

    def forward(self, x, hs):
        #print("LSTM (Original) Input:")
        #print(x.size())
        out, hs = self.lstm(x, hs)
        #print("LSTM (Middle) Output")
        #print(out.size())
        #out = out.reshape(-1, self.hidden_size)
        #print("Linear (Middle) Input")
        #print(out.size())
        out = self.fc(out)
        #print("Linear (End) Output")
        #print(out.size())
        #print()
        return out, hs

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = (weight.new(self.n_layers, batch_size, self.hidden_size).zero_().to(self.device),
                  weight.new(self.n_layers, batch_size, self.hidden_size).zero_().to(self.device))
        return hidden


class LSTMWrapper:
    def __init__(self, input_dim, output_dim, hidden_dim, n_layers, batch_size):
        self.batch_size = batch_size
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.net_lstm = AudioLSTM(input_dim, output_dim, hidden_dim, n_layers, self.device)
        self.net_lstm.to(self.device)


    def learn(self, train_load, validation_load, learning_rate, n_epochs, print_every):

        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(self.net_lstm.parameters(), lr=learning_rate)

        train_loss = []
        validation_loss = []

        for epoch in range(n_epochs):
            s_time = time.time()

            hidden_state = self.net_lstm.init_hidden(self.batch_size)
            total_loss = 0

            for batch_idx, (inputs, labels) in enumerate(train_load):
                output, hidden_state = self.net_lstm(inputs, hidden_state)
                hidden_state = tuple([h.data for h in hidden_state])

                loss = criterion(output, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

                leng = len(train_load) * self.batch_size
                n_blocks = int(batch_idx // (leng / self.batch_size / 20))
                space = " "
                bar = u'\u2588'
                if epoch < 9:
                    print(
                        f'\rEpoch {epoch + 1}  |{bar * n_blocks}{space * (20 - n_blocks)}| {batch_idx * self.batch_size}/{leng}',
                        end='')
                else:
                    print(
                        f'\rEpoch {epoch + 1} |{bar * n_blocks}{space * (20 - n_blocks)}| {batch_idx * self.batch_size}/{leng}',
                        end='')
            if epoch < 9:
                print(f'\rEpoch {epoch + 1}  |{bar * 20}| {leng}/{leng}', end='')
                print(f'   {(time.time() - s_time):.2f}s  Avg Loss: {(total_loss / (leng / self.batch_size)):.4f}')
            else:
                print(f'\rEpoch {epoch + 1} |{bar * 20}| {leng}/{leng}', end='')
                print(f'   {(time.time() - s_time):.2f}s  Avg Loss: {(total_loss / (leng / self.batch_size)):.4f}')

            if validation_load is not None:
                for batch_idx, (val_x, val_y) in enumerate(validation_load):
                    self.net_lstm.eval()
                    preds, _ = self.net_lstm(val_x, hidden_state)
                    v_loss = criterion(preds, val_y)
                    validation_loss.append(v_loss.item())

                    self.net_lstm.train()


        train_loss.append(np.mean(total_loss))
        self.hidden_state = hidden_state

    def test(self, test_load):
        print(self.hidden_state[0].size())
        print(self.hidden_state[1].size())
        hidden_state = (torch.unsqueeze(self.hidden_state[0][:, 0, :], 1).contiguous(), torch.unsqueeze(self.hidden_state[1][:, 0, :], 1).contiguous())
        print(hidden_state[0].size())
        print(hidden_state[1].size())
        prediction, _ = self.net_lstm(test_load, hidden_state)
        return prediction








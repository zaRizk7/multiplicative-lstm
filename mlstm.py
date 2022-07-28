from typing import *

from torch import Tensor, nn, sigmoid, tanh
from typer import Option


class LSTMCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, bias: bool = True) -> None:
        super().__init__()
        self.linear_hi = nn.Linear(input_size, hidden_size, bias)
        self.linear_hh = nn.Linear(hidden_size, hidden_size, bias)
        self.linear_ii = nn.Linear(input_size, hidden_size, bias)
        self.linear_ih = nn.Linear(hidden_size, hidden_size, bias)
        self.linear_oi = nn.Linear(input_size, hidden_size, bias)
        self.linear_oh = nn.Linear(hidden_size, hidden_size, bias)
        self.linear_fi = nn.Linear(input_size, hidden_size, bias)
        self.linear_fh = nn.Linear(hidden_size, hidden_size, bias)

    def forward(self, inputs: Tensor, hidden_state: Tuple[Tensor, Tensor]):
        h_t, c_t = hidden_state
        h_t = self.linear_hi(inputs) + self.linear_hh(h_t)
        i_t = sigmoid(self.linear_ii(inputs) + self.linear_ih(h_t))
        o_t = sigmoid(self.linear_oi(inputs) + self.linear_oh(h_t))
        f_t = sigmoid(self.linear_fi(inputs) + self.linear_fh(h_t))
        c_t = f_t * c_t + i_t * tanh(h_t)
        h_t = tanh(c_t) * o_t
        return h_t, c_t


class mLSTMCell(LSTMCell):
    def __init__(self, input_size, hidden_size, bias=True) -> None:
        super().__init__(input_size, hidden_size, bias)
        self.linear_mi = nn.Linear(input_size, hidden_size, bias)
        self.linear_mh = nn.Linear(hidden_size, hidden_size, bias)

    def forward(self, inputs, hidden_state):
        h_t, c_t = hidden_state
        m_t = self.linear_mi(inputs) * self.linear_mh(h_t)
        return super().forward(inputs, (m_t, c_t))


class mLSTM(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        bias: bool = True,
        batch_first: bool = False,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_first = batch_first
        layers = [mLSTMCell(input_size, hidden_size, bias)]
        for _ in range(num_layers - 1):
            layers.append(mLSTMCell(hidden_size, hidden_size))
        self.layers = nn.ModuleList(layers)

    def forward(
        self, inputs: Tensor, hidden_state: Optional[Tuple[Tensor, Tensor]] = None
    ):
        size = inputs.size()
        rank = len(size)
        if rank == 2:
            h_t = torch.zeros(self.hidden_size)
            c_t = torch.zeros(self.hidden_size)
        elif rank == 3:
            h_t = torch.zeros(size[0], self.hidden_size)
            c_t = torch.zeros(size[0], self.hidden_size)
            if self.batch_first:
                inputs = inputs.swapaxes(0, 1)
        else:
            raise ValueError("Tensor's rank must be either 2 or 3!")

        h, c = [], []
        for layer in self.layers:
            outputs = []
            for x_t in inputs:
                h_t, c_t = layer(x_t, (h_t, c_t))
                outputs.append(h_t)
            inputs = torch.stack(outputs)
            h.append(h_t)
            c.append(c_t)
        h = torch.stack(h)
        c = torch.stack(c)

        if self.batch_first and rank == 3:
            inputs = inputs.swapaxes(0, 1)

        return inputs, (h, c)


if __name__ == "__main__":
    import torch

    torch.manual_seed(2022)

    mlstm = mLSTM(64, 128, 4, True, True)
    inputs = torch.rand(32, 16, 64)
    outputs, (h_t, c_t) = mlstm(inputs)
    print(outputs.shape, h_t.shape, c_t.shape)

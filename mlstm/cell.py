from typing import Tuple

from torch import Tensor, nn, sigmoid, tanh


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
        h_t, c_t = h_t.detach(), c_t.detach()
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

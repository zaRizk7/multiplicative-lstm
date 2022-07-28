from typing import Optional, Tuple

from torch import Tensor, nn

from cell import mLSTMCell


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
        if rank == 3 and self.batch_first:
            inputs = inputs.swapaxes(0, 1)
        elif not rank in [2, 3]:
            raise ValueError(
                "Tensor must be either 2D (L, H_in) or 3D (L, N, H_in | N, L, H_in)!"
            )

        if not hidden_state:
            if rank == 2:
                hidden_state = (size[0], self.hidden_size)
            else:
                hidden_state = self.hidden_size
            hidden_state = [torch.zeros(hidden_state)] * 2

        h, c = [], []
        for layer in self.layers:
            outputs = []
            for x_t in inputs:
                hidden_state = layer(x_t, hidden_state)
                h_t, c_t = hidden_state
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
    print(outputs, h_t, c_t)

import torch
import numpy as np
from collections import Counter


def positional_encoding(n_positions: int, hidden_dim: int) -> torch.Tensor:
    def calc_angles(pos, i):
        rates = 1 / np.power(10000, (2*(i // 2)) / np.float32(hidden_dim))
        return pos * rates

    rads = calc_angles(np.arange(n_positions)[:, np.newaxis], np.arange(hidden_dim)[np.newaxis, :])

    rads[:, 0::2] = np.sin(rads[:, 0::2])
    rads[:, 1::2] = np.cos(rads[:, 1::2])

    pos_enc = rads[np.newaxis, ...]
    pos_enc = torch.tensor(pos_enc, dtype=torch.float32, requires_grad=False)
    return pos_enc


def subsequent_mask(size: int) -> torch.Tensor:
    """
    from Harvard NLP
    The Annotated Transformer

    http://nlp.seas.harvard.edu/2018/04/03/attention.html#batches-and-masking

    :param size: int
    :return: torch.Tensor
    """
    attn_shape = (size, size)
    mask = np.triu(np.ones(attn_shape), k=1).astype("float32")
    mask = torch.from_numpy(mask) == 0
    return mask.float()


def silu(x: torch.Tensor) -> torch.Tensor:
    output = x * torch.sigmoid(x)
    return output


class FixedSlidingWindow:
    def __init__(self, window_size: int, overlap_rate=0.5) -> None:
        """
        Fixed sliding window.
        >>> import numpy as np
        >>> from Typhoon.utils.functions import FixedSlidingWindow
        >>> x = np.random.randn(1024, 23)
        >>> y = np.random.randint(0, 9, 1024)
        >>> sw = FixedSlidingWindow(256, overlap_rate=0.5)
        >>> x, y = sw(x, y)
        >>> x.shape     # [6, 256, 23]
        >>> y.shape     # [6, ]

        :param window_size: int
        :param overlap_rate: float
        """
        self.window_size = window_size
        assert 0.0 < overlap_rate <= 1.0
        self.overlap_rate = overlap_rate
        self.overlap = int(window_size * overlap_rate)

    def transform(self, x: np.array) -> np.array:
        seq_len = x.shape[0]
        assert seq_len > self.window_size
        data = [x[i:i + self.window_size] for i in range(0, seq_len - self.window_size, self.overlap)]

        data = np.stack(data, 0)
        return data

    @staticmethod
    def clean(labels: np.array) -> np.array:
        tmp = []
        for l in labels:
            window_size = len(l)
            c = Counter(l)
            common = c.most_common()
            values = list(c.values())
            if common[0][0] == 0 and values[0] == window_size // 2:
                label = common[1][0]
            else:
                label = common[0][0]

            tmp.append(label)

        return np.array(tmp)

    def __call__(self, x: np.array, y: np.array) -> (np.array, np.array):
        data = self.transform(x)
        label = self.transform(y)
        label = self.clean(label)
        return data, label


class ScheduledOptimizer:
    """
    Reference: jadore801120/attention-is-all-you-need-pytorch
    https://github.com/jadore801120/attention-is-all-you-need-pytorch/blob/master/transformer/Optim.py
    """
    def __init__(self, optimizer, d_model: int, warm_up: int) -> None:
        self._optimizer = optimizer
        self.warm_up = warm_up
        self.n_current_steps = 0
        self.init_lr = np.power(d_model, -0.5)

    def step(self) -> None:
        self._update_learning_rate()
        self._optimizer.step()

    def zero_grad(self) -> None:
        self._optimizer.zero_grad()

    def _get_lr_scale(self) -> np.array:
        return np.min([
            np.power(self.n_current_steps, -0.5),
            np.power(self.warm_up, -1.5) * self.n_current_steps
        ])

    def get_lr(self):
        lr = self.init_lr * self._get_lr_scale()
        return lr

    def _update_learning_rate(self):
        self.n_current_steps += 1
        lr = self.get_lr()

        for param_group in self._optimizer.param_groups:
            param_group["lr"] = lr

    def state_dict(self):
        return self._optimizer.state_dict()

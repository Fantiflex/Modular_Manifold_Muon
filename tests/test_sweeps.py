import random

import numpy as np
import torch

from experiments.sweeps import set_seed


def test_set_seed_is_reproducible():
    set_seed(42)

    python_value_1 = random.random()
    numpy_value_1 = np.random.rand()
    torch_value_1 = torch.rand(1)

    set_seed(42)

    python_value_2 = random.random()
    numpy_value_2 = np.random.rand()
    torch_value_2 = torch.rand(1)

    assert python_value_1 == python_value_2
    assert numpy_value_1 == numpy_value_2
    assert torch.allclose(
        torch_value_1,
        torch_value_2,
    )
from src.training import linear_decay_lr


def test_linear_decay_lr_start():
    lr = linear_decay_lr(
        initial_lr=0.1,
        step=0,
        total_steps=100,
    )

    assert lr == 0.1


def test_linear_decay_lr_halfway():
    lr = linear_decay_lr(
        initial_lr=0.1,
        step=50,
        total_steps=100,
    )

    assert abs(lr - 0.05) < 1e-12


def test_linear_decay_lr_end():
    lr = linear_decay_lr(
        initial_lr=0.1,
        step=100,
        total_steps=100,
    )

    assert abs(lr) < 1e-12
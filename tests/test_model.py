from tinygrad import Tensor
from tinygrad.nn.optim import SGD
from tinygrad.nn.state import get_parameters

from tinygpt.model import SLP, MLP


def test__slp__fits__or_gate():
    inputs = Tensor([[0, 0], [0, 1], [1, 0], [1, 0]])
    labels = Tensor([[0], [1], [1], [1]])

    model = SLP(in_features=2, out_features=1)
    outputs = model(inputs)
    assert not outputs.isclose(labels).all().numpy()

    params = get_parameters(model)
    optimizer = SGD(params, lr=3)
    epochs = 1000

    print(f"model: {model.__class__}")
    print(f"n_params: {sum([t.numel() for t in params])}")
    with Tensor.train():
        for step in range(epochs):
            outputs = model(inputs)
            loss = outputs.binary_crossentropy(labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if step % (epochs / 10) == 0:
                print(f"step ({step} / {epochs}): loss: {loss.numpy()}")

    outputs = model(inputs)
    print(f"outputs:\n{outputs.numpy()}")

    assert outputs.isclose(labels, atol=1e-2).all().numpy()


def test__mlp__fits__or_gate():
    inputs = Tensor([[0, 0], [0, 1], [1, 0], [1, 0]])
    labels = Tensor([[0], [1], [1], [1]])

    model = MLP(in_features=2, out_features=1)
    outputs = model(inputs)
    assert not outputs.isclose(labels).all().numpy()

    params = get_parameters(model)
    optimizer = SGD(params, lr=3)
    epochs = 1500

    print(f"model: {model.__class__}")
    print(f"n_params: {sum([t.numel() for t in params])}")
    with Tensor.train():
        for step in range(epochs):
            outputs = model(inputs)
            loss = outputs.binary_crossentropy(labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if step % (epochs / 10) == 0:
                print(f"step ({step} / {epochs}): loss: {loss.numpy()}")

    outputs = model(inputs)
    print(f"outputs:\n{outputs.numpy()}")

    assert outputs.isclose(labels, atol=1e-3).all().numpy()

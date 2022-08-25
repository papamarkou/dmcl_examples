from pathlib import Path

data_root = Path(__file__).parent.parent.joinpath('data')

data_paths = {name: data_root.joinpath(name) for name in (
    'FashionMNIST',
    'hawks',
    'MNIST',
    'noisy_xor',
    'penguins',
    'pima'
)}

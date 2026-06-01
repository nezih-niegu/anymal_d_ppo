"""
Smoke tests — CPU-only, no MuJoCo, no GPU required.
"""
import yaml


def test_yaml_importable():
    assert yaml.__version__


def test_torch_importable():
    import torch
    assert torch is not None


def test_numpy_importable():
    import numpy as np
    assert np.__version__


def test_yaml_parse_sample():
    sample = """
    training:
      learning_rate: 3e-4
      batch_size: 64
      device: cpu
    """
    config = yaml.safe_load(sample)
    assert config["training"]["learning_rate"] == 3e-4
    assert config["training"]["device"] == "cpu"

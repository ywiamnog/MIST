"""Tests for SwinUNetr"""
from typing import Dict
import pytest
import torch

from mist.models.swin_unetr.swin_unetr import SwinUNETR

def create_valid_params(
    use_deep_supervision: bool=False,
    num_deep_supervision_heads: int=1
) -> Dict:
    params = {
        "patch_size": [96, 96, 96],
        "in_channels": 1,
        "out_channels": 2,
        "deep_supervision": use_deep_supervision,
        "deep_supr_num": num_deep_supervision_heads,
        "depths": [2, 2, 2, 2],
        "num_heads": [3, 6, 12, 24],
        "feature_size": 24,
        "norm_name": 'instance',
        "dropout_rate": 0.0,
        "attn_drop_rate": 0.0,
        "dropout_path_rate": 0.0,
        "normalize": True,
        "use_checkpoint": False,
        "spatial_dims": 3,
        "downsample": 'merging',
        "use_v2": False
    }
    return params


def test_forward_train_and_eval():
    params = create_valid_params(use_deep_supervision=True, num_deep_supervision_heads=1)
    model = SwinUNETR(**params)

    # Create 3D image
    x = torch.randn(1, params["in_channels"], 96, 96, 96)

    model.train()
    out_train = model(x)
    assert isinstance(out_train, dict)
    assert "prediction" in out_train
    # deep_supervision should be a list when deep supervision is enabled.
    assert out_train["deep_supervision"] is not None
    assert isinstance(out_train["deep_supervision"], list)
    # Check that the prediction is a tensor.
    assert torch.is_tensor(out_train["prediction"])

    # Test evaluation mode: output should be a tensor.
    model.eval()
    out_eval = model(x)
    assert torch.is_tensor(out_eval)


def test_negative_deep_supervision_heads():
    """Test that a ValueError is raised for negative deep supervision heads.
    """
    params = create_valid_params(
        use_deep_supervision=True, num_deep_supervision_heads=-1
    )
    with pytest.raises(ValueError) as excinfo:
        SwinUNETR(**params)
    assert (
        "num_deep_supervision_heads should be larger than 0."
        in str(excinfo.value)
    )

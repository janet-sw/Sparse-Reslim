import torch

from climate_learn.models.hub import EDM, Res_Slim_ViT_Adaptive
from climate_learn.metrics.functional import mse
from climate_learn.utils.fused_attn import FusedAttn


def _record_sequence_lengths(blocks):
    lengths = []
    handles = [
        block.register_forward_pre_hook(
            lambda _module, args: lengths.append(args[0].shape[1])
        )
        for block in blocks
    ]
    return lengths, handles


def test_deterministic_sparse_forward_backward():
    model = Res_Slim_ViT_Adaptive(
        default_vars=["a", "b"],
        img_size=(8, 16),
        in_channels=2,
        out_channels=1,
        history=1,
        superres_mag=1,
        cnn_ratio=2,
        patch_size=2,
        drop_path=0.0,
        drop_rate=0.0,
        embed_dim=32,
        depth=4,
        decoder_depth=1,
        num_heads=4,
        FusedAttn_option=FusedAttn.DEFAULT,
        num_constant_vars=0,
        keep_ratio=0.25,
        num_dense_early=1,
        num_sparse_middle=2,
    )
    lengths, handles = _record_sequence_lengths(model.blocks)
    inputs = torch.randn(2, 1, 2, 8, 16, requires_grad=True)
    output = model(inputs, ["a", "b"], ["a"])
    output.square().mean().backward()
    for handle in handles:
        handle.remove()

    assert output.shape == (2, 1, 8, 16)
    assert lengths == [32, 8, 8, 32]
    assert torch.isfinite(output).all()


def test_edm_sparse_forward_backward():
    model = EDM(
        default_vars=["a", "b"],
        img_size=(8, 16),
        in_channels=2,
        out_channels=1,
        history=1,
        cnn_ratio=2,
        patch_size=2,
        drop_path=0.0,
        drop_rate=0.0,
        embed_dim=32,
        depth=4,
        decoder_depth=1,
        num_heads=4,
        FusedAttn_option=FusedAttn.DEFAULT,
        sigma_data=0.5,
        keep_ratio=0.25,
        num_dense_early=1,
        num_sparse_middle=2,
    )
    lengths, handles = _record_sequence_lengths(model.blocks)
    noisy = torch.randn(2, 1, 1, 8, 16, requires_grad=True)
    conditions = torch.randn(2, 1, 2, 8, 16)
    sigma = torch.tensor([0.5, 1.0])
    output = model(noisy, ["a", "b"], ["a"], sigma, conditions)
    output.square().mean().backward()
    for handle in handles:
        handle.remove()

    assert output.shape == (2, 1, 8, 16)
    assert lengths == [32, 8, 8, 32]
    assert torch.isfinite(output).all()


def test_edm_loss_applies_noise_weights():
    prediction = torch.ones(2, 1, 2, 2)
    target = torch.zeros_like(prediction)
    weights = torch.tensor([1.0, 3.0]).view(2, 1, 1, 1)

    loss = mse(
        prediction,
        target,
        aggregate_only=True,
        diffusion_weights=weights,
    )

    assert torch.isclose(loss, torch.tensor(2.0))

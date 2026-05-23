"""Sanity test for the symmetric output-space ``ModalitySpecificMoEBlock``.

Exercises the new Variant B implementation end-to-end:

1. Constructs a block with the production config (cam_channels=80,
   lidar_channels=256, out_channels=256, 2+2 bottleneck experts).
2. Pushes random ``cam_bev`` (B, 80, H, W) and ``lidar_bev``
   (B, 256, H, W) tensors through ``forward``.
3. Verifies output shape, finiteness, and the new ``moe_info`` fields
   (``modality_specific_design`` etc.).
4. Confirms ``cam_mass + lidar_mass ≈ 1`` (a flat-routing
   sanity check — the gate produces a softmax over all E experts).
5. Confirms the camera path is **active at init** — swapping
   ``cam_bev`` for zeros changes the output by a non-zero relative
   amount.  (The symmetric design must NOT have a LiDAR-anchored
   identity contract.)
6. Runs a backward pass and verifies gradients flow to every learnable
   component: ``cam_direct_proj``, ``cam_experts``, ``lidar_experts``,
   ``refine``, ``gate``, ``cam_summary`` / ``lidar_summary``, and
   ``context_head``.

Run with::

    /home/users/u103958/miniconda3/envs/multimodal-moe/bin/python \\
        scripts/bev_scripts/sanity_modality_specific_moe.py

CPU-only by default — pass ``--cuda`` to run on GPU if available.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MMDET3D_ROOT = PROJECT_ROOT / 'mmdetection3d'
for p in (PROJECT_ROOT, MMDET3D_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

# Import the BEVFusion project (registers all MoE modules).
import projects.BEVFusion.bevfusion  # noqa: F401

from projects.BEVFusion.bevfusion.moe_bev import (  # noqa: E402
    ModalitySpecificMoEBlock)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', action='store_true',
                        help='Run on GPU if available.')
    parser.add_argument('--batch', type=int, default=2)
    parser.add_argument('--height', type=int, default=180)
    parser.add_argument('--width', type=int, default=180)
    args = parser.parse_args()

    device = torch.device('cuda' if args.cuda and torch.cuda.is_available()
                          else 'cpu')
    print(f'[sanity] device={device}')

    torch.manual_seed(0)

    # ── Construct block with the production config ─────────────────────
    block = ModalitySpecificMoEBlock(
        cam_channels=80,
        lidar_channels=256,
        out_channels=256,
        num_cam_experts=2,
        num_lidar_experts=2,
        gate_type='dense',
        gate_cfg=dict(temperature=1.0),
        gate_input_detach=True,
        importance_coef=0.005,
        load_coef=0.0,
        z_loss_coef=0.002,
        group_balance_coef=0.004,
        residual_gain=1.0,
        router_out_dim=128,
        expert_type='bottleneck',
        expert_hidden_channels=128,
        context_aux_cfg=dict(
            target_field='road_type',
            loss_coef=0.03,
            loss_type='weighted_ce',
            class_weights='inverse_frequency',
            label_smoothing=0.05,
        ),
    ).to(device)

    # ── Inputs ─────────────────────────────────────────────────────────
    cam_bev = torch.randn(args.batch, 80, args.height, args.width,
                          device=device)
    lidar_bev = torch.randn(args.batch, 256, args.height, args.width,
                            device=device)

    metas = [
        {'context': {'road_type': 'city'}},
        {'context': {'road_type': 'highway'}},
    ][:args.batch]

    # ── 1) Forward + shape contract ────────────────────────────────────
    block.train()
    fused, moe_info = block(cam_bev, lidar_bev, batch_input_metas=metas)

    assert fused.shape == (args.batch, 256, args.height, args.width), (
        f'Expected fused shape ({args.batch}, 256, {args.height}, '
        f'{args.width}), got {tuple(fused.shape)}')
    assert torch.isfinite(fused).all(), 'fused has NaN/Inf'
    print(f'[sanity] block.forward(train) fused.shape = {tuple(fused.shape)}'
          f'  max|fused| = {fused.abs().max().item():.3f}')

    # ── 2) moe_info contract ───────────────────────────────────────────
    required_keys = [
        'full_softmax_probs', 'topk_idx', 'topk_weights',
        'cam_expert_ids', 'lidar_expert_ids',
        'cam_group_mass', 'lidar_group_mass',
        'aux_loss', 'importance_loss', 'load_loss', 'router_z_loss',
        'group_balance_loss', 'ctx_aux_loss', 'ctx_aux_loss_weighted',
        'ctx_aux_acc', 'gate_type', 'dense_dispatch',
        # Symmetric-design fields:
        'modality_specific_design',
        'cam_direct_channels',
        'lidar_direct_channels',
        'expert_input_channels',
        'expert_output_channels',
    ]
    missing = [k for k in required_keys if k not in moe_info]
    assert not missing, f'moe_info missing keys: {missing}'

    assert moe_info['modality_specific_design'] == \
        'symmetric_output_space_bottleneck', (
            f"Unexpected modality_specific_design="
            f"'{moe_info['modality_specific_design']}'")
    assert moe_info['cam_direct_channels']    == 256
    assert moe_info['lidar_direct_channels']  == 256
    assert moe_info['expert_input_channels']  == 256
    assert moe_info['expert_output_channels'] == 256
    assert moe_info['cam_expert_ids']   == [0, 1]
    assert moe_info['lidar_expert_ids'] == [2, 3]

    # No leftover LiDAR-anchored field should be advertised.
    assert 'lidar_base_channels' not in moe_info, (
        "moe_info contains 'lidar_base_channels' — that field is from the "
        "rejected LiDAR-anchored design and must not appear in the "
        "symmetric implementation.")
    print(f'[sanity] moe_info OK: design='
          f"'{moe_info['modality_specific_design']}'  "
          f"cam_mass={moe_info['cam_group_mass']:.3f}  "
          f"lidar_mass={moe_info['lidar_group_mass']:.3f}")

    # ── 3) Flat-routing softmax sanity — masses sum to ≈ 1 ─────────────
    mass_sum = moe_info['cam_group_mass'] + moe_info['lidar_group_mass']
    assert abs(mass_sum - 1.0) < 1e-4, (
        f'cam_group_mass + lidar_group_mass = {mass_sum:.6f} '
        f'(expected ≈ 1.0 — softmax over all E experts).')
    print(f'[sanity] flat softmax: cam_mass + lidar_mass = {mass_sum:.6f}')

    # ── 4) Camera direct path is ACTIVE at init ────────────────────────
    # In the symmetric design, cam_direct_proj produces non-zero
    # features from step 0 (it is NOT zero-initialised).  Therefore
    # swapping cam_bev for zeros must change the output by a non-zero
    # relative amount — otherwise the camera path is dead and we have
    # silently reverted to a LiDAR-anchored design.
    with torch.no_grad():
        fused_zero_cam, _ = block(torch.zeros_like(cam_bev), lidar_bev,
                                  batch_input_metas=metas)
        rel_diff = ((fused - fused_zero_cam).abs().mean()
                    / (fused.abs().mean() + 1e-9)).item()
    print(f'[sanity] |fused - fused_zero_cam| / |fused| = {rel_diff:.4f}'
          f'   (must be > 0 — camera direct path is live at init)')
    assert rel_diff > 0.0, (
        'Camera direct path appears to be zero at init — the symmetric '
        'design requires cam_direct_proj to be live from step 0.')

    # ── 5) Auxiliary loss is finite ────────────────────────────────────
    assert torch.isfinite(moe_info['aux_loss']).all(), \
        'aux_loss has NaN/Inf'
    print(f"[sanity] aux_loss = {moe_info['aux_loss'].item():.4f}, "
          f"gb_loss = {moe_info['group_balance_loss'].item():.4f}, "
          f"ctx_acc = {moe_info['ctx_aux_acc'].item():.3f}")

    # ── 6) Backward pass — gradients reach every component ─────────────
    loss = fused.mean() + moe_info['aux_loss']
    loss.backward()

    # ── Components that MUST have non-zero gradient at step 0 ─────────
    # Everything outside the expert adapter branches receives gradient
    # immediately.  Each expert's *final* BatchNorm affine (γ, β) also
    # gets gradient at step 0 even though γ=0 makes the adapter output
    # exactly zero — that is the mechanism by which the expert symmetry
    # is broken from iteration 1 onwards.
    nonzero_grad_params = {
        'cam_direct_proj[0].weight':
            block.cam_direct_proj[0].weight,
        'refine[0].weight':
            block.refine[0].weight,
        'gate.gate.weight':
            block.gate.gate.weight,
        'cam_summary.proj[1].weight':
            block.cam_summary.proj[1].weight,
        'lidar_summary.proj[1].weight':
            block.lidar_summary.proj[1].weight,
        'context_head[0].weight':
            block.context_head[0].weight,
        # Final BN affine of each bottleneck expert — zero-init γ but
        # always receives non-zero gradient (BN's own dL/dγ is non-zero
        # whenever the normalised input is non-constant).
        'cam_experts[0].expand_bn.weight':
            block.cam_experts[0].expand_bn.weight,
        'cam_experts[0].expand_bn.bias':
            block.cam_experts[0].expand_bn.bias,
        'cam_experts[1].expand_bn.weight':
            block.cam_experts[1].expand_bn.weight,
        'lidar_experts[0].expand_bn.weight':
            block.lidar_experts[0].expand_bn.weight,
        'lidar_experts[0].expand_bn.bias':
            block.lidar_experts[0].expand_bn.bias,
        'lidar_experts[1].expand_bn.weight':
            block.lidar_experts[1].expand_bn.weight,
    }
    for name, param in nonzero_grad_params.items():
        grad = param.grad
        assert grad is not None, f'No gradient on {name}'
        assert torch.isfinite(grad).all(), f'NaN/Inf gradient on {name}'
        gnorm = grad.abs().mean().item()
        assert gnorm > 0.0, (
            f'Expected non-zero gradient on {name} — got mean|g|={gnorm}')
        print(f'[sanity] grad reached {name:<56} '
              f'mean|g|={gnorm:.3e}')

    # ── Identity-at-init contract on expert adapter inner conv weights ─
    # The bottleneck experts have a zero-init final BatchNorm (γ=0), so
    # the adapter contribution at step 0 is identically zero and the
    # inner conv weights inside the adapter branch (reduce/spatial/
    # expand_conv) receive *zero* gradient on the very first backward
    # pass.  This is by design — it is what gives the experts their
    # identity-at-init behaviour.  The symmetry is broken from
    # iteration 1 onwards via the expand_bn affine gradient checked
    # above.
    zero_grad_params = {
        'cam_experts[0].reduce[0].weight':
            block.cam_experts[0].reduce[0].weight,
        'cam_experts[1].reduce[0].weight':
            block.cam_experts[1].reduce[0].weight,
        'lidar_experts[0].reduce[0].weight':
            block.lidar_experts[0].reduce[0].weight,
        'lidar_experts[1].reduce[0].weight':
            block.lidar_experts[1].reduce[0].weight,
    }
    for name, param in zero_grad_params.items():
        grad = param.grad
        assert grad is not None, f'No gradient tensor on {name}'
        assert torch.isfinite(grad).all(), f'NaN/Inf gradient on {name}'
        gnorm = grad.abs().mean().item()
        assert gnorm == 0.0, (
            f'Expected ZERO gradient on {name} at step 0 (identity-at-init '
            f'contract of BEVBottleneckResidualExpert), got mean|g|={gnorm}')
        print(f'[sanity] identity-at-init: {name:<56} '
              f'mean|g|={gnorm:.3e} (expected 0)')

    # ── 7) Negative test: lidar_channels != out_channels must raise ────
    try:
        ModalitySpecificMoEBlock(
            cam_channels=80,
            lidar_channels=128,   # ← intentional mismatch
            out_channels=256,
            num_cam_experts=2,
            num_lidar_experts=2,
            gate_type='dense',
            gate_input_detach=True,
            context_aux_cfg=dict(
                target_field='road_type',
                loss_coef=0.03,
                loss_type='weighted_ce',
                class_weights='inverse_frequency',
                label_smoothing=0.05,
            ),
        )
    except ValueError as e:
        assert 'lidar_channels == out_channels' in str(e), (
            f'Expected lidar_channels mismatch ValueError, got: {e}')
        print('[sanity] lidar_channels != out_channels correctly rejected')
    else:
        raise AssertionError(
            'Constructing ModalitySpecificMoEBlock with '
            'lidar_channels != out_channels should have raised ValueError.')

    print('[sanity] ALL CHECKS PASSED')
    return 0


if __name__ == '__main__':
    sys.exit(main())

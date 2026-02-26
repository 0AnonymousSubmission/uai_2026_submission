# type: ignore
"""
Proper initialization strategies for tensor networks.

All normalization functions:
- Work inplace (modify the TN directly)
- Accept single TN or list of TNs (for TypeI models)
- Return the scale factor(s) applied
"""

import torch
import numpy as np
import quimb.tensor as qt
from typing import Union, List


def normalize_tn_output(
    tn, input_samples, output_dims, batch_dim="s", target_std=0.1, max_samples=1000, inplace=True
):
    """
    Normalize tensor network so initial outputs have target_std for given inputs.

    Strategy:
    1. Compute predictions on sample inputs
    2. Measure output std
    3. Scale all trainable tensors by factor: target_std / current_std

    Args:
        tn: TensorNetwork or list of TensorNetworks to normalize
        input_samples: Sample inputs (TensorNetwork from create_inputs)
        output_dims: Output dimension names
        batch_dim: Batch dimension name
        target_std: Target standard deviation for outputs (default: 0.1)
        max_samples: Maximum number of samples to use for estimation
        inplace: If True, modify TN in place. If False, return copy (default: True)

    Returns:
        scale_factor: The scaling factor(s) applied (single value or list)
    """
    if isinstance(tn, list):
        return [
            normalize_tn_output(
                t, input_samples, output_dims, batch_dim, target_std, max_samples, inplace
            )
            for t in tn
        ]

    if not inplace:
        tn = tn.copy()

    if hasattr(input_samples, "__iter__") and not isinstance(input_samples, (list, tuple)):
        sample_batch = next(iter(input_samples))
        if isinstance(sample_batch, tuple):
            inputs = sample_batch[0]
        else:
            inputs = sample_batch
    else:
        inputs = input_samples

    tn_backend = None
    for t in tn:
        if torch.is_tensor(t.data):
            tn_backend = "torch"
            break
        else:
            tn_backend = "numpy"
            break

    if tn_backend == "torch":
        for t in inputs:
            if not torch.is_tensor(t.data):
                t.modify(data=torch.from_numpy(np.asarray(t.data)))
    else:
        for t in inputs:
            if torch.is_tensor(t.data):
                t.modify(data=t.data.numpy())

    output_inds = [batch_dim] + output_dims
    full_tn = tn & inputs
    y_pred = full_tn.contract(output_inds=output_inds)

    if len(output_inds) > 0:
        y_pred.transpose_(*output_inds)

    pred_data = y_pred.data
    if torch.is_tensor(pred_data):
        current_std = pred_data.std().item()
    else:
        current_std = np.std(pred_data)

    if current_std < 1e-10:
        print(f"Warning: Current std is very small ({current_std:.2e}), using default scaling")
        scale_factor = target_std / 0.01
    else:
        scale_factor = target_std / current_std

    for tensor in tn:
        if "NT" not in tensor.tags:
            tensor.modify(data=tensor.data * scale_factor)

    if not inplace:
        return tn, scale_factor
    return scale_factor


def normalize_tn_ones(tn, input_dims, output_dims, target_output=1.0, inplace=True):
    """
    Normalize tensor network by passing ones through it.

    Strategy:
    1. Create input tensors of all ones
    2. Contract to get output
    3. Scale all trainable tensors so output magnitude equals target_output

    Args:
        tn: TensorNetwork or list of TensorNetworks to normalize
        input_dims: List of input dimension names (e.g., ['i0', 'i1', 'i2'])
        output_dims: List of output dimension names (e.g., ['out'])
        target_output: Target output magnitude (default: 1.0)
        inplace: If True, modify TN in place. If False, return copy (default: True)

    Returns:
        scale_factor: The scaling factor(s) applied
    """
    if isinstance(tn, list):
        return [normalize_tn_ones(t, input_dims, output_dims, target_output, inplace) for t in tn]

    if not inplace:
        tn = tn.copy()

    tn_backend = "torch" if torch.is_tensor(list(tn.tensors)[0].data) else "numpy"

    input_tensors = []
    for dim in input_dims:
        size = None
        for tensor in tn.tensors:
            if dim in tensor.inds:
                size = tensor.ind_size(dim)
                break
        if size is None:
            raise ValueError(f"Could not find dimension {dim} in tensor network")

        if tn_backend == "torch":
            data = torch.ones(size, dtype=torch.float64)
        else:
            data = np.ones(size)
        input_tensors.append(qt.Tensor(data, inds=[dim], tags=[f"Input_{dim}"]))

    input_tn = qt.TensorNetwork(input_tensors)
    full_tn = tn & input_tn
    result = full_tn.contract(output_inds=output_dims)

    output_data = result.data
    if tn_backend == "torch":
        current_magnitude = output_data.abs().mean().item()
    else:
        current_magnitude = np.abs(output_data).mean()

    if current_magnitude < 1e-15:
        print(f"Warning: Output magnitude is very small ({current_magnitude:.2e}), using default")
        scale_factor = 1.0
    else:
        n_tensors = sum(1 for t in tn.tensors if "NT" not in t.tags)
        scale_factor = (target_output / current_magnitude) ** (1.0 / n_tensors)

    for tensor in tn:
        if "NT" not in tensor.tags:
            tensor.modify(data=tensor.data * scale_factor)

    if not inplace:
        return tn, scale_factor
    return scale_factor


def normalize_tn_frobenius(tn, target_norm=1.0, exclude_tags=None, inplace=True):
    """
    Normalize tensor network by Frobenius norm.

    Scales all trainable tensors so the total Frobenius norm equals target_norm.

    Args:
        tn: TensorNetwork or list of TensorNetworks to normalize
        target_norm: Target Frobenius norm
        exclude_tags: Tags to exclude from normalization (e.g., ['NT'])
        inplace: If True, modify TN in place. If False, return copy (default: True)

    Returns:
        scale_factor: The scaling factor(s) applied (single value or list)
    """
    if isinstance(tn, list):
        return [normalize_tn_frobenius(t, target_norm, exclude_tags, inplace) for t in tn]

    if not inplace:
        tn = tn.copy()

    if exclude_tags is None:
        exclude_tags = ["NT"]

    total_norm_sq = 0.0
    for tensor in tn:
        if not any(tag in tensor.tags for tag in exclude_tags):
            if torch.is_tensor(tensor.data):
                total_norm_sq += (tensor.data**2).sum().item()
            else:
                total_norm_sq += np.sum(tensor.data**2)

    current_norm = np.sqrt(total_norm_sq)

    if current_norm < 1e-10:
        print(f"Warning: Current norm is very small ({current_norm:.2e}), skipping normalization")
        if not inplace:
            return tn, 1.0
        return 1.0

    scale_factor = target_norm / current_norm

    for tensor in tn:
        if not any(tag in tensor.tags for tag in exclude_tags):
            tensor.modify(data=tensor.data * scale_factor)

    if not inplace:
        return tn, scale_factor
    return scale_factor


def init_mps_normalized(
    L,
    bond_dim,
    phys_dim,
    output_dim,
    output_site=None,
    base_init=0.1,
    sample_inputs=None,
    target_std=0.1,
):
    """
    Initialize MPS and normalize based on sample inputs.

    This is the recommended initialization strategy:
    1. Initialize with base_init (e.g., 0.1)
    2. Normalize based on actual outputs on sample data

    Args:
        L, bond_dim, phys_dim, output_dim, output_site: Model parameters
        base_init: Initial scale before normalization
        sample_inputs: Sample inputs for normalization (if None, use Frobenius norm)
        target_std: Target output std (default: 0.1)

    Returns:
        Initialized and normalized tensor network
    """
    from model.MPO2_models import MPO2

    model = MPO2(
        L=L,
        bond_dim=bond_dim,
        phys_dim=phys_dim,
        output_dim=output_dim,
        output_site=output_site,
        init_strength=base_init,
    )

    if sample_inputs is not None:
        scale = normalize_tn_output(
            model.tn, sample_inputs, output_dims=model.output_dims, target_std=target_std
        )
        print(f"Normalized TN by factor {scale:.6f} (output-based)")
    else:
        target_norm = np.sqrt(L * bond_dim * phys_dim)
        scale = normalize_tn_frobenius(model.tn, target_norm=target_norm)
        print(f"Normalized TN by factor {scale:.6f} (Frobenius norm)")

    return model


def init_mps_orthogonal(
    L, bond_dim, phys_dim, output_dim, output_site=None, init_strength=0.1, method="qr"
):
    """
    Initialize MPS with orthogonal/unitary structure.

    Strategy:
    - method='qr': Apply QR decomposition to each tensor
    - method='quimb': Use quimb's canonical form (requires reshaping output dimension)

    Orthogonal initialization:
    - Preserves gradient flow (orthogonal matrices have unit singular values)
    - Reduces correlation between random directions
    - Common in RNN initialization (e.g., orthogonal LSTM)

    Args:
        L, bond_dim, phys_dim, output_dim, output_site: Model parameters
        init_strength: Overall scale factor after orthogonalization
        method: 'qr' for manual QR, 'quimb' for quimb canonicalization

    Returns:
        Initialized MPO2 model with orthogonal structure
    """
    from model.MPO2_models import MPO2

    if output_site is None:
        output_site = L // 2

    model = MPO2(
        L=L,
        bond_dim=bond_dim,
        phys_dim=phys_dim,
        output_dim=output_dim,
        output_site=output_site,
        init_strength=1.0,
    )

    if method == "quimb":
        output_tensor = None
        for tensor in model.tn:
            if f"Node{output_site}" in tensor.tags:
                output_tensor = tensor
                break

        if output_tensor is None:
            raise ValueError(f"Could not find output tensor at site {output_site}")

        original_shape = output_tensor.data.shape
        original_inds = output_tensor.inds

        if len(original_shape) == 3:
            temp_shape = (original_shape[0], original_shape[1] * original_shape[2])
            temp_inds = (original_inds[0], "temp_merged")
        elif len(original_shape) == 4:
            temp_shape = (
                original_shape[0],
                original_shape[1],
                original_shape[2] * original_shape[3],
            )
            temp_inds = (original_inds[0], original_inds[1], "temp_merged")
        else:
            temp_shape = (original_shape[0] * original_shape[1],)
            temp_inds = ("temp_merged",)

        temp_data = output_tensor.data.reshape(temp_shape)
        output_tensor.modify(data=temp_data, inds=temp_inds)

        try:
            model.tn.left_canonize(normalize=False)
        except:
            pass

        for tensor in model.tn:
            if f"Node{output_site}" in tensor.tags:
                current_data = tensor.data
                restored_data = current_data.reshape(original_shape)
                tensor.modify(data=restored_data * init_strength, inds=original_inds)
                break

        for tensor in model.tn:
            if f"Node{output_site}" not in tensor.tags:
                tensor.modify(data=tensor.data * init_strength)

    elif method == "qr":
        for i, tensor in enumerate(model.tn):
            data = tensor.data
            original_shape = data.shape

            if len(original_shape) == 2:
                rows, cols = original_shape
            elif len(original_shape) == 3:
                rows = original_shape[0]
                cols = original_shape[1] * original_shape[2]
            elif len(original_shape) == 4:
                rows = original_shape[0]
                cols = original_shape[1] * original_shape[2] * original_shape[3]
            else:
                raise ValueError(f"Unexpected tensor shape: {original_shape}")

            matrix = data.reshape(rows, cols)

            if torch.is_tensor(matrix):
                q, r = torch.linalg.qr(matrix, mode="reduced")
                min_dim = min(q.shape)
                signs = torch.sign(torch.diagonal(r)[:min_dim])
                signs[signs == 0] = 1
                q = q * signs.unsqueeze(0)

                if q.numel() == np.prod(original_shape):
                    orthogonal_data = q.reshape(original_shape) * init_strength
                else:
                    scaled_matrix = (
                        matrix * init_strength / (matrix.norm() / np.sqrt(matrix.numel()))
                    )
                    orthogonal_data = scaled_matrix.reshape(original_shape)
            else:
                q, r = np.linalg.qr(matrix, mode="reduced")
                min_dim = min(q.shape)
                signs = np.sign(np.diagonal(r)[:min_dim])
                signs[signs == 0] = 1
                q = q * signs[np.newaxis, :]

                if q.size == np.prod(original_shape):
                    orthogonal_data = q.reshape(original_shape) * init_strength
                else:
                    scaled_matrix = (
                        matrix * init_strength / (np.linalg.norm(matrix) / np.sqrt(matrix.size))
                    )
                    orthogonal_data = scaled_matrix.reshape(original_shape)

            tensor.modify(data=orthogonal_data)

    else:
        raise ValueError(f"Unknown method: {method}. Use 'qr' or 'quimb'")

    return model

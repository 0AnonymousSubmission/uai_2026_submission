# type: ignore
import torch
import quimb.tensor as qt
from typing import Optional


class CPD:
    """
    Canonical Polyadic Decomposition (CP) tensor network.

    Structure:
    - All nodes share a single bond index 'r' (the rank)
    - Node0: (x0, r)
    - Node1: (x1, r)
    - ...
    - NodeL-1: (xL-1, r) with optional output dimension
    """

    bond_prior_alpha = 5.0

    def __init__(
        self,
        L: int,
        bond_dim: int,
        phys_dim: int,
        output_dim: int = 1,
        output_site: Optional[int] = None,
        init_strength: float = 0.001,
        use_tn_normalization: bool = True,
        tn_target_std: float = 0.1,
        sample_inputs: Optional[qt.TensorNetwork] = None,
    ):
        self.L = L
        self.bond_dim = bond_dim
        self.phys_dim = phys_dim
        self.output_dim = output_dim
        self.output_site = output_site if output_site is not None else L - 1

        base_init = 0.1 if use_tn_normalization else init_strength

        tensors = []
        for i in range(L):
            shape = (phys_dim, bond_dim)
            inds = (f"x{i}", "r")

            if i == self.output_site and output_dim > 1:
                shape = shape + (output_dim,)
                inds = inds + ("out",)

            data = torch.randn(*shape) * base_init
            tensor = qt.Tensor(data=data, inds=inds, tags={f"Node{i}"})
            tensors.append(tensor)

        self.tn = qt.TensorNetwork(tensors)

        if use_tn_normalization:
            from model.initialization import normalize_tn_output, normalize_tn_frobenius

            if sample_inputs is not None:
                normalize_tn_output(
                    self.tn,
                    sample_inputs,
                    output_dims=["out"] if output_dim > 1 else [],
                    batch_dim="s",
                    target_std=tn_target_std,
                )
            else:
                import numpy as np

                target_norm = np.sqrt(L * bond_dim * phys_dim)
                normalize_tn_frobenius(self.tn, target_norm=target_norm)

        self.input_labels = [f"x{i}" for i in range(L)]
        self.input_dims = [f"x{i}" for i in range(L)]
        self.output_dims = ["out"] if output_dim > 1 else []

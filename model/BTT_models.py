# type: ignore
"""
Binary Tensor Tree (BTT) Models.
"""

import torch
import quimb.tensor as qt
import numpy as np
from typing import Optional


class BTT:
    bond_prior_alpha = 3.0

    def __init__(
        self,
        L: int,
        bond_dim: int,
        phys_dim: int,
        output_dim: int = 1,
        init_strength: float = 0.001,
        use_tn_normalization: bool = True,
        tn_target_std: float = 0.1,
        sample_inputs: Optional[qt.TensorNetwork] = None,
    ):
        if L < 2:
            raise ValueError(f"L must be >= 2, got {L}")

        self.L = L
        self.bond_dim = bond_dim
        self.phys_dim = phys_dim
        self.output_dim = output_dim

        base_init = 0.1 if use_tn_normalization else init_strength

        tensors = []

        for i in range(L):
            data = torch.randn(phys_dim, bond_dim) * base_init
            inds = (f"x{i}", f"b_leaf_{i}")
            tensor = qt.Tensor(data=data, inds=inds, tags={f"Leaf{i}"})
            tensors.append(tensor)

        child_bonds = [f"b_leaf_{i}" for i in range(L)]
        level = 0

        while len(child_bonds) > 2:
            new_child_bonds = []
            i = 0
            pos = 0
            while i < len(child_bonds):
                if i + 1 < len(child_bonds):
                    left_child = child_bonds[i]
                    right_child = child_bonds[i + 1]
                    parent_bond = f"b_lvl{level}_{pos}"

                    data = torch.randn(bond_dim, bond_dim, bond_dim) * base_init
                    inds = (left_child, right_child, parent_bond)
                    tensor = qt.Tensor(data=data, inds=inds, tags={f"Internal{level}_{pos}"})
                    tensors.append(tensor)
                    new_child_bonds.append(parent_bond)
                    i += 2
                else:
                    new_child_bonds.append(child_bonds[i])
                    i += 1
                pos += 1

            child_bonds = new_child_bonds
            level += 1

        left_child = child_bonds[0]
        right_child = child_bonds[1] if len(child_bonds) > 1 else None

        if right_child:
            if output_dim > 1:
                data = torch.randn(bond_dim, bond_dim, output_dim) * base_init
                inds = (left_child, right_child, "out")
            else:
                data = torch.randn(bond_dim, bond_dim) * base_init
                inds = (left_child, right_child)
        else:
            if output_dim > 1:
                data = torch.randn(bond_dim, output_dim) * base_init
                inds = (left_child, "out")
            else:
                data = torch.randn(bond_dim) * base_init
                inds = (left_child,)
        tensor = qt.Tensor(data=data, inds=inds, tags={"Root"})
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
                target_norm = np.sqrt(L * bond_dim * phys_dim)
                normalize_tn_frobenius(self.tn, target_norm=target_norm)

        self.input_labels = [f"x{i}" for i in range(L)]
        self.input_dims = [f"x{i}" for i in range(L)]
        self.output_dims = ["out"] if output_dim > 1 else []

    def draw(self, **kwargs):
        return self.tn.draw(**kwargs)

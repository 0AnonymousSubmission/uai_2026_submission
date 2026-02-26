# type: ignore
import torch
import random
from tensor.distributions import GammaDistribution, MultivariateGaussianDistribution
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
import quimb.tensor as qt  # Assuming quimb.tensor is available

torch.set_default_dtype(torch.float64)  # or torch.float64


class Inputs:
    """
    Input loader that pre-computes and stores batches in a list.
    Allows efficient multiple passes (epochs) over the data.
    """

    def __init__(
        self,
        inputs: List[Any],
        outputs: List[Any],
        outputs_labels: List[str],
        input_labels: List[str],
        batch_dim: str = "s",
        batch_size=None,
    ):
        # 1. Configuration
        self.inputs_data = inputs
        self.outputs_data = outputs
        self.outputs_labels = outputs_labels
        self.input_labels = input_labels
        self.batch_dim = batch_dim

        self.batch_size = inputs[0].shape[0] if batch_size is None else batch_size
        self.samples = outputs[0].shape[0]
        self.repeated = len(inputs) == 1

        # 2. Pre-compute all batches once and store them
        # Storage format: List[Tuple(mu_tensors, prime_tensors, y_tensor)]
        self.batches = self._create_batches()

    def _create_batches(self) -> List[Tuple[List[qt.Tensor], List[qt.Tensor], qt.Tensor]]:
        """Generates and stores the list of all batches."""
        batches = []

        # Generator for raw data slices
        raw_splits = self.batch_splits(self.inputs_data, self.outputs_data[0], self.batch_size)

        # Process into tensors
        for input_dict, y_tensor in raw_splits:
            if self.repeated:
                mu, prime = self.prepare_inputs_batch_repeated(input_dict)
            else:
                mu, prime = self.prepare_inputs_batch(input_dict)

            batches.append((mu, prime, y_tensor))

        return batches

    # --- Properties for Iteration ---

    @property
    def data_mu(self):
        """Yields only mu input tensors."""
        for mu, _, _ in self.batches:
            yield mu

    @property
    def data_sigma(self):
        """
        Yields full input for sigma network.
        Combines mu tensors + prime tensors (x, x').
        """
        for mu, prime, _ in self.batches:
            yield mu + prime

    @property
    def data_y(self):
        """Yields only target (y) tensors."""
        for _, _, y in self.batches:
            yield y

    @property
    def data_mu_y(self):
        """Yields (mu inputs, target)."""
        for mu, _, y in self.batches:
            yield mu, y

    # --- Processing Methods ---

    def batch_splits(self, xs, y, B):
        """Generates raw dictionary/array slices."""
        s = y.shape[0]
        for i in range(0, s, B):
            tensor = qt.Tensor(
                data=y[i : i + B], inds=(self.batch_dim, *self.outputs_labels), tags={"output"}
            )
            batch = {f"{j}": x[i : i + B] for j, x in zip(self.input_labels[: len(xs)], xs)}
            yield batch, tensor

    def prepare_inputs_batch(
        self, input_data: Dict[str, Any]
    ) -> Tuple[List[qt.Tensor], List[qt.Tensor]]:
        """
        Returns:
            tensors_mu: List of tensors for mu network [x1, x2...]
            tensors_prime: List of prime tensors for sigma network [x1', x2'...]
        """
        tensors_mu = []
        tensors_prime = []
        for k, v in input_data.items():
            # Mu tensor
            tensor = qt.Tensor(data=v, inds=(self.batch_dim, k), tags={f"input_{k}"})
            tensors_mu.append(tensor)

            # Sigma (prime) tensor
            prime_idx = f"{k}_prime"
            tensor_prime = qt.Tensor(
                data=v, inds=(self.batch_dim, prime_idx), tags={f"input_{prime_idx}"}
            )
            tensors_prime.append(tensor_prime)

        return tensors_mu, tensors_prime

    def shuffle(self):
        """Shuffles the internal list of batches in-place."""
        random.shuffle(self.batches)

    def __len__(self):
        """Returns the number of batches."""
        return len(self.batches)

    def __getitem__(self, idx):
        """
        Returns the raw batch at the specified index.

        Returns:
            Tuple: (mu_tensors, prime_tensors, y_tensor)

        Note: To get full sigma inputs from this, you must concatenate mu + prime.
        """
        return self.batches[idx]

    def prepare_inputs_batch_repeated(
        self, input_data: Dict[str, Any]
    ) -> Tuple[List[qt.Tensor], List[qt.Tensor]]:
        input_indices = self.input_labels
        single_key = list(input_data.keys())[0]
        data = input_data[single_key]

        tensors_mu = []
        tensors_prime = []
        for input_idx in input_indices:
            # Mu tensor
            tensor = qt.Tensor(
                data=data, inds=(self.batch_dim, input_idx), tags={f"input_{input_idx}"}
            )
            tensors_mu.append(tensor)

            # Sigma (prime) tensor
            prime_idx = f"{input_idx}_prime"
            tensor_prime = qt.Tensor(
                data=data, inds=(self.batch_dim, prime_idx), tags={f"input_{prime_idx}"}
            )
            tensors_prime.append(tensor_prime)

        return tensors_mu, tensors_prime

    def trim(self, label: str, keep_indices: List[int]):
        """
        Trim input dimension for a specific label across all batches.
        Called by BTN when it trims an input dimension.
        
        Args:
            label: Input label to trim (e.g., 'x0')
            keep_indices: List of indices to keep
        """
        prime_label = f"{label}_prime"
        
        for mu_tensors, prime_tensors, _ in self.batches:
            for tensor in mu_tensors:
                if label in tensor.inds:
                    label_axis = tensor.inds.index(label)
                    slicer = [slice(None)] * tensor.data.ndim
                    slicer[label_axis] = keep_indices
                    new_data = tensor.data[tuple(slicer)]
                    tensor.modify(data=new_data)
            
            for tensor in prime_tensors:
                if prime_label in tensor.inds:
                    label_axis = tensor.inds.index(prime_label)
                    slicer = [slice(None)] * tensor.data.ndim
                    slicer[label_axis] = keep_indices
                    new_data = tensor.data[tuple(slicer)]
                    tensor.modify(data=new_data)

    def __str__(self):
        """Summary of the loader structure."""
        if not self.batches:
            return ">>> InputLoader (Empty)"

        # Peek at the first stored batch
        mu, prime, y = self.batches[0]

        mu_inds = [list(t.inds) for t in mu]
        prime_inds = [list(t.inds) for t in prime]

        header = (
            f"\n>>> InputLoader Summary (Batch Size: {self.batch_size}, "
            f"Samples: {self.samples}, Batches: {len(self.batches)})\n"
            f"{'TYPE':<8} | {'SHAPE':<15} | {'INDICES'}\n"
            f"{'-' * 60}\n"
        )

        row_y = f"{'Target':<8} | {str(y.shape):<15} | {y.inds}\n"
        row_mu = f"{'Mu':<8} | {str(mu[0].shape):<15} | {mu_inds} ... ({len(mu)} tensors)\n"
        # Note: Sigma output is Mu + Prime, but we store them separate.
        row_sig = f"{'Sigma':<8} | {str(prime[0].shape):<15} | Mu + {prime_inds} ... (+{len(prime)} tensors)"

        return header + row_y + row_mu + row_sig


class BTNBuilder:
    """
    Builder class for constructing the Bayesian Tensor Network components.
    """

    def __init__(
        self,
        mu: qt.TensorNetwork,
        output_dimensions: List[str],
        batch_dim: str = "s",
        device=None,
        bond_prior_alpha: float = 1.0,
    ):
        self.mu = mu
        self.output_dimensions = output_dimensions
        self.batch_dim = batch_dim
        self.backend = self.mu.backend
        self.device = device if device is not None else torch.device("cpu")
        self.bond_prior_alpha = bond_prior_alpha

        self.p_bonds: Dict[str, GammaDistribution] = {}
        self.p_nodes: Dict[str, MultivariateGaussianDistribution] = {}

        self.q_bonds: Dict[str, GammaDistribution] = {}
        self.q_nodes: Dict[str, MultivariateGaussianDistribution] = {}

        self.sigma_tn: Optional[qt.TensorNetwork] = None

    def build_model(self) -> Tuple[Dict, Dict, Dict, Dict, qt.TensorNetwork]:
        all_inds = self.mu.ind_map
        self._build_edge_distributions(all_inds)
        self.sigma_tn = self._construct_sigma_topology()
        self._build_node_distributions()
        return self.p_bonds, self.p_nodes, self.q_bonds, self.q_nodes, self.sigma_tn

    def _build_edge_distributions(self, all_inds_map: Dict[str, Any]):
        """
        Constructs Gamma distributions with quimb.Tensor parameters.
        Tags: {ind}_alpha and {ind}_beta.

        NOW ALSO INCLUDES OUTPUT DIMENSIONS - treat them as variational parameters!

        FIXED: Prior (p_bonds) now uses FIXED hyperparameters (alpha=1, beta=1) for uninformative prior.
        """

        for ind, tids in all_inds_map.items():
            # Skip only batch dimension, NOT output dimensions anymore
            if ind == self.batch_dim:
                continue

            exemplar_tensor = self.mu.tensor_map[next(iter(tids))]
            dim_size = exemplar_tensor.shape[exemplar_tensor.inds.index(ind)]

            p_alpha = qt.Tensor(
                data=torch.ones(dim_size, dtype=torch.float64, device=self.device) * 10.0,
                inds=(ind,),
                tags={f"{ind}_alpha", "prior"},
            )
            p_beta = qt.Tensor(
                data=torch.ones(dim_size, dtype=torch.float64, device=self.device),
                inds=(ind,),
                tags={f"{ind}_beta", "prior"},
            )

            self.p_bonds[ind] = GammaDistribution(
                concentration=p_alpha, rate=p_beta, backend=self.backend
            )

            q_alpha = qt.Tensor(
                data=torch.ones(dim_size, dtype=torch.float64, device=self.device)
                * self.bond_prior_alpha,
                inds=(ind,),
                tags={f"{ind}_alpha", "posterior"},
            )
            q_beta = qt.Tensor(
                data=torch.ones(dim_size, dtype=torch.float64, device=self.device),
                inds=(ind,),
                tags={f"{ind}_beta", "posterior"},
            )

            self.q_bonds[ind] = GammaDistribution(
                concentration=q_alpha, rate=q_beta, backend=self.backend
            )

    def _construct_sigma_topology(self) -> qt.TensorNetwork:
        """
        Constructs the Sigma (Covariance) Tensor Network.
        This network represents the covariance of the nodes.
        It has a doubled structure (original indices + primed indices).
        We initialize it to be close to an identity matrix (diagonal) on the
        non-output dimensions to ensure positive definiteness.
        """
        sigma_tensors = []
        for tensor in self.mu:
            original_inds = tensor.inds
            original_tags = tensor.tags

            non_output_inds = []
            output_inds = []

            for ix in original_inds:
                if ix in self.output_dimensions:
                    output_inds.append(ix)
                else:
                    non_output_inds.append(ix)

            non_output_primes = [f"{ix}_prime" for ix in non_output_inds]
            sigma_inds = tuple(non_output_inds + non_output_primes + output_inds)

            shape_map = dict(zip(tensor.inds, tensor.shape))

            # 1. Calculate dimensions for non-output and output parts
            non_out_dims = [shape_map[ix] for ix in non_output_inds]
            out_dims = [shape_map[ix] for ix in output_inds]

            # 2. Total flattened size of non-output dimensions
            #    (e.g., if indices are 'a', 'b' with sizes 2, 3 -> d_non_out = 6)
            d_non_out = int(np.prod(non_out_dims))

            # 3. Create Identity Matrix for the non-output part
            eye_matrix = torch.eye(d_non_out, dtype=torch.float64, device=self.device)

            # 4. Reshape Identity back to tensor structure
            #    (d_non_out, d_non_out) -> (dim_a, dim_b, ..., dim_a_prime, dim_b_prime, ...)
            #    Note: This assumes the order in sigma_inds matches [non_outputs, non_outputs_prime, outputs]
            eye_tensor = eye_matrix.reshape(non_out_dims + non_out_dims)

            # 5. Handle Output Dimensions (Broadcasting)
            #    The covariance structure is usually repeated/shared across output dimensions for isotropic initialization
            if out_dims:
                # Expand dims for outputs
                for _ in out_dims:
                    eye_tensor = eye_tensor.unsqueeze(-1)

                # Expand values (broadcast)
                final_shape = non_out_dims + non_out_dims + out_dims
                sigma_data = eye_tensor.expand(final_shape).clone()
            else:
                sigma_data = eye_tensor.clone()

            # 6. Scaling
            #    Scale to be small enough
            # Just for testing
            shape_len = len(sigma_data.shape)
            if len(sigma_data.shape) == 4:
                shape_len = 6
            avg_rank = (
                torch.tensor(sigma_data.shape, dtype=torch.float64)
                .prod()
                .pow(1.0 / shape_len)
                .item()
            )
            new_shape_total_size = np.prod(sigma_data.shape)
            scale = 1.0 / (avg_rank * self.mu.num_tensors)
            sigma_data = sigma_data * scale

            sigma_tags = {f"{tag}" for tag in original_tags}

            sigma_tensor = qt.Tensor(data=sigma_data, inds=sigma_inds, tags=sigma_tags)
            sigma_tensors.append(sigma_tensor)

        return qt.TensorNetwork(sigma_tensors)

    def _build_node_distributions(self):
        for tensor in self.mu:
            node_key = tuple(sorted(list(tensor.tags)))

            sigma_tensor = None
            for tag in tensor.tags:
                sigma_tag = f"{tag}"
                try:
                    candidates = self.sigma_tn[sigma_tag]
                    if isinstance(candidates, qt.Tensor):
                        sigma_tensor = candidates
                        break
                except KeyError:
                    continue

            if sigma_tensor is None:
                raise ValueError(
                    f"Could not find corresponding Sigma tensor for node {tensor.tags}"
                )

            self.q_nodes[node_key] = MultivariateGaussianDistribution(
                loc=tensor, covariance_matrix=sigma_tensor
            )

            prior_loc = tensor.copy()
            prior_loc.modify(data=torch.zeros_like(tensor.data))

            self.p_nodes[node_key] = MultivariateGaussianDistribution(
                loc=prior_loc, covariance_matrix=None
            )

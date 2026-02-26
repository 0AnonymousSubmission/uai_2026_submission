# type: ignore
import torch.distributions as dist
import importlib
from typing import List, Dict, Optional, Tuple
import quimb.tensor as qt
import numpy as np
import torch
from tensor.builder import BTNBuilder, Inputs
from tensor.distributions import GammaDistribution

torch.set_default_dtype(torch.float64)  # or torch.float64

# Tag for nodes that should not be trained
NOT_TRAINABLE_TAG = "NT"


class BTN:
    def __init__(
        self,
        mu: qt.TensorNetwork,
        data_stream: Inputs,
        batch_dim: str = "s",
        not_trainable_nodes: List[str] = None,
        method="cholesky",
        device=None,
        bond_prior_alpha: float = 1.0,
    ):
        """
        Bayesian Tensor Network.

        Args:
            mu: The mean TensorNetwork topology.
            data_stream: Input data stream.
            batch_dim: Batch dimension label.
            not_trainable_nodes: List of node tags that should not be trained.
            method: Method for covariance computation ('cholesky' or 'direct').
            device: Device to use (defaults to 'cuda' if available, else 'cpu').
        """
        import torch

        self.device = (
            device
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.method = method
        self.mse = None
        self.sigma_forward = None
        self.mu = mu
        self.output_dimensions = data_stream.outputs_labels
        self.batch_dim = batch_dim
        self.input_indices = data_stream.input_labels
        self.data = data_stream
        self.train_data = data_stream
        self.val_data = None
        self.test_data = None
        self._all_data_streams = [data_stream]

        not_trainable_nodes = not_trainable_nodes or []
        for node_tag in not_trainable_nodes:
            tensors = self.mu.select_tensors(node_tag, which="any")
            for tensor in tensors:
                tensor.add_tag(NOT_TRAINABLE_TAG)
        self.not_trainable_nodes = not_trainable_nodes

        builder = BTNBuilder(
            self.mu,
            self.output_dimensions,
            self.batch_dim,
            device=self.device,
            bond_prior_alpha=bond_prior_alpha,
        )

        self.backend = self.mu.backend
        self.p_tau = GammaDistribution(
            concentration=torch.tensor(1.0, device=self.device),
            rate=torch.tensor(1.0, device=self.device),
            backend=self.backend,
        )
        self.q_tau = GammaDistribution(
            concentration=torch.tensor(1.0, device=self.device),
            rate=torch.tensor(1.0, device=self.device),
            backend=self.backend,
        )
        (self.p_bonds, self.p_nodes, self.q_bonds, self.q_nodes, self.sigma) = builder.build_model()

        self.threshold = 0.95
        self.delta_elbo = 0.0
        self.elbo_initial = None

        self.input_dims = self.input_indices
        self.output_dims = self.output_dimensions

        self._cache = {}

    def _clear_cache(self):
        """Clear all cached computations. Called after any model update."""
        self._cache = {}

    def register_data_streams(self, *data_streams):
        """Register additional data streams (val, test) for trimming."""
        for ds in data_streams:
            if ds is not None and ds not in self._all_data_streams:
                self._all_data_streams.append(ds)

    def _trim_input_data(self, bond_tag, indices_to_keep):
        """Trim input data streams for the given bond."""
        for data_stream in self._all_data_streams:
            data_stream.trim(bond_tag, indices_to_keep)

    def _get_cached(self, key):
        """Get cached value if exists."""
        return self._cache.get(key, None)

    def _set_cached(self, key, value):
        """Set cached value."""
        self._cache[key] = value
        return value

    def _compute_e_log_p(self):
        e_log_p_nodes = self._compute_e_log_p_nodes()
        e_log_p_bonds = self._compute_e_log_p_bonds()
        e_log_p_tau = self._compute_e_log_p_tau()
        return e_log_p_nodes + e_log_p_bonds + e_log_p_tau

    def _compute_e_log_p_nodes(self):
        e_log_p_nodes = 0
        node_tags = self.mu.tags
        for node_tag in node_tags:
            e_log_p_node = self.e_log_p_node(node_tag)
            e_log_p_nodes += e_log_p_node
        return e_log_p_nodes

    def _compute_e_log_p_bonds(self):
        total_e_log_p = 0.0
        for bond_tag in self.p_bonds:
            if hasattr(self, "_pending_soft_trims") and bond_tag in self._pending_soft_trims:
                indices_to_keep = self._pending_soft_trims[bond_tag]["indices_to_keep"]
                total_e_log_p += self._e_log_p_bond_subset(bond_tag, indices_to_keep)
            else:
                total_e_log_p += self.e_log_p_bond(bond_tag)
        return total_e_log_p

    def e_log_p_bond(self, bond_tag):
        # Parameters for q (variational)
        rate_q = self.q_bonds[bond_tag].rate
        conc_q = self.q_bonds[bond_tag].concentration

        # Parameters for p (prior)
        rate_p = self.p_bonds[bond_tag].rate
        conc_p = self.p_bonds[bond_tag].concentration

        # E_q[lambda]
        e_lambda = conc_q / rate_q

        # E_q[log lambda]
        e_log_lambda = torch.digamma(conc_q.data) - torch.log(rate_q.data)

        # Expected log p
        term1 = conc_p * torch.log(rate_p.data) - torch.lgamma(conc_p.data)
        term2 = (conc_p - 1) * e_log_lambda
        term3 = -rate_p * e_lambda
        e_log_p_bond = term1 + term2 + term3
        return e_log_p_bond.data.sum()

    def _e_log_p_bond_subset(self, bond_tag, indices):
        """Compute e_log_p for only specified indices of a bond."""
        q_bond = self.q_bonds[bond_tag]
        p_bond = self.p_bonds[bond_tag]

        rate_q = q_bond.rate.data[indices] if hasattr(q_bond.rate, "data") else q_bond.rate[indices]
        conc_q = (
            q_bond.concentration.data[indices]
            if hasattr(q_bond.concentration, "data")
            else q_bond.concentration[indices]
        )
        rate_p = p_bond.rate.data[indices] if hasattr(p_bond.rate, "data") else p_bond.rate[indices]
        conc_p = (
            p_bond.concentration.data[indices]
            if hasattr(p_bond.concentration, "data")
            else p_bond.concentration[indices]
        )

        e_lambda = conc_q / rate_q
        e_log_lambda = torch.digamma(conc_q) - torch.log(rate_q)

        term1 = conc_p * torch.log(rate_p) - torch.lgamma(conc_p)
        term2 = (conc_p - 1) * e_log_lambda
        term3 = -rate_p * e_lambda
        return (term1 + term2 + term3).sum()

    def e_log_p_node(self, node_tag):

        node = self.mu[node_tag]
        excluded_indices = self.output_dimensions + [self.batch_dim]

        bond_indices = [ind for ind in node.inds if ind not in excluded_indices]

        node_bonds_e_q = 0
        dims = 1
        for ind in bond_indices:
            bond = self.q_bonds[ind]
            conc = (
                bond.concentration.data
                if isinstance(bond.concentration, qt.Tensor)
                else bond.concentration
            )
            rate = bond.rate.data if isinstance(bond.rate, qt.Tensor) else bond.rate

            if hasattr(self, "_pending_soft_trims") and ind in self._pending_soft_trims:
                indices_to_keep = self._pending_soft_trims[ind]["indices_to_keep"]
                conc = conc[indices_to_keep]
                rate = rate[indices_to_keep]

            e_log_lambda = torch.digamma(conc) - torch.log(rate)
            dims = dims * e_log_lambda.shape[0]
            node_bonds_e_q += e_log_lambda.sum()

        e_log_p_node = 0.5 * node_bonds_e_q
        costant = -0.5 * dims * np.log(2 * np.pi)
        mu_node = self.mu[node_tag] ** 2
        sigma_node = self._unprime_indices_tensor(self.sigma[node_tag])
        partial_mu = self._get_partial_trace(mu_node, node_tag, None)
        partial_sigma = self._get_partial_trace(sigma_node, node_tag, None)
        trace = -0.5 * (partial_mu + partial_sigma)
        return e_log_p_node + costant + trace

    def _get_mu_covariance(self, node_tag):
        mu = self.mu[node_tag]
        sigma = self.sigma[node_tag].copy()
        index_bases = mu.inds
        col_inds = sorted(index_bases)
        row_inds = [i + "_prime" for i in col_inds if i not in self.output_dimensions]
        covariance = sigma.to_dense(col_inds, row_inds)
        mean = mu.to_dense(col_inds)
        return mean, covariance

    def _H_nodes(self):
        h = 0
        for key in self.q_nodes.keys():
            loc, cov = self._get_mu_covariance(key)
            # try:
            node_entropy = dist.MultivariateNormal(loc=loc, covariance_matrix=cov).entropy()
            # except Exception:
            #     return torch.full((), float("nan"), device=cov.device, dtype=cov.dtype)
            h += node_entropy
        return h

    def _H_bonds(self):
        h = 0
        for key in self.q_bonds.keys():
            if hasattr(self, "_pending_soft_trims") and key in self._pending_soft_trims:
                indices_to_keep = self._pending_soft_trims[key]["indices_to_keep"]
                q_bond = self.q_bonds[key]
                conc = (
                    q_bond.concentration.data[indices_to_keep]
                    if hasattr(q_bond.concentration, "data")
                    else q_bond.concentration[indices_to_keep]
                )
                rate = (
                    q_bond.rate.data[indices_to_keep]
                    if hasattr(q_bond.rate, "data")
                    else q_bond.rate[indices_to_keep]
                )
                node_entropy = dist.Gamma(conc, rate).entropy().sum()
            else:
                node_entropy = self.q_bonds[key].forward().entropy().sum()
            h += node_entropy
        return h

    def _H_tau(self):
        """Compute entropy of the tau (noise precision) variational distribution."""
        return self.q_tau.forward().entropy().sum()

    def _compute_e_log_p_tau(self):
        """Compute E_q[log p(tau)] - expected log prior for tau under variational distribution."""
        # Parameters for q (variational)
        rate_q = self.q_tau.rate
        conc_q = self.q_tau.concentration

        # Parameters for p (prior)
        rate_p = self.p_tau.rate
        conc_p = self.p_tau.concentration

        # E_q[tau]
        e_tau = conc_q / rate_q

        # E_q[log tau]
        e_log_tau = torch.digamma(conc_q.data) - torch.log(rate_q.data)

        # Expected log p(tau) under q(tau)
        # log p(tau) = conc_p * log(rate_p) - log(Gamma(conc_p)) + (conc_p - 1) * log(tau) - rate_p * tau
        term1 = conc_p * torch.log(rate_p.data) - torch.lgamma(conc_p.data)
        term2 = (conc_p - 1) * e_log_tau
        term3 = -rate_p * e_tau
        e_log_p_tau = term1 + term2 + term3
        return e_log_p_tau.data.sum()

    def _H_(self):
        h_nodes = self._H_nodes()

        if torch.isnan(h_nodes):
            return h_nodes

        h_bonds = self._H_bonds()
        h_tau = self._H_tau()

        return h_bonds + h_nodes + h_tau

    def _compute_raw_elbo(self, force=False):
        if not force and self.has_pending_soft_trims():
            return getattr(self, "_last_valid_elbo", float("nan"))

        q_entropy = self._H_()

        if torch.isnan(q_entropy):
            return float("nan")

        e_likelyhood = self.compute_expected_log_likelihood()
        e_priors = self._compute_e_log_p()

        elbo = e_likelyhood + e_priors + q_entropy
        result = elbo.detach().item()
        self._last_valid_elbo = result
        return result

    def compute_expected_log_likelihood(self, verbose=False):
        """
        Compute E_q[log p(y|θ,τ)] - the expected log likelihood of the data.

        For Gaussian likelihood p(y|μ,τ) with precision τ:
        log p(y|μ,τ) = -0.5 * τ * ||y - μx||² + 0.5 * log(τ) - 0.5 * log(2π)

        Taking expectation over q(θ,τ):
        E_q[log p(y|θ,τ)] = -0.5 * E[τ] * (MSE + sigma_forward)
                           + 0.5 * N * E_q[log τ]
                           - 0.5 * N * log(2π)

        Where:
        - MSE = Σ_n (y_n - μ_x_n)² (sum over all data points)
        - sigma_forward = forward(sigma) (variance contribution from σ)
        - N = number of data points × output dimension size
        - E[τ] = concentration/rate (mean of Gamma)
        - E_q[log τ] = digamma(concentration) - log(rate) for Gamma distribution

        Returns:
            Scalar value of expected log likelihood
        """
        # Compute MSE: E_q[(y - μx)²]
        mse = self._calc_mu_mse()

        # Compute sigma forward: E_q[forward(σ)]
        sigma_forward = self._calc_sigma_forward()

        # Get E[τ] (mean of q_tau)
        tau_mean = self.q_tau.mean()

        # Get E_q[log τ] for Gamma distribution
        # For Gamma(α, β): E[log X] = digamma(α) - log(β)
        concentration = self.q_tau.concentration
        rate = self.q_tau.rate

        # Extract data if they're quimb tensors and convert to torch tensors
        if isinstance(concentration, qt.Tensor):
            concentration = concentration.data
        elif torch.is_tensor(concentration):
            concentration = (
                concentration.detach().clone().to(dtype=torch.float32, device=self.device)
            )
        else:
            concentration = torch.tensor(concentration, dtype=torch.float32, device=self.device)

        if isinstance(rate, qt.Tensor):
            rate = rate.data
        elif torch.is_tensor(rate):
            rate = rate.detach().clone().to(dtype=torch.float32, device=self.device)
        else:
            rate = torch.tensor(rate, dtype=torch.float32, device=self.device)

        if isinstance(tau_mean, qt.Tensor):
            tau_mean = tau_mean.data

        expected_log_tau = torch.digamma(concentration) - torch.log(rate)

        output_dim_size = 1
        for out_label in self.output_dimensions:
            output_dim_size *= self.mu.ind_size(out_label)

        N = self.data.samples * output_dim_size

        # Compute expected log likelihood
        # E_q[log p(y|θ,τ)] = -0.5 * E[τ] * (MSE + sigma_forward)
        #                    + 0.5 * N * E_q[log τ]
        #                    - 0.5 * N * log(2π)

        likelihood_term1 = -0.5 * tau_mean * (mse + sigma_forward)
        likelihood_term2 = 0.5 * N * expected_log_tau
        likelihood_term3 = -0.5 * N * np.log(2 * np.pi)

        expected_log_likelihood = likelihood_term1 + likelihood_term2 + likelihood_term3

        return expected_log_likelihood

    def compute_elbo(self, verbose=True, print_components=False, relative=True):
        """
        Compute the Evidence Lower BOund (ELBO).

        ELBO = E_q[log p(y|θ,τ)] - KL[q(θ)||p(θ)] - KL[q(τ)||p(τ)]

        When relative=True (default), returns ELBO relative to initial state,
        adjusted for trimming: ELBO() - E_initial - delta_ELBO

        This makes the ELBO comparable across different model sizes after trimming.
        """
        if print_components:
            e_likelyhood = self.compute_expected_log_likelihood()
            e_log_p_nodes = self._compute_e_log_p_nodes()
            e_log_p_bonds = self._compute_e_log_p_bonds()
            e_log_p_tau = self._compute_e_log_p_tau()
            e_priors = e_log_p_nodes + e_log_p_bonds + e_log_p_tau
            h_bonds = self._H_bonds()
            h_nodes = self._H_nodes()
            h_tau = self._H_tau()
            q_entropy = h_bonds + h_nodes + h_tau
            raw_elbo = (e_likelyhood + e_priors + q_entropy).item()

            print(f"\n{' ELBO COMPONENTS ':=^40}")
            print(f"{'Likelihood (E[log p(x|z)]):':<25} {e_likelyhood.item():>12.4f}")
            print(f"{'Priors (E[log p(z)]):':<25} {e_priors.item():>12.4f}")
            print(
                f"  └─ Nodes: {e_log_p_nodes.item():>8.4f} | Bonds: {e_log_p_bonds.item():>8.4f} | Tau: {e_log_p_tau.item():>8.4f}"
            )
            print(f"{'Entropy (H[q(z)]):':<25} {q_entropy.item():>12.4f}")
            print(
                f"  └─ Nodes: {h_nodes.item():>8.4f} | Bonds: {h_bonds.item():>8.4f} | Tau: {h_tau.item():>8.4f}"
            )
            print("-" * 40)
            print(f"{'TOTAL ELBO:':<25} {raw_elbo:>12.4f}")
            print(f"{'=' * 40}\n")
        else:
            raw_elbo = self._compute_raw_elbo()

        if self.elbo_initial is None:
            self.elbo_initial = raw_elbo

        if relative:
            elbo = raw_elbo - self.elbo_initial - self.delta_elbo
        else:
            elbo = raw_elbo

        if verbose:
            print(f"\n{'=' * 60}")
            print(f"ELBO COMPUTATION")
            print(f"{'=' * 60}")
            print(f"Raw ELBO: {raw_elbo:.4f}")
            if relative:
                print(f"Initial ELBO: {self.elbo_initial:.4f}")
                print(f"Delta ELBO (from trimming): {self.delta_elbo:.4f}")
                print(f"Relative ELBO: {elbo:.4f}")
            print(f"{'=' * 60}\n")

        return elbo

    def _copy_data(self, data):
        """Helper to copy data arrays, backend-agnostic."""
        if hasattr(data, "clone"):  # PyTorch
            return data.clone()
        elif hasattr(data, "copy"):  # NumPy, JAX
            return data.copy()
        else:
            # Fallback
            import numpy as np

            return np.array(data)

    def _batch_forward(self, inputs: List[qt.Tensor], tn, output_inds: List[str]) -> qt.Tensor:
        """Helper for forward pass, contracting a single batch of inputs."""

        full_tn = tn & inputs
        res = full_tn.contract(output_inds=output_inds)
        if len(output_inds) > 0:
            res.transpose_(*output_inds)
        return res

    def forward(
        self,
        tn: qt.TensorNetwork,
        input_generator,
        sum_over_batch: bool = False,
        sum_over_output: bool = False,
    ):
        """
        Performs the batched forward pass.

        Args:
            tn: TensorNetwork to contract
            inputs: List of batches, where each batch is a list of input tensors
            sum_over_batch: If True, removes batch_dim from output_inds (sums over it)
            sum_over_output: If True, also removes output_dimensions from output_inds

        Returns:
            Result tensor with appropriate indices based on flags.
        """
        # Determine output indices based on flags
        if sum_over_output:
            if sum_over_batch:
                target_inds = []  # Sum over everything - scalar
            else:
                target_inds = [self.batch_dim]  # Keep only batch dim
        elif sum_over_batch:
            target_inds = self.output_dimensions  # Only keep output dims
        else:
            target_inds = [self.batch_dim] + self.output_dimensions  # Keep both

        if sum_over_batch:
            result = self._sum_over_batches(
                self._batch_forward, input_generator, tn=tn, output_inds=target_inds
            )
        else:
            result = self._concat_over_batches(
                self._batch_forward, input_generator, tn=tn, output_inds=target_inds
            )
        return result

    def get_environment(
        self,
        tn: qt.TensorNetwork,
        target_tag: str,
        input_generator,
        copy: bool = True,
        sum_over_batch: bool = False,
        sum_over_output: bool = False,
    ):
        """
        Calculates the environment for a target tensor over multiple batches.

        Args:
            tn_base: Base TensorNetwork (without inputs)
            target_tag: Tag identifying the tensor to remove
            input_batches: List of batches, where each batch is a list of input tensors
            copy: Whether to copy the network before modification
            sum_over_batch: If True, sums over batch dimension
            sum_over_output: If True, also sums over output dimensions

        Returns:
            Environment tensor. If sum_over_batch=False, concatenates batches.
            If sum_over_batch=True, sums on-the-fly across batches.
        """
        if copy:
            tn_base = tn.copy()
        else:
            tn_base = tn
        if sum_over_batch:
            result = self._sum_over_batches(
                self._batch_environment,
                input_generator,
                tn=tn_base,
                target_tag=target_tag,
                sum_over_batch=sum_over_batch,
                sum_over_output=sum_over_output,
            )
        else:
            result = self._concat_over_batches(
                self._batch_environment,
                input_generator,
                tn=tn_base,
                target_tag=target_tag,
                sum_over_batch=sum_over_batch,
                sum_over_output=sum_over_output,
            )
        return result

    def _batch_environment(
        self,
        inputs,
        tn: qt.TensorNetwork,
        target_tag: str,
        sum_over_batch: bool = False,
        sum_over_output: bool = False,
    ) -> qt.Tensor:
        node_inds = list(tn[target_tag].inds)
        
        env_tn = tn & inputs
        env_tn.delete(target_tag)

        final_env_inds = []
        
        if not sum_over_batch:
            final_env_inds.append(self.batch_dim)
        
        for ind in node_inds:
            if sum_over_output and ind in self.output_dimensions:
                continue
            final_env_inds.append(ind)

        env_tensor = env_tn.contract(output_inds=final_env_inds)

        return env_tensor

    def forward_with_target(
        self,
        input_generator,
        tn: qt.TensorNetwork,
        mode: str = "dot",
        sum_over_batch: bool = False,
        output_inds=[],
    ):
        """
        Forward to pass a dot product or compute MSE.
        The generator should return mu_y or sigma_y
        """
        if sum_over_batch:
            result = self._sum_over_batches(
                self._batch_forward_with_target,
                input_generator,
                tn=tn,
                mode=mode,
                sum_over_batch=sum_over_batch,
                output_inds=output_inds,
            )
        else:
            result = self._concat_over_batches(
                self._batch_forward_with_target,
                input_generator,
                tn=tn,
                mode=mode,
                sum_over_batch=sum_over_batch,
                output_inds=output_inds,
            )
        return result

    def _batch_forward_with_target(
        self,
        inputs: List[qt.Tensor],
        y: qt.Tensor,
        tn: qt.TensorNetwork,
        mode: str = "dot",
        sum_over_batch: bool = False,
        output_inds: List[str] = None,
    ):
        """
        Forward pass coupled with target output y.

        Args:
            tn: TensorNetwork to contract
            inputs: List of input tensors (single batch)
            y: Target output tensor with indices (batch_dim, output_dims...)
            mode: 'dot' for scalar product, 'squared_error' for (forward - y)^2
            sum_over_batch: If True, sums over batch dimension in the result

        Returns:
            Result tensor based on mode:
            - 'dot': scalar product forward · y
            - 'squared_error': (forward - y)^2
        """
        output_inds = [] if output_inds is None else output_inds
        if mode == "dot":
            # Scalar product: add y to the network and contract
            # y has indices (s, y1, y2, ...), forward will match these
            full_tn = tn & inputs & y

            if sum_over_batch:
                # Contract everything (sum over all dims including batch)
                result = full_tn.contract(output_inds=output_inds)
            else:
                # Keep batch dimension
                result = full_tn.contract(output_inds=[self.batch_dim] + output_inds)

            return result

        elif mode == "squared_error":
            # First compute forward using quimb
            target_inds = [self.batch_dim] + self.output_dimensions
            forward_result = self._batch_forward(inputs, tn, output_inds=target_inds)
            forward_result.transpose_(*target_inds)

            # Compute difference using quimb tensor subtraction
            diff = forward_result - y

            # Square it using quimb tensor operations
            squared_diff = diff**2

            # Now we need to sum over output dimensions
            # Create a network with squared_diff and contract over output dims
            if sum_over_batch:
                # Contract all dimensions (sum over batch and outputs)
                result = squared_diff.contract(output_inds=[])
            else:
                # Contract only output dimensions, keep batch
                result = squared_diff.contract(output_inds=[self.batch_dim])

            return result

        else:
            raise ValueError(f"Unknown mode: {mode}. Use 'dot' or 'squared_error'")

    def _concat_batch_results(self, batch_results: List):
        """
        Concatenate batch results along the batch dimension.
        Uses the appropriate backend (numpy, torch, jax) for concatenation.

        Args:
            batch_results: List of tensors or scalars from each batch

        Returns:
            Concatenated result (qt.Tensor or scalar)
        """
        if len(batch_results) == 0:
            raise ValueError("No batch results to concatenate")

        # Check if results are scalars
        first_result = batch_results[0]
        if not isinstance(first_result, qt.Tensor):
            # Scalars - just sum them if needed or stack
            # For scalars, concatenation doesn't make sense, return as is
            return batch_results

        # Get the backend from the data type
        first_data = first_result.data

        # Determine backend and concatenate
        if hasattr(first_data, "__array__"):  # numpy-like
            import numpy as np

            concat_data = np.concatenate([t.data for t in batch_results], axis=0)
        else:
            # Try generic concatenate
            try:
                import numpy as np

                concat_data = np.concatenate([t.data for t in batch_results], axis=0)
            except:
                raise NotImplementedError(
                    f"Concatenation not implemented for backend: {type(first_data)}"
                )

        return qt.Tensor(concat_data, inds=first_result.inds)

    def _concat_over_batches(self, batch_operation, data_iterator, *args, **kwargs):
        """
        Iterates over data_iterator, collects results, and concatenates them.
        """
        results = []

        for batch_idx, batch_data in enumerate(data_iterator):
            # Ensure proper unpacking (handle tuple vs single item)
            inputs = batch_data if isinstance(batch_data, tuple) else (batch_data,)

            # Execute operation
            batch_res = batch_operation(*inputs, *args, **kwargs)
            results.append(batch_res)

        return self._concat_batch_results(results)

    def _sum_over_batches(self, batch_operation, data_iterator, *args, **kwargs) -> qt.Tensor:
        """
        Args:
            batch_operation: Function accepting (batch_idx, *unpacked_data, *args, **kwargs)
            data_iterator: The generator property (e.g., loader.mu_sigma_y_batches)
        """
        result = None

        for batch_data in data_iterator:
            # Ensure data is a tuple for unpacking
            inputs = batch_data if isinstance(batch_data, tuple) else (batch_data,)

            # Unpack inputs into the operation (e.g., mu, sigma, y)
            batch_result = batch_operation(*inputs, *args, **kwargs)

            result = batch_result if result is None else result + batch_result

        return result

    def theta_block_computation(
        self, node_tag: str, exclude_bonds: Optional[List[str]] = None
    ) -> qt.Tensor:
        """
        Compute theta^B(i) for a given node: the outer product of expected bond probabilities.

        From theoretical model:
            θ^B(i) = ⊗_{b ∈ B(i)} E[λ_b]  where E[λ] = α/β

        This creates a tensor representing the expectation of the bond variables (precisions)
        connected to a specific node, excluding output dimensions and optionally other bonds.

        Args:
            node_tag: Tag identifying the node in the tensor network
            exclude_bonds: Optional list of bond indices (labels) to exclude from computation.
                          Output dimensions and batch_dim are always excluded.

        Returns:
            quimb.Tensor with shape matching the node's shape minus excluded dimensions.
            Acts as a diagonal matrix when used in linear algebra operations.

        Example:
            # Node has indices ['a', 'b', 'c', 'out'] with shapes [2, 3, 4, 5]
            # Output dimensions = ['out'], batch_dim = 's'
            # theta = btn.theta_block_computation('node1')
            # Result has indices ['a', 'b', 'c'] with shapes [2, 3, 4]
            # Data is outer product: E[λ_a] ⊗ E[λ_b] ⊗ E[λ_c]
        """
        exclude_bonds = exclude_bonds or []

        # Get the node tensor from mu network
        node = self.mu[node_tag]
        excluded_indices = self.output_dimensions + [self.batch_dim] + exclude_bonds

        # Identify bond indices: all indices except output dims, batch dim, and excluded
        bond_indices = [ind for ind in node.inds if ind not in excluded_indices]

        if len(bond_indices) == 0:
            return qt.Tensor(data=torch.tensor(1.0, device=self.device), inds=(), tags={node_tag})

        bond_means = [self.q_bonds[bond_ind].mean() for bond_ind in bond_indices]
        theta_tn = qt.TensorNetwork(bond_means)

        # Contract to get the outer product (preserves all indices and labels)
        theta = theta_tn.contract()
        return theta

    def _get_concentration_update(self, bond_tag):
        bond_concentration = self.p_bonds[bond_tag].concentration
        num_trainable_nodes, _ = self.count_trainable_nodes_on_bond(bond_tag)
        bond_dim = self.mu.inds_size([bond_tag])
        update = bond_concentration + bond_dim * num_trainable_nodes * 0.5
        return update

    def _get_rate_update(self, bond_tag):
        p_rate = self.p_bonds[bond_tag].rate
        _, tag_trainable_nodes = self.count_trainable_nodes_on_bond(bond_tag)
        update = None
        for node_tags in tag_trainable_nodes:
            node_tag = next(iter(node_tags))
            mu_node = self.mu[node_tag] ** 2
            sigma_node = self._unprime_indices_tensor(self.sigma[node_tag])
            partial_mu = self._get_partial_trace(mu_node, node_tag, bond_tag)
            partial_sigma = self._get_partial_trace(sigma_node, node_tag, bond_tag)
            if update is None:
                update = 0.5 * (partial_mu + partial_sigma)
            else:
                update += 0.5 * (partial_mu + partial_sigma)
        return p_rate + update

    def update_bond(self, bond_tag):
        concentration_update = self._get_concentration_update(bond_tag)
        rate_update = self._get_rate_update(bond_tag)
        self.q_bonds[bond_tag].update_parameters(
            concentration=concentration_update, rate=rate_update
        )
        self._clear_cache()
        return

    def _calc_mu_mse(self, inputs=None, use_cache=True):
        if inputs is None:
            inputs = self.data
            if use_cache:
                cached = self._get_cached("mu_mse")
                if cached is not None:
                    return cached

        mu_mse = self.forward_with_target(inputs.data_mu_y, self.mu, "squared_error", True, [])
        self.mse = mu_mse

        if inputs is self.data and use_cache:
            self._set_cached("mu_mse", mu_mse)
        return mu_mse

    def _calc_sigma_forward(self, inputs=None, use_cache=True):
        if inputs is None:
            inputs = self.data
            if use_cache:
                cached = self._get_cached("sigma_forward")
                if cached is not None:
                    return cached

        sigma_forward = self.forward(self.sigma, inputs.data_sigma, True, True)

        if inputs is self.data and use_cache:
            self._set_cached("sigma_forward", sigma_forward)
        return sigma_forward

    def _get_tau_update(self):
        concentration = self.p_tau.concentration + self.data.samples * 0.5
        mse = self._calc_mu_mse()
        sigma_f = self._calc_sigma_forward()
        rate = self.p_tau.rate + 0.5 * (mse + sigma_f)
        return concentration, rate

    def update_tau(self, min_tau=1e-4):
        concentration, rate = self._get_tau_update()
        max_rate = concentration / min_tau
        rate = torch.clamp(rate, max=max_rate)
        self.q_tau.update_parameters(concentration=concentration, rate=rate)
        self._clear_cache()

    def _get_partial_trace(self, node: qt.Tensor, node_tag, bond_tag):
        output_inds = [] if bond_tag is None else [bond_tag]
        exclude_bonds = None if bond_tag is None else [bond_tag]
        theta = self.theta_block_computation(node_tag, exclude_bonds)
        tn = node & theta
        return tn.contract(output_inds=output_inds)

    def count_trainable_nodes_on_bond(self, bond_ind: str) -> int:
        """
        Count how many trainable nodes share a given bond.

        A trainable node is one that does NOT have the NOT_TRAINABLE_TAG ('NT').

        Args:
            bond_ind: The bond index (label) to check

        Returns:
            Number of trainable nodes that have this bond index

        Example:
            # Bond 'a' is shared by node1 (trainable), node2 (trainable), and input (not trainable)
            # count_trainable_nodes_on_bond('a') returns 2
        """
        # Get tensor IDs that have this bond index using ind_map
        tids = self.mu.ind_map.get(bond_ind, set())
        trainable = [
            self.mu.tensor_map[tid]
            for tid in tids
            if NOT_TRAINABLE_TAG not in self.mu.tensor_map[tid].tags
        ]
        return len(trainable), [t.tags for t in trainable]

    def get_tau_mean(self):
        return self.q_tau.mean()

    def compute_precision_node(
        self,
        node_tag: str,
    ) -> qt.Tensor:

        tau_expectation = self.get_tau_mean()

        sigma_env = self.get_environment(
            tn=self.sigma,
            target_tag=node_tag,
            input_generator=self.data.data_sigma,
            copy=False,
            sum_over_batch=True,
            sum_over_output=True,
        )

        mu_outer_env = self.outer_operation(
            tn=self.mu, node_tag=node_tag, input_generator=self.data.data_mu, sum_over_batches=True
        )

        theta = self.theta_block_computation(
            node_tag=node_tag,
        )

        original_inds = theta.inds  # Get a copy of indices to iterate over

        for old_ind in original_inds:
            primed_ind = old_ind + "_prime"
            # Expand the original index into the desired diagonal pair
            # The original values are placed along the diagonal of (old_ind, primed_ind)
            theta = theta.new_ind_pair_diag(old_ind, old_ind, primed_ind)
        precision = tau_expectation * (sigma_env + mu_outer_env) + theta
        # Broadcast over original output dimension
        # To do after inverting.
        # The inverse of a broadcast is the broadcast of the inverse o.o
        # Just because we only care of the diagonal part of the broadcasted dimension, so It is the same as outer producting with an identity
        return precision

    def _get_sigma_update(self, node_tag):
        block_output_ind = [i for i in self.mu[node_tag].inds if i in self.output_dimensions]
        block_variational_ind = [
            i for i in self.mu[node_tag].inds if i not in self.output_dimensions
        ]
        precision = self.compute_precision_node(node_tag)
        tag = node_tag
        sigma_node = self.invert_ordered_tensor(precision, block_variational_ind, tag=tag)
        for i in block_output_ind:
            sigma_node.new_ind(i, self.mu.ind_size(i))
        return sigma_node

    def _get_mu_update(self, node_tag, debug=False):
        mu_idx = self.mu[node_tag].inds
        self.mu.delete(node_tag)
        rhs = self.forward_with_target(
            self.data.data_mu_y, self.mu, "dot", sum_over_batch=True, output_inds=mu_idx
        )
        relabel_map = {}
        for ind in mu_idx:
            if ind in self.output_dimensions:
                pass
            else:
                relabel_map[ind] = ind + "_prime"
        rhs = rhs.reindex(relabel_map)
        rhs = rhs * self.q_tau.mean()
        tn = rhs & self.sigma[node_tag]

        mu_update = tn.contract(output_inds=mu_idx)
        mu_update.modify(tags=[node_tag])
        if debug:
            return mu_update, rhs, self.sigma[node_tag]
        return mu_update

    def update_sigma_node(self, node_tag):
        sigma_update = self._get_sigma_update(node_tag)
        self.update_node(self.sigma, sigma_update, node_tag)
        self._clear_cache()
        return

    def update_mu_node(self, node_tag):
        mu_update = self._get_mu_update(node_tag)
        self.update_node(self.mu, mu_update, node_tag)
        self._clear_cache()
        return

    def trim_zero_variance(self, verbose=False, threshold=1e-10):
        """
        Trim bond dimensions with near-zero variance (diagonal covariance elements).
        
        This is a convenience wrapper around trim_bonds_by_variance with a very
        small absolute threshold to remove numerically unstable dimensions.
        
        Args:
            verbose: Print trimming info
            threshold: Absolute variance threshold (dimensions below this are trimmed)
            
        Returns:
            Total dimensions trimmed
        """
        return self.trim_bonds_by_variance(
            threshold=threshold, relative=False, verbose=verbose
        )

    def update_node(self, tn, tensor, node_tag):
        """
        Updates the tensor network by replacing the node with the given tag
        with the new tensor.
        """

        if node_tag in tn.tag_map:
            tn.delete(node_tag)

        tn.add_tensor(tensor)

    def get_backend(self, data):
        module = type(data).__module__
        if "torch" in module:
            return "torch", importlib.import_module("torch")
        elif "jax" in module:
            return "jax", importlib.import_module("jax.numpy")
        elif "numpy":
            return "numpy", np

    def _apply_cholesky_inv(self, matrix, lib, backend_name):
        if backend_name == "torch":
            return lib.cholesky_inverse(lib.linalg.cholesky(matrix))

        if backend_name == "jax":
            # Solve LX = I for stability
            return lib.linalg.solve(matrix, lib.eye(matrix.shape[0], dtype=matrix.dtype))

        # Numpy / Default Cholesky
        L = lib.linalg.cholesky(matrix)
        inv_L = lib.linalg.inv(L)
        return inv_L.T.conj() @ inv_L

    def _apply_general_inv(self, matrix, lib, backend_name):
        if backend_name in ("torch", "jax", "numpy"):
            return lib.linalg.inv(matrix)
        raise ValueError(f"Backend '{backend_name}' not supported for general inversion.")

    def invert_ordered_tensor(self, tensor, index_bases, method="cholesky", tag=None):
        tag = tag or tensor.tags
        col_inds = sorted(index_bases)
        row_inds = [i + "_prime" for i in col_inds]

        matrix_data = tensor.to_dense(col_inds, row_inds)
        backend_name, lib = self.get_backend(matrix_data)

        if method == "cholesky":
            inv_data = self._apply_cholesky_inv(matrix_data, lib, backend_name)

        if method != "cholesky":
            inv_data = self._apply_general_inv(matrix_data, lib, backend_name)

        new_shape = tuple(tensor.ind_size(i) for i in col_inds + row_inds)
        return qt.Tensor(data=inv_data.reshape(new_shape), inds=col_inds + row_inds, tags=tag)

    def outer_operation(self, input_generator, tn, node_tag, sum_over_batches):
        if sum_over_batches:
            result = self._sum_over_batches(
                self._batch_outer_operation,
                input_generator,
                tn=tn,
                node_tag=node_tag,
                sum_over_batches=sum_over_batches,
            )
        else:
            result = self._concat_over_batches(
                self._batch_outer_operation,
                input_generator,
                tn=tn,
                node_tag=node_tag,
                sum_over_batches=sum_over_batches,
            )
        return result

    def _batch_outer_operation(self, inputs, tn, node_tag, sum_over_batches: bool):
        """
        Compute the outer product of mu environment with itself, summed over batches:
        Σ_n (T_i μ x_n) ⊗ (T_i μ x_n)

        This is used in computing the precision update for variational inference.
        The environment is computed batch-by-batch, outer product computed, then summed.

        From theoretical model (Step 1, node update):
            Part of Σ⁻¹ computation requires: Σ_n (T_i μ x_n) ⊗ (T_i μ x_n)

        Args:
            node_tag: Tag identifying the node to exclude from environment
            inputs: Input data prepared with prepare_inputs (list of tensors with batch dim)

        Returns:
            Tensor representing Σ_n (T_i μ x_n) ⊗ (T_i μ x_n), with indices being
            the bonds of the node and their primed versions.
            Shape: (bond_a, bond_b, ..., bond_a_prime, bond_b_prime, ...)
        """

        env = self._batch_environment(
            inputs, tn, target_tag=node_tag, sum_over_batch=False, sum_over_output=True
        )

        sample_dim = [self.batch_dim] if not sum_over_batches else []
        # Prime indices (exclude output)
        env_prime = self._prime_indices_tensor(
            env, exclude_indices=self.output_dimensions + [self.batch_dim]
        )
        # Outer product via tensor network (sums over shared output indices)

        env_inds = env.inds + env_prime.inds
        outer_tn = env & env_prime
        out_indices = sample_dim + [i for i in env_inds if i not in [self.batch_dim]]
        batch_result = outer_tn.contract(output_inds=out_indices)
        return batch_result

    def _prime_indices_tensor(
        self,
        tensor: qt.Tensor,
        exclude_indices: Optional[List[str]] = None,
        prime_suffix: str = "_prime",
    ) -> qt.Tensor:
        """
        Helper to prime indices of any tensor (not just from network).

        Args:
            tensor: The tensor to prime
            exclude_indices: Indices to NOT prime
            prime_suffix: Suffix to add (default "_prime")

        Returns:
            New tensor with primed indices
        """
        exclude_indices = exclude_indices or []

        reindex_map = {
            ind: f"{ind}{prime_suffix}" for ind in tensor.inds if ind not in exclude_indices
        }

        return tensor.reindex(reindex_map)

    def _unprime_indices_tensor(self, tensor: qt.Tensor, prime_suffix: str = "_prime") -> qt.Tensor:

        reindex_map = {
            ind: ind[: -len(prime_suffix)] for ind in tensor.inds if ind.endswith(prime_suffix)
        }
        return tensor.reindex(reindex_map)

    def fancy_isel(self, tensor, index_dict):
        """Slice tensor along specified indices, modifying in-place."""
        new_data = tensor.data

        for index_name, keep_indices in index_dict.items():
            try:
                axis_idx = tensor.inds.index(index_name)
            except ValueError:
                raise ValueError(f"Index '{index_name}' not found in tensor indices: {tensor.inds}")

            slicer = [slice(None)] * new_data.ndim
            slicer[axis_idx] = keep_indices
            new_data = new_data[tuple(slicer)]

        tensor.modify(data=new_data)
        return tensor

    def _apply_bond_trim(self, bond_tag, indices_to_keep):
        """Apply trimming to a bond given the indices to keep."""
        _, nodes = self.count_trainable_nodes_on_bond(bond_tag)

        for node_tags in nodes:
            node_tag = next(iter(node_tags))
            self.fancy_isel(self.mu[node_tag], {bond_tag: indices_to_keep})
            self.fancy_isel(
                self.sigma[node_tag],
                {bond_tag: indices_to_keep, f"{bond_tag}_prime": indices_to_keep},
            )

        if bond_tag in self.p_bonds:
            self.p_bonds[bond_tag].trim(indices_to_keep)
        if bond_tag in self.q_bonds:
            self.q_bonds[bond_tag].trim(indices_to_keep)

        self._trim_input_data(bond_tag, indices_to_keep)

    def _trim_bond(self, bond_tag, verbose=False):
        """
        Trim bond dimensions based on relevance threshold.
        Relevance = 1/E[λ]. Removes dimensions where relevance < threshold.
        """
        precision = self.q_bonds[bond_tag].mean().data
        relevance = 1.0 / precision
        threshold = self.threshold
        
        indices_to_keep = torch.where(relevance >= threshold)[0].tolist()

        if len(indices_to_keep) == len(relevance):
            return 0

        if len(indices_to_keep) == 0:
            indices_to_keep = [relevance.argmax().item()]
            if verbose:
                print(f"  {bond_tag}: all below threshold, keeping max relevance")

        self._apply_bond_trim(bond_tag, indices_to_keep)

        if verbose:
            print(
                f"  {bond_tag}: kept {len(indices_to_keep)}/{len(relevance)} dims "
                f"(threshold={threshold:.4f}, min_1/λ={relevance.min():.4f}, max_1/λ={relevance.max():.4f})"
            )

        return len(relevance) - len(indices_to_keep)

    def trim_bonds(self, verbose=False):
        """Trim all bonds based on relevance threshold, returns total dimensions trimmed."""
        elbo_before = self._compute_raw_elbo()

        excluded = set(self.output_dimensions) | set(self.input_indices)
        bonds_to_trim = [b for b in self.q_bonds if b not in excluded]

        total_trimmed = 0
        for bond in bonds_to_trim:
            trimmed = self._trim_bond(bond, verbose=verbose)
            total_trimmed += trimmed

        if total_trimmed > 0:
            self._clear_cache()

            elbo_after = self._compute_raw_elbo()
            self.delta_elbo += elbo_after - elbo_before
            if verbose:
                print(f"Delta ELBO from trimming: {elbo_after - elbo_before:.4f}")

        return total_trimmed

    def _prepare_soft_trim_bond(self, bond_tag, high_precision=1e8, verbose=False):
        """
        Prepare bond for soft trimming by setting high precision on to-be-trimmed dimensions.

        Instead of immediately removing dimensions, sets their precision very high
        so neighboring nodes can adjust before the actual trim.
        """
        precision = self.q_bonds[bond_tag].mean().data
        threshold = self.threshold

        relevance = 1.0 / precision

        indices_to_keep = torch.where(relevance >= threshold)[0].tolist()
        
        if len(indices_to_keep) == 0:
            indices_to_keep = [relevance.argmax().item()]

        all_indices = set(range(len(precision)))
        indices_to_trim = list(all_indices - set(indices_to_keep))

        if len(indices_to_trim) == 0:
            return 0

        if not hasattr(self, "_pending_soft_trims"):
            self._pending_soft_trims = {}

        self._pending_soft_trims[bond_tag] = {
            "indices_to_keep": indices_to_keep,
            "indices_to_trim": indices_to_trim,
        }

        q_bond = self.q_bonds[bond_tag]
        if hasattr(q_bond.concentration, "data"):
            conc_data = q_bond.concentration.data.clone()
            rate_data = q_bond.rate.data.clone()
            conc_data[indices_to_trim] = high_precision
            rate_data[indices_to_trim] = 1.0
            q_bond.concentration.modify(data=conc_data)
            q_bond.rate.modify(data=rate_data)
        else:
            q_bond.concentration = q_bond.concentration.clone()
            q_bond.rate = q_bond.rate.clone()
            q_bond.concentration[indices_to_trim] = high_precision
            q_bond.rate[indices_to_trim] = 1.0
        q_bond._distribution = None

        if verbose:
            print(
                f"Bond {bond_tag}: marked {len(indices_to_trim)} dims for soft trim (precision={high_precision})"
            )

        return len(indices_to_trim)

    def prepare_soft_trim_bonds(self, high_precision=1e8, verbose=False):
        """Prepare all bonds for soft trimming. Returns total dimensions marked."""
        elbo_before = self._compute_raw_elbo()

        excluded = set(self.output_dimensions)
        bonds_to_trim = [b for b in self.q_bonds if b not in excluded]

        total_marked = 0
        for bond in bonds_to_trim:
            marked = self._prepare_soft_trim_bond(
                bond, high_precision=high_precision, verbose=verbose
            )
            total_marked += marked

        if total_marked > 0:
            self._soft_trim_elbo_before = elbo_before
            self._clear_cache()

        return total_marked

    def _finalize_soft_trim_bond(self, bond_tag, verbose=False):
        """Finalize soft trim by actually removing the marked dimensions."""
        if not hasattr(self, "_pending_soft_trims"):
            return 0

        if bond_tag not in self._pending_soft_trims:
            return 0

        trim_info = self._pending_soft_trims.pop(bond_tag)
        indices_to_keep = trim_info["indices_to_keep"]
        n_trimmed = len(trim_info["indices_to_trim"])

        if n_trimmed == 0:
            return 0

        _, nodes = self.count_trainable_nodes_on_bond(bond_tag)

        for node in nodes:
            node_tag = next(iter(node))
            self.fancy_isel(self.mu[node_tag], {f"{bond_tag}": indices_to_keep})
            self.fancy_isel(
                self.sigma[node_tag],
                {f"{bond_tag}": indices_to_keep, f"{bond_tag}_prime": indices_to_keep},
            )

        if bond_tag in self.p_bonds:
            self.p_bonds[bond_tag].trim(indices_to_keep)

        if bond_tag in self.q_bonds:
            self.q_bonds[bond_tag].trim(indices_to_keep)

        self._trim_input_data(bond_tag, indices_to_keep)

        if verbose:
            orig_size = len(indices_to_keep) + n_trimmed
            print(
                f"Bond {bond_tag}: finalized trim, kept {len(indices_to_keep)}/{orig_size} dimensions"
            )

        return n_trimmed

    def finalize_soft_trim_bonds(self, verbose=False):
        """Finalize all pending soft trims. Returns total dimensions trimmed."""
        if not hasattr(self, "_pending_soft_trims") or not self._pending_soft_trims:
            return 0

        elbo_before = getattr(self, "_soft_trim_elbo_before", self._compute_raw_elbo())

        bonds_to_finalize = list(self._pending_soft_trims.keys())
        total_trimmed = 0
        for bond in bonds_to_finalize:
            trimmed = self._finalize_soft_trim_bond(bond, verbose=verbose)
            total_trimmed += trimmed

        if total_trimmed > 0:
            self._clear_cache()
            elbo_after = self._compute_raw_elbo()
            self.delta_elbo += elbo_after - elbo_before
            if hasattr(self, "_soft_trim_elbo_before"):
                del self._soft_trim_elbo_before
            if verbose:
                print(f"Delta ELBO from soft trim: {elbo_after - elbo_before:.4f}")

        return total_trimmed

    def has_pending_soft_trims(self):
        """Check if there are any pending soft trims."""
        return hasattr(self, "_pending_soft_trims") and bool(self._pending_soft_trims)

    def get_soft_trim_excluded_bonds(self):
        """Get list of bonds currently in soft trim state (should not be updated)."""
        if not hasattr(self, "_pending_soft_trims"):
            return []
        return list(self._pending_soft_trims.keys())

    def _get_bond_variances(self, node_tag, bond_tag, debug=False):
        sigma = self.sigma[node_tag]
        return self._get_bond_variances_from_tensor(sigma, bond_tag, debug=debug)

    def _get_bond_variances_from_tensor(self, sigma, bond_tag, debug=False):
        p = f"{bond_tag}_prime"
        others = [i for i in sigma.inds if not i.endswith('_prime') and i != bond_tag]
        others_p = [i for i in sigma.inds if i.endswith('_prime') and i != p]
        
        # Quimb isolation
        matrix = sigma.to_dense(others + [bond_tag], others_p + [p])
        joint_diag = torch.diag(matrix)
        
        # Using Quimb ind_size for all calculations
        other_size = 1
        for i in others:
            other_size *= sigma.ind_size(i)
        bond_size = sigma.ind_size(bond_tag)
        
        return joint_diag.view(other_size, bond_size)[0, :]


    def _trim_bond_by_variance(self, bond_tag, threshold=1e-10, relative=False, verbose=False):
        _, nodes = self.count_trainable_nodes_on_bond(bond_tag)
        if not nodes:
            return 0

        total_variance = None
        for node_tags in nodes:
            node_tag = next(iter(node_tags))
            variance = self._get_bond_variances(node_tag, bond_tag)
            if total_variance is None:
                total_variance = variance
            else:
                total_variance = total_variance + variance

        total_variance = total_variance / len(nodes)

        if relative:
            effective_threshold = threshold * total_variance.max()
        else:
            effective_threshold = threshold

        indices_to_keep = torch.where(total_variance >= effective_threshold)[0].tolist()

        if len(indices_to_keep) == len(total_variance):
            return 0

        if len(indices_to_keep) == 0:
            indices_to_keep = [total_variance.argmax().item()]
            if verbose:
                print(f"  {bond_tag}: all below threshold, keeping max")

        self._apply_bond_trim(bond_tag, indices_to_keep)

        trimmed = len(total_variance) - len(indices_to_keep)
        if verbose:
            print(
                f"  {bond_tag}: kept {len(indices_to_keep)}/{len(total_variance)} (threshold: {effective_threshold:.2e})"
            )

        return trimmed

    def trim_bonds_by_variance(self, threshold=1e-10, relative=False, verbose=False):
        elbo_before = self._compute_raw_elbo()
        
        excluded = set(self.output_dimensions) | set(self.input_indices)
        bonds_to_trim = [b for b in self.q_bonds if b not in excluded]

        total_trimmed = 0
        for bond in bonds_to_trim:
            trimmed = self._trim_bond_by_variance(bond, threshold, relative, verbose)
            total_trimmed += trimmed
        if verbose:
            print(f"total tirmmed = {total_trimmed}")

        if total_trimmed > 0:
            self._clear_cache()
            for node_tag in self.mu.tags:
                self.update_sigma_node(node_tag)

            elbo_after = self._compute_raw_elbo()
            self.delta_elbo += elbo_after - elbo_before
            if verbose:
                print(f"Delta ELBO from variance trimming: {elbo_after - elbo_before:.4f}")

        return total_trimmed

    def _get_effective_parameters(self, bond_tag, debug=False):
        _, nodes = self.count_trainable_nodes_on_bond(bond_tag)
        if not nodes:
            return None

        if debug:
            print(f"\n{'='*60}")
            print(f"DEBUG: _get_effective_parameters for {bond_tag}")
            print(f"{'='*60}")
            q_bond = self.q_bonds[bond_tag]
            e_lambda = q_bond.mean().data
            print(f"  E[λ] shape: {e_lambda.shape}")
            print(f"  E[λ] = conc/rate: {e_lambda.tolist()}")
            print(f"  E[λ] stats: min={e_lambda.min():.6f}, max={e_lambda.max():.6f}, mean={e_lambda.mean():.6f}")
            print(f"  Number of nodes: {len(nodes)}")

        gamma_right = None
        for idx, node_tags in enumerate(nodes):
            node_tag = next(iter(node_tags))
            
            if debug:
                print(f"\n  --- Node {idx}: {node_tag} ---")
                # Show raw Sigma diagonal for this bond (before theta multiplication)
                sigma_diag_raw = self._get_bond_variances(node_tag, bond_tag)
                print(f"  Σ_ii (raw variance): min={sigma_diag_raw.min():.6f}, max={sigma_diag_raw.max():.6f}, mean={sigma_diag_raw.mean():.6f}")
                print(f"  Σ_ii values: {sigma_diag_raw.tolist()}")
            
            theta = self.theta_block_computation(node_tag)
            
            if debug:
                print(f"  θ shape: {theta.shape}, inds: {theta.inds}")
                # Extract diagonal of theta for this bond to see E[λ] contribution
                theta_data = theta.data
                print(f"  θ (E[λ] outer product) stats: min={theta_data.min():.6f}, max={theta_data.max():.6f}")
            
            original_inds = theta.inds  # Get a copy of indices to iterate over
            for old_ind in original_inds:
                primed_ind = old_ind + "_prime"
                theta = theta.new_ind_pair_diag(old_ind, old_ind, primed_ind)
            
            variance = self.sigma[node_tag] * theta
            total_variance = self._get_bond_variances_from_tensor(variance, bond_tag, debug=False)
            
            if debug:
                print(f"  (θ * Σ)_ii: min={total_variance.min():.6f}, max={total_variance.max():.6f}, mean={total_variance.mean():.6f}")
                print(f"  (θ * Σ)_ii values: {total_variance.tolist()}")
            
            if gamma_right is None:
                gamma_right = total_variance
            else:
                gamma_right = total_variance + gamma_right

        if debug:
            print(f"\n  --- Aggregation ---")
            print(f"  Sum of (θ * Σ)_ii across {len(nodes)} nodes: {gamma_right.tolist()}")
            print(f"  Sum stats: min={gamma_right.min():.6f}, max={gamma_right.max():.6f}, mean={gamma_right.mean():.6f}")
            avg = gamma_right / len(nodes)
            print(f"  Averaged (÷{len(nodes)}): min={avg.min():.6f}, max={avg.max():.6f}, mean={avg.mean():.6f}")
        
        gamma = 1 - gamma_right / len(nodes)
        
        if debug:
            print(f"\n  --- Final γ = 1 - avg(θ * Σ)_ii ---")
            print(f"  γ values: {gamma.tolist()}")
            print(f"  γ stats: min={gamma.min():.6f}, max={gamma.max():.6f}, mean={gamma.mean():.6f}")
            # Show how many would be kept at various thresholds
            for thresh in [0.1, 0.5, 0.7, 0.9]:
                kept = (gamma >= thresh).sum().item()
                print(f"  Threshold {thresh}: would keep {kept}/{len(gamma)} dims")
        
        return gamma

    def _trim_bond_by_gamma(self, bond_tag, threshold=0.1, verbose=False):
        """
        Trim bond dimensions where effective parameter gamma_i < threshold.
        gamma_i = 1 - (theta * Sigma)_ii. When close to 0, dimension doesn't contribute.
        """
        gamma = self._get_effective_parameters(bond_tag, debug=verbose)
        if verbose:
            print(f"  {bond_tag}: γ mean={gamma.mean():.4f}, min={gamma.min():.4f}, max={gamma.max():.4f}")

        assert gamma.min() >= 0, f"ERROR: gamma smaller than 0: {gamma}"

        indices_to_keep = torch.where(gamma >= threshold)[0].tolist()

        if len(indices_to_keep) == len(gamma):
            return 0

        if len(indices_to_keep) == 0:
            indices_to_keep = [gamma.argmax().item()]
            if verbose:
                print(f"  {bond_tag}: all gamma below threshold, keeping max gamma")

        self._apply_bond_trim(bond_tag, indices_to_keep)

        trimmed = len(gamma) - len(indices_to_keep)
        if verbose:
            print(
                f"  {bond_tag}: kept {len(indices_to_keep)}/{len(gamma)} dims "
                f"(threshold={threshold:.2f}, min_γ={gamma.min():.4f}, max_γ={gamma.max():.4f})"
            )

        return trimmed

    def trim_bonds_by_gamma(self, threshold=0.1, verbose=False):
        """
        Trim bonds based on effective parameter gamma_i = 1 - E[lambda_i] * Sigma_ii.
        
        Prunes dimensions where gamma_i < threshold (not contributing effectively).
        
        Args:
            threshold: Minimum gamma value to keep (default 0.1)
            verbose: Print trimming details
            
        Returns:
            Total number of dimensions trimmed
        """
        elbo_before = self._compute_raw_elbo()

        excluded = set(self.output_dimensions) | set(self.input_indices)
        bonds_to_trim = [b for b in self.q_bonds if b not in excluded]

        total_trimmed = 0
        for bond in bonds_to_trim:
            trimmed = self._trim_bond_by_gamma(bond, threshold, verbose)
            total_trimmed += trimmed

        if verbose:
            print(f"Total trimmed by gamma: {total_trimmed}")

        if total_trimmed > 0:
            self._clear_cache()
            for node_tag in self.mu.tags:
                self.update_sigma_node(node_tag)

            elbo_after = self._compute_raw_elbo()
            self.delta_elbo += elbo_after - elbo_before
            if verbose:
                print(f"Delta ELBO from gamma trimming: {elbo_after - elbo_before:.4f}")

        return total_trimmed

    def _to_torch(self, tensor, requires_grad=False):
        data = tensor.data if isinstance(tensor, qt.Tensor) else tensor
        if not torch.is_tensor(data):
            data = torch.from_numpy(data)
        if not data.is_floating_point():
            data = data.float()
        if requires_grad:
            data.requires_grad_(True)
        return data

    def _batch_evaluate(self, inputs, y_true, metrics):
        with torch.no_grad():
            output_inds = [self.batch_dim] + self.output_dimensions
            y_pred = self._batch_forward(inputs, self.mu, output_inds)

            y_pred_th = self._to_torch(y_pred)
            y_true_th = self._to_torch(y_true)
            if y_pred_th.numel() == y_true_th.numel():
                if y_pred_th.shape != y_true_th.shape:
                    y_pred_th = y_pred_th.view_as(y_true_th)

            results = {}
            for name, func in metrics.items():
                val = func(y_pred_th, y_true_th)
                results[name] = val

        return results

    def evaluate(self, metrics, data_stream=None, verbose=False):
        """
        Evaluate model on dataset using provided metrics.

        Args:
            metrics: Dict mapping name -> callable(y_pred, y_true)
            data_stream: Optional Inputs object. If None, uses self.data
            verbose: If True, print results

        Returns:
            Dict of computed metric values
        """
        if data_stream is None:
            data_stream = self.data

        aggregates = {}

        for i, batch_data in enumerate(data_stream.data_mu_y):
            inputs, y_true = batch_data
            batch_results = self._batch_evaluate(inputs, y_true, metrics)

            if i == 0:
                aggregates = batch_results
            else:
                for name, res in batch_results.items():
                    if isinstance(res, tuple):
                        aggregates[name] = tuple(a + b for a, b in zip(aggregates[name], res))
                    else:
                        aggregates[name] += res

        final_scores = {}
        for name, val in aggregates.items():
            if isinstance(val, (tuple, list)) and len(val) == 2:
                numerator, denominator = val
                if torch.is_tensor(denominator):
                    denominator = denominator.item()
                if denominator != 0:
                    final_scores[name] = (
                        (numerator / denominator).item()
                        if torch.is_tensor(numerator)
                        else (numerator / denominator)
                    )
                else:
                    final_scores[name] = 0.0
            elif isinstance(val, tuple) and len(val) == 4:
                final_scores[name] = val
            else:
                final_scores[name] = val.item() if torch.is_tensor(val) else val

        if verbose:
            print(f"Evaluation: {final_scores}")

        return final_scores

    def _copy_model_state(self):
        """Copy current model state (mu, sigma, distributions) for restoration."""
        return {
            "mu": self.mu.copy(),
            "sigma": self.sigma.copy(),
            "q_tau": (self.q_tau.concentration, self.q_tau.rate),
            "q_bonds": {k: (v.concentration, v.rate) for k, v in self.q_bonds.items()},
        }

    def _restore_model_state(self, state):
        """Restore model state from saved copy."""
        self.mu = state["mu"]
        self.sigma = state["sigma"]
        self.q_tau.update_parameters(*state["q_tau"])
        for k, (c, r) in state["q_bonds"].items():
            self.q_bonds[k].update_parameters(concentration=c, rate=r)

    def debug_print(self, node_tag=None, bond_tag=None):
        bonds = (
            [i for i in self.mu.ind_map if i not in self.output_dimensions]
            if bond_tag is None
            else [bond_tag]
        )
        nodes = list(self.mu.tag_map.keys()) if node_tag is None else [node_tag]

        print("=" * 30)
        print(" CONCENTRATIONS ".center(30, "="))

        for b in bonds:
            c = self.q_bonds[b].concentration
            r = self.q_bonds[b].rate
            print(
                f"{b}: conc={np.array2string(c.data, precision=3, separator=',')}",
                f"rate={np.array2string(r.data, precision=3, separator=',')}",
            )

        print("=" * 30)
        print(" MU & SIGMA SUMS ".center(30, "="))

        for n in nodes:
            mu_val = self.mu[n].data.sum()
            sigma_val = self.sigma[n].data.sum()
            print(f"{n}: mu={mu_val:.3f}, sigma={sigma_val:.3f}")

        print("=" * 30)

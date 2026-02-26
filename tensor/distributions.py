# type: ignore
import torch
import torch.distributions as dist
import quimb.tensor as qt
import numpy as np
from typing import List, Union, Optional, Any
torch.set_default_dtype(torch.float32)   # or torch.float64


def _extract_data(param: Any, dtype=torch.float64) -> torch.Tensor:
    """Helper to extract torch tensor data from potential quimb.Tensor."""
    if isinstance(param, qt.Tensor):
        # We assume the data inside quimb tensor might already be torch or numpy
        data = param.data
        if isinstance(data, np.ndarray):
            return torch.tensor(data, dtype=dtype)
        elif isinstance(data, torch.Tensor):
            return data.to(dtype=dtype)
        return torch.tensor(data, dtype=dtype)
    elif isinstance(param, np.ndarray):
        return torch.tensor(param, dtype=dtype)
    elif isinstance(param, torch.Tensor):
        return param.to(dtype=dtype)
    else:
        return torch.tensor(param, dtype=dtype)

def _wrap_if_quimb(data: torch.Tensor, reference: Any, backend: str) -> Any:
    """
    Wraps the result in a quimb.Tensor if the reference was a quimb.Tensor.
    Preserves indices.
    """
    data = to_backend(data, backend)
    if isinstance(reference, qt.Tensor):
        # We store the torch tensor (potentially with grads) directly in the quimb tensor data
        return qt.Tensor(data=data, inds=reference.inds, tags=reference.tags)
    return data

def to_backend(data, backend: str):
    if backend == 'torch':
        return data

    arr = data.detach().cpu().numpy() if hasattr(data, 'detach') else data

    if backend == 'numpy':
        return arr
    
    raise ValueError(f"Unsupported backend: {backend}")

class GammaDistribution:
    """
    Gamma distribution wrapper for PyTorch distributions.
    Supports quimb.Tensor parameters and returns expectations as quimb.Tensors.
    """
    
    def __init__(self, concentration, rate, backend = 'torch'):
        """
        Initialize Gamma distribution.
        
        Args:
            concentration: Shape parameter (alpha). Can be quimb.Tensor.
            rate: Rate parameter (beta). Can be quimb.Tensor.
        """
        self.backend = backend
        self.concentration = concentration
        self.rate = rate
        self._distribution = None
    
    def update_parameters(self, concentration=None, rate=None):
        if concentration is not None:
            self.concentration = concentration
        if rate is not None:
            self.rate = rate
        self._distribution = None
    
    def forward(self):
        if self._distribution is None:
            c_data = _extract_data(self.concentration)
            r_data = _extract_data(self.rate)
            self._distribution = dist.Gamma(c_data, r_data)
        return self._distribution
    
    def mean(self):
        """
        Returns E[X] = alpha / beta.
        If inputs are quimb.Tensors, returns a quimb.Tensor with the same bond indices.
        """
        c_data = _extract_data(self.concentration)
        r_data = _extract_data(self.rate)
        res = c_data / r_data
        if isinstance(res, qt.Tensor):
            res = res.data
        return _wrap_if_quimb(res, self.concentration, self.backend)

    def trim(self, indices):
        """
        Keeps only the specified indices of the parameters.
        Args:
            indices: List or array of integers to keep.
        """
        def _slice_param(param, idxs):
            # Case A: It is a quimb.Tensor
            if hasattr(param, 'modify'): 
                # Slice the underlying data directly
                # We assume param is 1D (vector of weights), so slicing axis 0 is correct.
                new_data = param.data[idxs]
                param.modify(data=new_data)
                return param
            # Case B: It is a raw Torch/NumPy tensor
            else:
                return param[idxs]

        self.concentration = _slice_param(self.concentration, indices)
        self.rate = _slice_param(self.rate, indices)
        
        # Reset the cached distribution so it regenerates with new shape next time
        self._distribution = None

    def to(self, device):
        def _move(param):
            if hasattr(param, 'apply_to_arrays'):
                param.apply_to_arrays(lambda x: x.to(device) if hasattr(x, 'to') else x)
                return param
            elif hasattr(param, 'to'):
                return param.to(device)
            return param
        self.concentration = _move(self.concentration)
        self.rate = _move(self.rate)
        self._distribution = None
        return self


class MultivariateGaussianDistribution:
    """
    Multivariate Gaussian (Normal) distribution wrapper.
    Supports quimb.Tensor parameters.
    """
    
    def __init__(self, loc, covariance_matrix=None, scale_tril=None, backend='torch'):
        self.loc = loc
        self.covariance_matrix = covariance_matrix
        self.scale_tril = scale_tril
        self._distribution = None
    
    def update_parameters(self, loc=None, covariance_matrix=None, scale_tril=None):
        if loc is not None:
            self.loc = loc
        if covariance_matrix is not None:
            self.covariance_matrix = covariance_matrix
        if scale_tril is not None:
            self.scale_tril = scale_tril
        self._distribution = None
    
    def _get_params(self):
        loc_data = _extract_data(self.loc)
        if loc_data.ndim > 1:
            loc_data = loc_data.flatten()
            
        cov_data = None
        scale_data = None
        
        if self.covariance_matrix is not None:
            cov_data = _extract_data(self.covariance_matrix)
            if cov_data.ndim != 2:
                d = loc_data.shape[0]
                cov_data = cov_data.reshape(d, d)
                
        if self.scale_tril is not None:
            scale_data = _extract_data(self.scale_tril)
            if scale_data.ndim != 2:
                d = loc_data.shape[0]
                scale_data = scale_data.reshape(d, d)
                
        return loc_data, cov_data, scale_data
    
    def forward(self, loc=None, cov=None, scale=None):
        """
        Create and return a PyTorch MultivariateNormal distribution.
        
        Args:
            loc: Mean vector (if None, uses self.loc.data)
            cov: Covariance matrix
            scale: Scale (lower triangular) matrix (not currently used)
        """
        if loc is None:
            loc = self.loc.data
        return dist.MultivariateNormal(loc=loc, covariance_matrix=cov)
    
    def mean(self):
        """Returns the mean. If loc is a quimb.Tensor, returns it directly."""
        return self.loc




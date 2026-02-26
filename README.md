# BTNR: Bayesian Tensor Networks for Regression

## Project Overview
BTNR is a library for Bayesian Tensor Networks (BTN) designed for regression tasks. It leverages the `quimb` library for efficient tensor operations and provides a framework for running experiments with various tensor network architectures, including Bayesian Tensor Networks and Tensor Network Alternating Least Squares (TN-ALS).

## Project Structure
```
./
├── experiments/           # Experiment runners and configs
│   ├── configs/          # JSON configuration files
│   ├── run_grid_search_btn.py   # BTN grid search runner
│   ├── run_grid_search_tn_als.py # TN-ALS grid search runner
│   ├── dataset_loader.py  # Dataset loading utilities
│   └── config_parser.py   # Config file parsing
├── tensor/               # Core tensor network code
│   ├── btn.py           # Bayesian Tensor Network
│   ├── tn_als.py        # TN Alternating Least Squares
│   └── builder.py       # Inputs class and BTN builder
├── model/               # Model architectures
│   ├── MPO2_models.py   # MPO2, LMPO2, CMPO2, MMPO2
│   ├── BTT_models.py    # BTT model
│   └── CPD_models.py    # CPD model
├── results/             # Experiment results output
└── runs/                # Tracker logs
```

## Configuration
Experiments are configured using JSON files. Below is an example configuration (`btn_concrete_mpo2.json`):

```json
{
  "experiment_name": "btn_concrete",
  "dataset": "concrete",
  "task": "regression",
  "parameter_grid": {
    "model": ["MPO2"],
    "L": [3, 4],
    "bond_dim": [18],
    "trimming_threshold": [0.9, 0.95, 0.99],
    "trim_method": ["relevance"],
    "added_bias": [false, true]
  },
  "fixed_params": {
    "batch_size": 256,
    "n_epochs": 100,
    "patience": 100,
    "min_delta": 0.0001,
    "soft_trim_relaxation": 1,
    "trim_every": 6,
    "init_strength": 0.1,
    "seeds": [42, 7, 123, 256, 999],
    "warmup_epochs": 5,
    "bond_prior_alpha": 5.0
  },
  "tracker": {
    "backend": "file",
    "tracker_dir": "runs",
    "aim_repo": ""
  },
  "output": {
    "results_dir": "results/btn_concrete_MPO2",
    "save_models": false,
    "save_individual_runs": true
  }
}
```

### Configuration Fields
- `experiment_name`: A unique identifier for the experiment.
- `dataset`: The name of the dataset to use.
- `parameter_grid`: A dictionary of parameters to sweep over during grid search.
- `fixed_params`: Parameters that remain constant across all runs in the grid search.
- `tracker`: Configuration for experiment tracking (e.g., logging to files or Aim).
- `output`: Directory and settings for saving results and models.

### Available Datasets
The following datasets are supported: `student_perf`, `abalone`, `obesity`, `bike`, `realstate`, `energy_efficiency`, `concrete`, `ai4i`, `appliances`, `seoulBike`.

### Available Models
The following model architectures are available: `MPO2`, `LMPO2`, `BTT`, `CPD`.

## Defining New Models

Models are defined in the `model/` folder. Each model class must expose:

### Required Attributes

```python
class MyModel:
    bond_prior_alpha = 5.0  # Prior strength for bond precision
    
    def __init__(self, L, bond_dim, phys_dim, output_dim=1, ...):
        # Build your tensor network
        self.tn = qt.TensorNetwork(tensors)
        
        # Define which indices are inputs vs outputs (names are up to you)
        self.input_dims = [...]    # List of input index labels
        self.output_dims = [...]   # List of output index labels (empty for scalar output)
```

### Key Points

1. **`self.tn`**: A `quimb.TensorNetwork` containing your model tensors
2. **`self.input_dims`**: List of index names that will receive input data
3. **`self.output_dims`**: List of index names for output (empty list `[]` for scalar regression)
4. **Node tags**: Each tensor should have a unique tag (e.g., `Node0`, `Node1`, ...) for the BTN update steps

Index names and tensor structure are flexible - define them as needed for your architecture.

### Example

```python
class MyModel:
    bond_prior_alpha = 5.0
    
    def __init__(self, L, bond_dim, phys_dim, output_dim=1, **kwargs):
        tensors = []
        for i in range(L):
            # Define your tensor shape and indices
            data = torch.randn(phys_dim, bond_dim) * 0.1
            tensor = qt.Tensor(data=data, inds=(f"in{i}", f"bond{i}"), tags={f"Node{i}"})
            tensors.append(tensor)
        
        self.tn = qt.TensorNetwork(tensors)
        self.input_dims = [f"in{i}" for i in range(L)]
        self.output_dims = []  # scalar output
```

### Registering

Add to `model/__init__.py`:

```python
from model.my_model import MyModel
MODELS = { ..., "MyModel": MyModel }
```

## Running Experiments
You can run experiments using the provided grid search runners.

### BTN Experiments
```bash
python -m experiments.run_grid_search_btn --config experiments/configs/btn_concrete_mpo2.json --verbose
```

### TN-ALS Experiments
```bash
python -m experiments.run_grid_search_tn_als --config experiments/configs/als_concrete_mpo2.json --verbose
```

### Polynomial Experiments
```bash
python -m experiments.run_grid_search_polynomial --config experiments/configs/polynomial_1d_test.json --verbose
```

### CLI Flags

| Flag | Description | Default |
|------|-------------|---------|
| `--config` | Path to JSON config file (required) | - |
| `--verbose` | Print detailed progress per epoch | `False` |
| `--force` | Remove existing results and re-run | `False` |
| `--tracker` | Tracking backend: `file`, `aim`, `both`, `none` | `file` |
| `--tracker-dir` | Directory for file tracker logs | `experiment_logs` |
| `--aim-repo` | AIM repository path or `aim://host:port` | `None` |
| `--output-dir` | Output directory for results | from config |

### Flag Priority

Priority order: **CLI flags (non-default) > Config file > Defaults**

```bash
# Config says tracker: "file", but CLI overrides to "aim"
python -m experiments.run_grid_search_btn --config config.json --tracker aim

# Config says tracker_dir: "runs", but CLI overrides
python -m experiments.run_grid_search_btn --config config.json --tracker-dir my_logs
```

Note: If you pass a flag with its default value (e.g., `--tracker file`), the config file value takes precedence since the runner cannot distinguish between "explicitly passed default" and "not passed".

## Key Classes
- `Inputs` (`tensor/builder.py`): A utility class for batching and loading data into the tensor network models.
- `BTN` (`tensor/btn.py`): The core implementation of the Bayesian Tensor Network.
- `TNALS` (`tensor/tn_als.py`): Implementation of Tensor Networks optimized using Alternating Least Squares.

## Adding New Datasets
New datasets can be added by modifying `experiments/dataset_loader.py`. The project uses the UCI Machine Learning Repository via the `ucimlrepo` package to fetch and preprocess datasets.

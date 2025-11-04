# CLR: Conservative Learning-to-Round for Maximum Cut

**Status**: ✅ **Implementación Completa** (Listo para Training y Evaluación)

A hybrid approach combining SDP relaxation with learned rounding policies for the Maximum Cut problem.

## Key Features

- ✅ **Corrected SDP Solver** with proper vector factorization
- ✅ **Probabilistic GW (pGW)** baseline implementation  
- ✅ **Quality Certification System** - unique contribution for ML methods
- ✅ **Mixed Sampler** - π_λ = λ·Uniform + (1-λ)·π_learned ⭐ NEW
- ✅ **GNN Policy Architecture** - 3-layer GCN for learned rounding ⭐ NEW
- ✅ **RL Trainer** - REINFORCE with baseline for policy optimization ⭐ NEW
- ✅ **Training Script** - End-to-end training pipeline ⭐ NEW
- ✅ **Evaluation Script** - Benchmark-compatible evaluation ⭐ NEW

## Installation

```bash
# From MaxCut-Bench root
cd MaxCut-Bench

# Dependencies already installed in benchenv
conda activate benchenv

# Additional requirements (if needed)
pip install cvxpy
```

## Quick Start

### 1. Test SDP Solver

```bash
python solvers/CLR/tests/test_sdp_solver.py
```

Expected output:
- ✅ Test 1: Triangle graph (Z_SDP ≈ 2.25)
- ✅ Test 2: K4 graph (Z_SDP = 4.0)
- ✅ Test 3: Real graph from BA_20

### 2. Test pGW Baseline

```bash
python solvers/CLR/tests/test_pgw.py
```

Expected output:
- pGW(K=50) improves over GW(K=1) by ~10%
- Gap reduces from ~20% to ~11%

### 3. Compare Old vs New SDP

```bash
python solvers/CLR/tests/compare_old_vs_new_sdp.py
```

This demonstrates the bug in the original SDP implementation.

## Current Status

### Implemented ✅

- **SDP Solver** (`core/sdp_solver.py`)
  - Correct factorization of SDP matrix
  - Caching system for efficiency
  - Robust error handling
  
- **Baseline Solvers** (`utils/baseline_solvers.py`)
  - Goemans-Williamson (GW)
  - Probabilistic GW (pGW)
  
- **Quality Certification** (`core/quality_certificate.py`)
  - Gap computation (absolute & relative)
  - Approximation ratio
  - Certification levels

- **Comprehensive Tests**
  - Unit tests for all components
  - Validation on small and real graphs

### Coming Soon 🔲

- **Mixed Sampler** - Convex combination of uniform and learned policies
- **GNN Policy** - Graph Neural Network for learned rounding
- **RL Trainer** - Training pipeline for the GNN
- **Full Evaluation** - Integration with benchmark framework

## Architecture

```
solvers/CLR/
├── core/
│   ├── sdp_solver.py           # SDP relaxation (corrected)
│   ├── quality_certificate.py  # Gap-based certification
│   ├── mixed_sampler.py        # π_λ = λ·Uniform + (1-λ)·Learned
│   └── hyperplane_rounding.py  # Rounding utilities
├── models/
│   ├── gnn_policy.py           # GNN architecture
│   └── rl_trainer.py           # REINFORCE training
├── utils/
│   └── baseline_solvers.py     # GW & pGW
└── tests/
    ├── test_sdp_solver.py
    ├── test_pgw.py
    └── compare_old_vs_new_sdp.py
```

## Key Contributions

### 1. Quality Certification (Unique to CLR)

CLR is the first ML-based MaxCut method that provides **post-hoc quality certificates**:

```python
certificate = {
    'cut': 11.0,
    'Z_SDP': 11.66,
    'gap_relative': 0.0567,  # 5.67%
    'certification_level': 'GOOD'
}
```

This means: **"We guarantee this solution is within 5.67% of optimal"**

### 2. Corrected SDP Implementation

The original `solvers/SDP/evaluate.py` has several bugs:
- Uses `matrix.value` directly without factorization
- Only 1 sample (no pGW)
- No gap computation

Our implementation fixes all of these.

### 3. Conservative Mixing (Coming Soon)

```python
π_λ = λ · Uniform(S^(n-1)) + (1-λ) · π_learned
```

- λ=1.0: Pure GW (guaranteed)
- λ=0.0: Pure learned (high performance)
- λ=0.5: Balanced (our default)

## Preliminary Results

### pGW on BA_20 (10 graphs)

| Metric | GW (K=1) | pGW (K=50) | Improvement |
|--------|----------|------------|-------------|
| Mean cut | 12.60 | 13.90 | +10.3% |
| Mean gap | 19.97% | 11.41% | -43% |

### SDP Performance

- Small graphs (n=20): ~0.01s
- Medium graphs (n=200): ~0.1-1s (estimated)
- Caching: 2nd solve ~0s

## Usage Example

```python
from solvers.CLR.core.sdp_solver import SDPSolver
from solvers.CLR.utils.baseline_solvers import probabilistic_gw
from solvers.CLR.core.quality_certificate import compute_quality_certificate

# Load graph
import numpy as np
adjacency = ...  # Your graph

# Solve SDP
solver = SDPSolver()
sdp_result = solver.solve(adjacency)

# Run pGW
pgw_result = probabilistic_gw(adjacency, K=50)

# Get certificate
cert = compute_quality_certificate(
    cut_value=pgw_result['cut'],
    Z_SDP=pgw_result['Z_SDP']
)

print(f"Cut: {cert['cut']:.2f}")
print(f"Gap: {cert['gap_relative']*100:.2f}%")
print(f"Level: {cert['certification_level']}")
```

## Comparison with State-of-the-Art

| Method | Performance | Guarantees | Certification | Speed |
|--------|-------------|------------|---------------|-------|
| **CLR** | High (target) | Partial | ✅ Yes | Slow (SDP) |
| GW | Medium | 0.878-approx | ❌ No | Medium |
| pGW | Medium-High | None | ❌ No | Medium |
| Tabu Search | Very High | ❌ None | ❌ No | ✅ Fast |
| S2V-DQN | High | ❌ None | ❌ No | Fast |

**CLR's advantage**: Only method with quality certification per instance.

---

## 🚀 Training CLR Policy

### Prerequisites

Install PyTorch and PyTorch Geometric:

```bash
# Install PyTorch (check pytorch.org for your CUDA version)
pip install torch torchvision torchaudio

# Install PyTorch Geometric
pip install torch-geometric

# Install additional dependencies
pip install networkx
```

### Training on Synthetic Graphs

```bash
# Train on BA_20 (fast, for testing)
python solvers/CLR/train.py \
    --distribution BA_20 \
    --num_graphs 1000 \
    --epochs 50 \
    --gpu

# Train on larger graphs (for real experiments)
python solvers/CLR/train.py \
    --distribution BA_200 \
    --num_graphs 5000 \
    --epochs 100 \
    --n 200 \
    --m 3 \
    --gpu \
    --use_value_baseline

# Train using existing data
python solvers/CLR/train.py \
    --distribution BA_20 \
    --load_from_disk \
    --data_path data/training/BA_20 \
    --epochs 50
```

**Training parameters:**
- `--distribution`: Graph type (BA, ER, WS)
- `--num_graphs`: Number of training graphs
- `--epochs`: Training epochs
- `--lr`: Learning rate (default: 1e-4)
- `--baseline_K`: Uniform samples for baseline (default: 10)
- `--policy_K`: Policy samples per graph (default: 10)
- `--gpu`: Use GPU if available

**Outputs:**
- `solvers/CLR/pretrained/{distribution}/policy_best.pth`: Best model
- `solvers/CLR/pretrained/{distribution}/policy_final.pth`: Final model
- `solvers/CLR/pretrained/{distribution}/training_history.pkl`: Training curves

### Monitoring Training

Check training progress:
```python
import pickle
with open('solvers/CLR/pretrained/BA_20/training_history.pkl', 'rb') as f:
    history = pickle.load(f)

# Plot improvement over epochs
import matplotlib.pyplot as plt
train_improvements = [epoch['improvement'] for epoch in history['train']]
plt.plot(train_improvements)
plt.xlabel('Epoch')
plt.ylabel('Improvement over Uniform (%)')
plt.title('CLR Training Progress')
plt.show()
```

---

## 📊 Evaluation

### Evaluate on Test Distribution

```bash
# Evaluate with learned policy (λ=0.5)
python solvers/CLR/evaluate.py \
    --test_distribution BA_800vertices_weighted \
    --train_distribution BA_20 \
    --lambda_mix 0.5 \
    --K 50

# Evaluate with pure uniform (λ=1.0, equivalent to pGW)
python solvers/CLR/evaluate.py \
    --test_distribution BA_800vertices_weighted \
    --lambda_mix 1.0 \
    --K 50

# Evaluate with pure learned (λ=0.0, risky)
python solvers/CLR/evaluate.py \
    --test_distribution BA_800vertices_weighted \
    --train_distribution BA_20 \
    --lambda_mix 0.0 \
    --K 50
```

**Evaluation parameters:**
- `--test_distribution`: Test dataset (e.g., BA_800vertices_weighted)
- `--train_distribution`: Training distribution (defaults to test)
- `--lambda_mix`: Mixing parameter λ ∈ [0,1]
  - `λ=1.0`: Pure uniform (pGW baseline)
  - `λ=0.5`: Conservative mixing
  - `λ=0.0`: Pure learned (no guarantees)
- `--K`: Number of samples (best-of-K)

**Outputs:**
- Results saved in `results/{test_distribution}/CLR` (pickle format)
- Compatible with benchmark evaluation scripts

### Ablation Studies

**Lambda ablation:**
```bash
for lambda in 0.0 0.25 0.5 0.75 1.0; do
    python solvers/CLR/evaluate.py \
        --test_distribution BA_800vertices_weighted \
        --lambda_mix $lambda \
        --K 50
done
```

**K ablation:**
```bash
for K in 1 10 50 100; do
    python solvers/CLR/evaluate.py \
        --test_distribution BA_800vertices_weighted \
        --lambda_mix 0.5 \
        --K $K
done
```

### Compare Results

```python
import pandas as pd

# Load CLR results
clr_results = pd.read_pickle('results/BA_800vertices_weighted/CLR')

# Load baseline results
tabu_results = pd.read_pickle('results/BA_800vertices_weighted/TS')
gw_results = pd.read_pickle('results/BA_800vertices_weighted/Standard Greedy')

# Compare cuts
print(f"CLR mean cut: {clr_results['cut'].mean():.2f}")
print(f"Tabu mean cut: {tabu_results['cut'].mean():.2f}")
print(f"GW mean cut: {gw_results['cut'].mean():.2f}")

# CLR unique: gap certification
print(f"\nCLR mean gap: {clr_results['gap_relative'].mean()*100:.2f}%")
print("Certification levels:")
print(clr_results['certification_level'].value_counts())
```

---

## Timeline

- **Week 0** (✅ Done): SDP solver, pGW, certification, mixed sampler, GNN, trainer
- **Week 1**: Training on BA_20, ER_200, initial experiments
- **Week 2**: Full evaluation on benchmark, ablation studies
- **Week 3**: Refinement, comparison with baselines
- **Week 4+**: Paper writing and submission

## References

1. Goemans & Williamson (1995): GW algorithm
2. Maliakal et al. (2025): Dataless RL for rounding
3. Nath & Kuhnle (2024): MaxCut benchmark
4. Qiu et al. (2025): ROS framework

## Team

Daniel Miranda, Nikolai Navea, Vicente Arratia, Javier Sepúlveda  
Course: OII464 - Desarrollo de Diseños Híbridos para Optimización  
Instructor: Emanuel Vega

## License

This project extends the MaxCut-Bench framework. Please cite both:
- Original benchmark: Nath & Kuhnle (2024)
- CLR method: Miranda et al. (2025, in preparation)

---

**Status**: Setup complete ✅ | Implementation in progress 🔲

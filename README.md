# Axon-SNN-Training

### Spiking Nueral Network Through LLM and SLM ###
Spiking Neural Networks are a class of artificial neural networks that mimic the behavior of biological neurons more closely than traditional neural networks. In SNNs, neurons communicate by sending discrete spikes, which represent changes in voltage across a neuron's membrane. These spikes are generated when the membrane potential exceeds a certain threshold.

The human brain consists of approximately 86 billion neurons, which communicate through electrical impulses known as action potentials or spikes. This communication method is energy-efficient and highly effective for processing information. SNNs aim to replicate this spiking behavior, leveraging the brain's mechanisms for computation and learning.

# snn_core — Biological Spiking Neural Network Library

A Rust library with Python bindings for biologically realistic spiking
neural networks. Built on the 8-layer biological neuron with full
Hodgkin-Huxley dynamics, ternary signaling, and STDP learning.

---

## Installation

### Prerequisites

```bash
# 1. Install Rust (one-time)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source "$HOME/.cargo/env"

# 2. Install maturin (Python build tool for Rust extensions)
pip install maturin

# 3. (Optional but recommended) Create a virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install maturin numpy
```

### Build & Install

```bash
# Clone or download this folder, then:
cd snn_lib

# Development install (rebuilds on every cargo build --release)
maturin develop --release

# OR: Build a .whl wheel you can pip install anywhere
maturin build --release
pip install target/wheels/snn_core-*.whl
```

That's it. Now `import snn_core` works in any Python file.

---

## Quick Start

```python
import snn_core

# Single neuron (LIF — fast)
neuron = snn_core.Neuron(id=0, use_hh=False, inhibitory=False)
for step in range(300):
    t   = step * 0.1
    out = neuron.step(0.0, 12.0, 0.0, 0.0, t, 0.1)
    if out != snn_core.TernarySpike.Silent:
        print(f"  t={t:.1f}ms  {out.symbol()}")

print(f"Total spikes: {neuron.spike_count}")
```

---

## File Structure

```
snn_lib/
├── Cargo.toml          ← Rust package config + pyo3 dependency
├── pyproject.toml      ← maturin build config
├── example.py          ← Full usage examples (run after maturin develop)
├── README.md
└── src/
    ├── lib.rs          ← PyO3 module entry point (registers all classes)
    ├── neuron.rs       ← Neuron, HHGating, SynapticState, TernarySpike
    ├── cluster.rs      ← NeuronCluster, ClusterConfig, Synapse
    ├── plasticity.rs   ← STDPRule, BCMRule, HomeostaticScaling, ReadoutLayer
    ├── encoding.rs     ← Vocabulary, PopulationEncoder, TernaryStateCollector
    └── network.rs      ← LanguageSNN, TrainingConfig
```

---

## Python API

### Neuron

```python
# LIF neuron (fast, good for large networks)
n = snn_core.Neuron(id=0, use_hh=False, inhibitory=False)

# HH neuron (full Hodgkin-Huxley, biologically accurate)
n = snn_core.Neuron(id=0, use_hh=True, inhibitory=False)

# Step the neuron (dt in milliseconds)
# Returns TernarySpike.Excitatory / .Silent / .Inhibitory
out = n.step(i_syn, i_ext, glu, gaba, t, dt)

n.v_m          # membrane potential (mV)
n.spike_count  # total spikes fired
n.ca           # intracellular calcium
n.pre_trace    # STDP pre-synaptic eligibility trace
n.post_trace   # STDP post-synaptic eligibility trace
n.ternary()    # returns -1, 0, or +1
n.reset()      # reset to resting state
```

### NeuronCluster

```python
cfg = snn_core.ClusterConfig()
cfg.n_excitatory = 800    # glutamatergic neurons
cfg.n_inhibitory = 200    # GABAergic interneurons
cfg.p_connect    = 0.10   # 10% connection probability
cfg.w_ee = 0.5            # E→E weight
cfg.w_ei = 0.8            # E→I weight
cfg.w_ie = 1.2            # I→E weight (strong inhibition)
cfg.w_ii = 0.4            # I→I weight
cfg.noise_sigma  = 0.3    # background noise (pA)
cfg.use_hh       = False  # True = Hodgkin-Huxley, False = LIF

cluster = snn_core.NeuronCluster(cfg)
cluster.inject([0, 1, 2], 10.0)    # inject 10pA into neurons 0,1,2
cluster.inject_vector(currents)     # inject full current vector
spikes = cluster.step(dt=0.1)       # returns [(neuron_id, ternary_val), ...]

cluster.ternary_state()    # [-1, 0, 1, 0, 1, ...] for all neurons
cluster.total_neurons()    # int
cluster.total_synapses()   # int
cluster.total_spikes()     # u64
cluster.mean_weight()      # float
cluster.pop_rate_exc       # excitatory population rate (EMA)
cluster.pop_rate_inh       # inhibitory population rate (EMA)
```

### STDP

```python
stdp = snn_core.STDPRule()
stdp.a_plus   = 0.01   # LTP amplitude
stdp.a_minus  = 0.012  # LTD amplitude
stdp.tau_plus = 20.0   # LTP time window (ms)
stdp.tau_minus= 20.0   # LTD time window (ms)

# Update weight by spike timing (dt = t_post - t_pre, ms)
w = stdp.update_by_timing(weight=0.5, dt=5.0)   # pre before post → LTP
w = stdp.update_by_timing(weight=0.5, dt=-5.0)  # post before pre → LTD

# Update by trace
w = stdp.ltp(weight, pre_trace)
w = stdp.ltd(weight, post_trace)
```

### Language SNN

```python
cfg = snn_core.TrainingConfig()
cfg.n_input_neurons = 200
cfg.n_reservoir     = 1000   # total reservoir neurons
cfg.dt              = 0.5    # ms
cfg.window_ms       = 15.0   # ms per character
cfg.learning_rate   = 0.005
cfg.n_epochs        = 10
cfg.use_stdp        = True
cfg.temperature     = 0.8    # sampling temperature for generation

snn = snn_core.LanguageSNN(training_text, cfg)
snn.train(training_text, verbose=True)

# Generate text autoregressively
text = snn.generate(seed="t", length=100)

snn.loss_history   # [float] per epoch
snn.acc_history    # [float] per epoch
snn.reservoir_stats()  # string summary
```

---

## Scaling Guide

| Neurons | Speed         | Notes                              |
|---------|---------------|------------------------------------|
| 200     | ~1s/epoch     | Python fallback, works on anything |
| 1,000   | ~1s/epoch     | Rust LIF, good baseline            |
| 5,000   | ~3s/epoch     | Rust LIF + Rayon parallel          |
| 20,000  | ~10s/epoch    | Enable Rayon fully in cluster.rs   |
| 100,000 | ~1min/epoch   | Needs CUDA port (future work)      |

---

## Troubleshooting

**`maturin develop` fails with "linker not found"**
```bash
# Linux
sudo apt install build-essential

# macOS
xcode-select --install

# Windows — install Visual Studio Build Tools
# https://visualstudio.microsoft.com/visual-cpp-build-tools/
```

**`import snn_core` gives ModuleNotFoundError**
```bash
# Make sure you're in the same venv where you ran maturin develop
source .venv/bin/activate
python -c "import snn_core; print(snn_core.__version__)"
```

**Rust compile error: "pyo3 version mismatch"**
```bash
# Update pyo3 in Cargo.toml to match your Python version
# Check: python --version, then: cargo update
cargo update
maturin develop --release
```

---

## GPU Training (GTX 1050 Ti)

### 1. Install GPU dependencies

```bash
# PyTorch with CUDA 11.8 (compatible with GTX 1050 Ti)
pip install torch torchvision \
  --index-url https://download.pytorch.org/whl/cu118

# SpikingJelly — GPU SNN framework
pip install spikingjelly

# Verify everything works
python setup_gpu.py
```

### 2. Get training data

```bash
python get_data.py --task chat     # DailyDialog + TinyStories
```

### 3. Train the GPU chatbot

```bash
# Auto-sizes to your VRAM (256 neurons/layer for 4GB)
python train_gpu_chat.py --corpus data/chat_corpus.txt

# More epochs, bigger batch
python train_gpu_chat.py --corpus data/chat_corpus.txt --epochs 30 --batch 64

# Resume from checkpoint
python train_gpu_chat.py --resume checkpoint.pt

# Just chat (no training)
python train_gpu_chat.py --chat --resume checkpoint.pt
```

### 4. Train on real spike data (SHD)

```bash
pip install tonic
python get_data.py --task spikes
python train_shd.py
```

### GTX 1050 Ti specifics

| Setting | Value | Why |
|---------|-------|-----|
| Neurons/layer | 256 | Fits in 4GB VRAM |
| Total neurons | 768 | 3-layer hierarchy |
| FP16 training | ON | Cuts VRAM use in half |
| Batch size | 32 | Safe for 4GB |
| T_fast | 8 steps | Speed/accuracy balance |

The script auto-enables FP16 mixed precision, which halves VRAM use
and speeds up training ~1.5x on Pascal-architecture GPUs (1050 Ti).

### Architecture (GPU version)

```
Input char
    ↓
CharacterEncoder  (sparse spike patterns, 256 neurons, T=8 steps)
    ↓
ReservoirLayer 1  (256 neurons, τ=10ms,  Dale's law, fast dynamics)
    ↓  proj_12
ReservoirLayer 2  (256 neurons, τ=25ms,  medium timescale)
    ↓  proj_23
ReservoirLayer 3  (256 neurons, τ=80ms,  slow / working memory)
    ↓
Linear Readout    (768 → vocab_size)  with dropout
    ↓
Next character prediction
```

Training uses **surrogate gradient backpropagation** (ATan surrogate)
through all three spiking layers end-to-end — the GPU version of
e-prop that works with PyTorch autograd.

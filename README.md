# LASERNet

Deep learning models for predicting microstructure evolution in laser welding simulations using CNN-LSTM and PredRNN architectures.

## Project Overview

LASERNet trains and compares multiple deep learning models to predict microstructure evolution from temperature and past microstructure sequences. The models learn spatiotemporal patterns from 3D point cloud data extracted from laser welding simulations.

## Quick Start

### Setup

```bash
# Install dependencies and setup environment
make init
```

This will:
- Install `uv` package manager if not available
- Sync all dependencies from `pyproject.toml`
- Install Jupyter kernel for the virtual environment

### Environment Configuration

The project uses the `BLACKHOLE` environment variable to locate data files. This is configured using a `.env` file:

1. **Copy the example environment file:**
   ```bash
   cp .env.example .env
   ```

2. **Edit `.env` and set your data path:**
   ```bash
   BLACKHOLE=/path/to/your/blackhole/directory
   ```

3. **Ensure your data directory structure:**
   Your `BLACKHOLE` path should contain either:
   - `Data/` with CSV files (for initial loading), or
   - `processed/data/` with preprocessed `.pt` files (for fast loading)

**Example paths:**
- DTU HPC: `BLACKHOLE=/dtu/blackhole/06/168550`
- Local: `BLACKHOLE=/Users/username/dtu/LASERNet/BLACKHOLE`

### Running the Models

**The main entry point is the [lasernet.ipynb](MICROnet.ipynb) notebook**, which is a concatination of the MICROnet.ipynb and the temperature-prediction.ipynb.

trains and evaluates multiple model configurations:

```bash
# Execute the notebook locally
make lasernet_notebook

# Or submit to HPC job queue
make submit_lasernet_notebook
```

### Viewing Results

After running the notebook, results from the micronet models are available in two places:

1. **In the notebook itself** - View training plots, predictions, and comparisons directly in [lasernet.ipynb](notebootks/lasernet.ipynb)
2. **In the output directory** - All artifacts saved to `notebooks/MICROnet_output/`:
   ```
   MICROnet_output/
   ├── <model_name>/
   │   ├── config.json                    # Model configuration
   │   ├── history.json                   # Training history
   │   ├── training_losses.png            # Loss curves
   │   ├── pred_t23_s47.png              # Prediction visualization
   │   ├── solidification_mask_t23_s47.png  # Solidification mask (combined loss models)
   │   └── checkpoints/
   │       ├── best_model.pt             # Best validation checkpoint
   │       └── final_model.pt            # Final model
   └── all_models_comparison.png          # Comparison across all models
   ```

### Other Commands

```bash
# Show all available commands
make help

# Clean up logs and cache
make clean
```

## Repository Structure

```
LASERNet/
├── notebooks/
│   ├── lasernet.ipynb           # Main notebook - train & evaluate models
│   └── MICROnet_output/         # Output directory for all results
├── lasernet/
│   ├── model/                   # Neural network architectures
│   │   ├── MicrostructureCNN_LSTM.py   # CNN-LSTM model
│   │   ├── MicrostructurePredRNN.py    # PredRNN model
│   │   └── losses.py                   # Loss functions (MSE, Combined)
│   ├── dataset/                 # Data loading and preprocessing
│   │   ├── loading.py          # Dataset classes
│   │   └── fast_loading.py     # Fast loading from preprocessed data
│   └── utils/                   # Visualization utilities
│       ├── plot.py             # Loss curves
│       └── visualize.py        # Prediction visualizations
├── microstructure_training.py   # Training and prediction utilities
├── batch/scripts/               # HPC job submission scripts
├── makefile                     # Build commands
└── pyproject.toml              # Dependencies
```

## Data

Data is stored in `$BLACKHOLE/Data/` and automatically loaded by the notebook:

- **Raw data**: `Alldata_withpoints_*.csv` files (one per timestep)
  - Contains: 3D coordinates, temperature, microstructure (IPF channels)

- **Preprocessed data**: `processed/data/` (for fast loading)
  - Pre-extracted 2D slices from 3D point clouds
  - Significantly faster loading times

The notebook automatically detects and uses preprocessed data if available, otherwise falls back to loading from CSV files.

# Leveraging Strava Metro Data to Enhance Urban Cycling Infrastructure Development in Brussels

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)

This repository contains the code for the paper **"Leveraging Strava Metro Data to enhance urban cycling infrastructure development in Brussels"**. 

> **Note:** The raw Strava Metro data is not publicly available due to licensing restrictions. All other data used in this research can be found at: [Link to Data Repository]

## 📋 Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Data Processing Pipeline](#data-processing-pipeline)
- [Model Training](#model-training)
- [Usage](#usage)
- [Results](#results)
- [Citation](#citation)

## 🎯 Overview

This project investigates how **Strava Metro data** can be leveraged to understand and improve **urban cycling infrastructure** in the Brussels Capital Region. The key contributions include:

1. **Spatial Analysis**: Matching Strava edge data with cycling infrastructure using fuzzy spatial matching algorithms
2. **Bike Counter Correlation**: Associating Strava trips with official bike counter measurements using Voronoi tessellation
3. **Multi-Head Neural Network**: A deep learning model to predict bike counter readings from Strava data and contextual features (weather, temporal)
4. **Extrapolation Framework**: Using the trained model to extrapolate cycling volumes across the entire Brussels network

## 📁 Project Structure

```
├── data/                          # Data directory (not tracked in git)
│   ├── bike_counters/             # Bike counter data and metadata
│   ├── bike_infrastructure/       # Cycling infrastructure shapefiles
│   ├── brussels/                  # Brussels region shapefiles
│   ├── network/                   # Preprocessed datasets for training
│   ├── strava_edge_data/          # Strava Metro edge data
│   └── weather/                   # Weather data from AWS
│
├── scripts/                       # Main processing scripts
│   ├── preprocessing/             # Data preprocessing scripts
│   │   ├── merged_strava.py       # Merge yearly Strava files
│   │   ├── add_location_to_strava_total.py
│   │   ├── bike_infrastructure.py
│   │   ├── brussels_tessalation.py
│   │   ├── counter_information.py
│   │   ├── strava_shape.py
│   │   └── weather_dataset_creation.py
│   │
│   ├── network_training/          # Model training scripts
│   │   └── sweep.py               # Hyperparameter sweep with W&B
│   │
│   ├── notebooks/                 # Jupyter notebooks for analysis
│   │   ├── extrapolating_strava.ipynb
│   │   ├── figure_3.ipynb
│   │   └── total_dataset_analysis.ipynb
│   │
│   ├── matching_strava_infrastructure.py
│   ├── stravaedge_counter_tessalation_association.py
│   ├── total_edge_counter_merged_dataset.py
│   ├── total_strava_enhanced_dataset.py
│   └── training_testing_dataset.py
│
├── src/                           # Source code (installable package)
│   ├── __init__.py
│   ├── aux_utilities.py           # Auxiliary utility functions
│   ├── geospatial_utilities.py    # Geospatial processing utilities
│   ├── loading_datasets.py        # Dataset loading functions
│   ├── network.py                 # Multi-head neural network architecture
│   ├── network_datasets.py        # PyTorch dataset classes
│   └── plotting_utilities.py      # Visualization utilities
│
├── results/                       # Output results and figures
│   ├── datasets/                  # Processed datasets
│   ├── figures/                   # Generated figures
|   ├── network_results/
│
├── docs/                          # Documentation
├── setup.py                       # Package setup file
└── README.md
```

## 🔧 Installation

### Prerequisites

- Python 3.9 or higher
- CUDA-capable GPU (recommended for training)

### Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/strava-for-cycling-infrastructure.git
   cd strava-for-cycling-infrastructure
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   pip install -e .
   ```

4. **Set up environment variables:**
   
   Create a `.env` file in the root directory:
   ```env
   DATA_PATH=/path/to/data
   RESULTS_PATH=/path/to/results
   FULL_PAGE_WIDTH_CM=17
   FULL_PAGE_LENGTH_CM=24
   ```

## 🔄 Data Processing Pipeline

### Step 1: Preprocess Strava Data

```bash
# Merge all yearly Strava files (2019-2024)
python scripts/preprocessing/merged_strava.py

# Add location IDs based on Brussels tessellation
python scripts/preprocessing/add_location_to_strava_total.py

```

### Step 2: Process Infrastructure Data

```bash
# Process bike infrastructure data
python scripts/preprocessing/bike_infrastructure.py

# Create Brussels tessellation (Voronoi regions)
python scripts/preprocessing/brussels_tessalation.py
```

### Step 3: Match Strava Edges with Infrastructure

```bash
# Fuzzy spatial matching between Strava edges and infrastructure
python scripts/matching_strava_infrastructure.py
```

### Step 4: Associate Strava with Bike Counters

```bash
# Create tessellation-based association
python scripts/stravaedge_counter_tessalation_association.py

# Create merged dataset
python scripts/total_edge_counter_merged_dataset.py
```

### Step 5: Create Enhanced Dataset

```bash
# Add weather and temporal features, normalize data
python scripts/total_strava_enhanced_dataset.py

# Create train/val/test splits
python scripts/training_testing_dataset.py
```

## 🧠 Model Training

The model uses a **Multi-Head Neural Network** architecture with:
- Shared layers for common feature extraction
- Separate heads for each bike counter location
- Input features: Strava trips, weather data, temporal features

### Training with Hyperparameter Sweep

```bash
python scripts/network_training/sweep.py
```

The sweep uses [Weights & Biases](https://wandb.ai/) for experiment tracking and supports the following hyperparameters:
- `num_shared_layers`: Number of shared layers
- `shared_width`: Width of shared layers
- `num_head_layers`: Number of layers per head
- `head_width`: Width of head layers
- `learning_rate`: Learning rate for AdamW optimizer

## 📊 Usage

### Loading Datasets

```python
from src.loading_datasets import (
    loading_bike_counters,
    loading_brussels_shape,
    loading_brussels_cycling_network,
    loading_separated_bike_infrastructure
)

# Load bike counter data
bike_counters = loading_bike_counters()

# Load Brussels region shapes
brussels_region, brussels_municipalities = loading_brussels_shape()

# Load cycling infrastructure
cycling_network = loading_brussels_cycling_network()
```

### Jupyter Notebooks

Explore the analysis notebooks in `scripts/notebooks/`:
- `total_dataset_analysis.ipynb` - Exploratory data analysis
- `extrapolating_strava.ipynb` - Extrapolate cycling volumes

## 📈 Results

Results are saved to the `results/` directory:
- **Datasets**: Processed datasets in `results/datasets/`
- **Figures**: Generated visualizations in `results/figures/`
- **Model checkpoints**: Best models parameters in `results/model_checkpoints/`

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@article{author2025strava,
  title={Leveraging Strava Metro Data to enhance urban cycling infrastructure development in Brussels},
  author={Author Name},
  journal={Journal Name},
  year={2025},
  publisher={Publisher}
}
```

## 🤝 Acknowledgments

- Strava Metro for providing the cycling data
- Brussels Mobility for bike counter and infrastructure data
- Vrije Universiteit Brussel for supporting this research
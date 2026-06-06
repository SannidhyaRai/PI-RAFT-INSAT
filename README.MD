# PI-RAFT: Physics-Informed Recurrent All-Pairs Field Transform for AMV Retrieval

This repository contains the modularized implementation of **PI-RAFT**, a physics-informed deep learning model for retrieving **Atmospheric Motion Vectors (AMVs) [wind speed and direction vectors estimated by tracking cloud features in sequential satellite images]** using geostationary satellite data.

## Project Purpose
The primary goal of this research project is to estimate high-resolution wind velocity fields directly from sequential satellite images. Rather than treating this purely as an **optical flow [the pattern of apparent motion of objects or surfaces in a visual scene]** problem, the model incorporates meteorological physical laws (such as fluid dynamics and pressure heights) into the neural network optimization objective to ensure that the retrieved winds are physically realistic.

---

## Model Architecture Overview
The model is structured around a recurrent refinement network inspired by RAFT:
1.  **Feature and Context Encoders**: Custom convolutional architectures that downsample raw multi-spectral grids to extract spatial representations.
2.  **All-Pairs Correlation Volume**: A 4D tensor representing the dot product similarity between features across successive frames.
3.  **ConvGRU [Convolutional Gated Recurrent Unit]**: An iterative solver that updates the displacement vector fields over multiple steps.
4.  **Multi-Head Prediction**: Outputs horizontal wind vectors `[u, v]` and vertical **Cloud-Top Pressure (CTP) [the vertical atmospheric pressure level at the top of a cloud, measured in hectopascals (hPa)]**.
5.  **Physics-Informed Loss**: Couples standard data-matching loss with **Brightness Constancy [the physical assumption that a moving cloud feature retains its brightness/temperature value over short intervals]**, spatial gradient constancy, and fluid smoothness regularization.

---

## Directory Layout Guide

```text
PI-RAFT-INSAT/
│
├── configs/                # YAML configuration files for train, eval, and experiments
├── datasets/               # Dataset loaders, preprocessing, calibration, and geolocation
├── models/                 # Model architectures (encoders, correlation, ConvGRU, pressure head)
├── losses/                 # Physics-informed loss functions and warping utilities
├── evaluation/             # Metrics, consistency checks, and visualization scripts
├── export/                 # NetCDF4 exporting code
├── scripts/                # Execution entrypoints for training and inference
├── tests/                  # Unit and integration tests
├── notebooks/              # Orchestration notebooks for experiments and Colab execution
├── checkpoints/            # Serialized model and optimizer weights
└── outputs/                # Local cache and exported NetCDF files
```

---

## Project Status

### Implemented: ✅
*   Modular model components (`ResidualBlock`, `FeatureEncoder`, `ConvGRUCell`).
*   Differentiable spatial warping and multi-constraint physics loss framework.
*   CF-compliant NetCDF4 exporter.
*   GOES-16 proxy data streaming from public S3 buckets.

### In Progress: ⚠
*   Establishing a local validation pipeline with quantitative metrics (Mean Vector Difference).
*   Scaffolding INSAT-3DS HDF5 data loading interfaces.
*   Designing temporal frame sequence builders for 4-frame tracking.

### Planned: 📌
*   Optimization of the cloud-top pressure head via external reference data.
*   Integration of actual topography (Digital Elevation Model) inputs.
*   Resolution of the correlation lookup mapping (replacing the random placeholder).

---

## Onboarding and Getting Started

### Local Setup
Clone the repository and install it in editable mode:
```bash
git clone https://github.com/imd-amd-dof/PI-RAFT.git
cd PI-RAFT
pip install -r requirements.txt
pip install -e .
```

### Run Training
To start training the model using the default configuration parameters:
```bash
python scripts/train.py --config configs/train.yaml
```

### Run Inference
To execute inference, evaluate iteration scaling, and export wind vectors:
```bash
python scripts/inference.py --config configs/inference.yaml
```

### Google Colab Execution

The orchestration notebook at [`notebooks/baseline.ipynb`](notebooks/baseline.ipynb) runs the full pipeline (data loading, training, inference, and NetCDF export) headless in Colab. Follow these steps:

1. **Open in Colab**: Go to [colab.research.google.com](https://colab.research.google.com), select **File → Upload notebook**, and upload `notebooks/baseline.ipynb` from this repository.

2. **Set runtime to GPU**: Click **Runtime → Change runtime type** and set **Hardware accelerator** to **GPU** (required for training).

3. **Mount Google Drive (optional)**: If you want to persist checkpoints and output files, add a cell at the top with:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```

4. **Clone the repo**: Add a cell after the drive mount (or at the very top if not using Drive) to clone the repository:
   ```python
   !git clone https://github.com/imd-amd-dof/PI-RAFT.git
   %cd PI-RAFT
   !pip install -r requirements.txt
   !pip install -e .
   ```

5. **Run all cells**: Select **Runtime → Run all**. The notebook will:
   - Install dependencies (`boto3`, `netcdf4`, `xarray`)
   - Stream GOES-16 proxy data from public S3 buckets
   - Generate an untrained baseline wind field visualization
   - Train the model for 3 epochs (AdamW optimizer, physics-informed loss)
   - Run inference with iteration scaling (4 vs. 12 iterations)
   - Export retrieved wind vectors to a CF-compliant NetCDF4 file

   Checkpoints are saved to `./model_checkpoints/` and the final NetCDF to `./outputs/meteorological_outputs/INSAT_3DS_AMV_Derived_Wind.nc`.
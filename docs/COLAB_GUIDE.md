# Google Colab Execution Guide (Beginner-Proof)

This guide provides a complete, step-by-step walkthrough for setting up, training, and running inference on the **PI-RAFT** model using **Google Colab [a free, cloud-based notebook environment for running Python code]**.

---

## 1. What This Repository Is

This project implements **PI-RAFT (Physics-Informed Recurrent All-Pairs Field Transform) [a deep learning model that tracks cloud features across satellite images while respecting the physical laws of meteorology]**.

### What It Does
It takes successive satellite images and estimates **AMVs (Atmospheric Motion Vectors) [wind speed and direction vectors calculated by tracking the motion of clouds or water vapor features]**.

### Key Files in This Repository
*   **[baseline.ipynb](file:///c:/Users/Yash/OneDrive/Desktop/IMD_AMV_DOF/PI-RAFT-INSAT/notebooks/baseline.ipynb) (Experimentation Layer)**: This is your playground notebook. Use it to load data, run tests, and visualize output wind maps.
*   **[models/](file:///c:/Users/Yash/OneDrive/Desktop/IMD_AMV_DOF/PI-RAFT-INSAT/models/) (Core System Modules)**: Reusable python packages containing the model's neural network structure.
*   **[losses/](file:///c:/Users/Yash/OneDrive/Desktop/IMD_AMV_DOF/PI-RAFT-INSAT/losses/) (Loss Modules)**: Mathematical formulas that teach the model to respect brightness constancy and fluid smoothness.
*   **[configs/](file:///c:/Users/Yash/OneDrive/Desktop/IMD_AMV_DOF/PI-RAFT-INSAT/configs/) (Configuration Settings)**: YAML files containing parameters like learning rates and epoch counts.

### Notebook vs. Modules
*   **The Notebook (`baseline.ipynb`)** is only for **orchestration [coordinating high-level steps like trigger commands and showing plots]** and experimentation.
*   **The Modules (`models/`, `datasets/`, `losses/`)** contain the permanent system code. Do not paste massive class definitions into your notebook; write them in modules and import them.

---

## 2. First-Time Setup Step-by-Step

### Step 1: Open Google Colab
Go to [Google Colab](https://colab.research.google.com/) and upload the [baseline.ipynb](file:///c:/Users/Yash/OneDrive/Desktop/IMD_AMV_DOF/PI-RAFT-INSAT/notebooks/baseline.ipynb) file from this repository.

---

### Step 2: Connect GPU
Before executing any code, enable a **GPU (Graphics Processing Unit) [a specialized hardware accelerator that runs deep learning calculations significantly faster than a standard CPU]**:
1.  In the top menu, click **Runtime** $\rightarrow$ **Change runtime type**.
2.  Under *Hardware accelerator*, select **T4 GPU** (or any active GPU option).
3.  Click **Save**.

> [!NOTE]
> **Why do we need a GPU?** Training convolutional neural networks on standard CPUs can take hours or days; a GPU reduces this to minutes.
> *Free Tier Warning:* Google limits free GPU access. If you leave your notebook idling, Google will disconnect your GPU runtime. Remember to save checkpoints frequently!

---

### Step 3: Clone the Repository
Add a code cell to the top of your Colab notebook and run:
```python
# Clone the repository to the Colab virtual machine disk
!git clone https://github.com/imd-amd-dof/PI-RAFT.git

# Move (change directory) into the project root directory
%cd PI-RAFT
```
*   `!git clone` downloads a copy of the project files to Colab's temporary local drive.
*   `%cd` moves Colab's active path context inside that directory.

---

### Step 4: Install Dependencies
Run the following cell to install **dependencies (external libraries that our project code needs to run)**:
```python
# Install all required python packages listed in the requirements file
!pip install -r requirements.txt
```
This installs core packages like `boto3` (for streaming satellite files from Amazon Web Services), `netCDF4` (for opening standard weather data grids), and `xarray` (for multidimensional grid manipulation).

---

### Step 5: Mount Google Drive
Because Colab's local disk is wiped clean when your session expires, you must mount **Google Drive [your persistent cloud storage]** to store training outputs and checkpoints:
```python
from google.colab import drive
# Link your personal Google Drive to Colab's file system path '/content/drive'
drive.mount('/content/drive')
```
*   *Authentication:* Colab will pop up a window asking for permissions. Select your Google account and click "Allow".
*   *Why?* Without mounting Drive, all your trained model weights will be deleted when you close the browser tab.

---

### Step 6: Dataset Placement and Folder Structures
Create a directory structure inside your Google Drive. Go to your Google Drive homepage and create a folder named `PI_RAFT_DATA` with the following subdirectories:

```text
Google Drive (MyDrive/)
└── PI_RAFT_DATA/
    ├── raw/              # Raw data files (.nc for GOES-16, .h5 for INSAT-3DS)
    ├── processed/        # Extracted patches and crop files
    ├── outputs/          # Exported wind vector files (.nc files)
    └── checkpoints/      # Saved model parameters (.pth files)
```

Ensure your `configs/train.yaml` and `configs/inference.yaml` files have their directory paths pointed to these Google Drive locations, for example:
```yaml
logging:
  checkpoint_dir: "/content/drive/MyDrive/PI_RAFT_DATA/checkpoints"
export:
  output_dir: "/content/drive/MyDrive/PI_RAFT_DATA/outputs"
```

---

### Step 7: Running Training
To start training the model, run the training command:
```python
# Train the model using the configurations defined in the YAML file
!python scripts/train.py --config configs/train.yaml
```
#### Expected Outputs and Logs:
*   You will see lines printed to the console showing loss metrics:
    `Step: 002 | Batch: 1/4 | TRAIN TOTAL: 0.7850 ---> [Data: 0.6540 | Physics: 0.1310]`
*   At the end of each **epoch (one complete pass of the training data through the model)**, validation total loss will be printed.
*   Success looks like:
    `Successfully checkpointed model state to disk container: /content/drive/MyDrive/PI_RAFT_DATA/checkpoints/PI_RAFT_epoch_3.pth`

---

### Step 8: Running Inference
Once training completes, you can evaluate model **inference [running new frames through the trained model to predict winds without updating weights]**:
```python
# Run evaluation and save output NetCDF files
!python scripts/inference.py --config configs/inference.yaml
```
#### What happens during inference:
1.  The script loads the saved **checkpoint (saved model weights during training)** from your Drive folder.
2.  It runs a forward pass on validation frames.
3.  It overlays the predicted wind vector arrows on the moisture imagery and displays it.
4.  It exports the final zonal `u`, meridional `v`, and vertical pressure `P` variables to a NetCDF4 file inside `MyDrive/PI_RAFT_DATA/outputs/`.

---

### Step 9: Common Beginner Mistakes and Fixes

| Error / Symptom | Likely Cause | Fix |
| :--- | :--- | :--- |
| `FileNotFoundError` when loading files | Google Drive is not mounted or paths are misspelled | Run Step 5 again, and verify the path string matches your Drive folders exactly. |
| `RuntimeError: CUDA out of memory` | Model inputs are too large or GPU ran out of space | Restart your runtime, decrease crop sizes, or ensure you are running in `with torch.no_grad():` during evaluation. |
| `ModuleNotFoundError: No module named 'boto3'` | Forgot to run the install dependencies step | Run `!pip install -r requirements.txt` again. |
| Model predictions look like random noise | Training checkpoint is missing or failed to load | Check if the `.pth` file exists in your checkpoints folder and verify the path in `configs/inference.yaml`. |
| Execution is very slow (hours per epoch) | The Colab session is running on CPU rather than GPU | Run Step 2 to enable GPU acceleration, then restart your runtime. |

---

### Step 10: Resetting Colab Safely
If your Colab notebook hangs, runs out of memory, or crashes:
1.  Click **Runtime** in the top menu $\rightarrow$ **Restart session**.
2.  This resets Python's memory but does not delete files on your Google Drive.
3.  **To resume training from a checkpoint:** update your training script to load the epoch `.pth` file weights using `model.load_state_dict()` before entering the training loop.

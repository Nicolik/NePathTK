# Installation Guide for NePathTK

This document provides instructions to set up **NePathTK (NephroPathology Toolkit)**, including environment preparation and configuration of required files.

---

## 1. Prerequisites

- Python 3.7 (recommended)  
- Conda for environment management  
- QuPath (https://github.com/qupath/qupath) installed on your system  
- OpenSlide (https://openslide.org/) binaries available locally (for use with `openslide-python`)  

---

## 2. Clone the Repository

```
git clone https://github.com/Nicolik/NePathTK.git
cd NePathTK
```

---

## 3. Python Environment Setup

We recommend creating a new environment:

```
conda create -n nepathtk python=3.7.16 -y
conda activate nepathtk
```

Then install required dependencies:

```
pip install numpy cached-property
pip install torch==1.13.1 torchvision==0.14.1 --index-url https://download.pytorch.org/whl/cu117
pip install -r requirements.txt
```

Install `mmsegmentation` dependencies:

```
pip install -U openmim
mim install mmengine
mim install "mmcv==2.0.0"
pip install "mmsegmentation>=1.0.0"
mim install mmdet
pip install ftfy
```

---

## 4. Configuration Files

After installation, some `.base` configuration files must be adapted to your local setup:

### 4.1 definitions.py
- Copy from the provided base file:

```
cp definitions.py.base definitions.py
```

- Edit `definitions.py` and provide valid paths for:
  - `DATASET_DIR` → directory where the classification dataset is stored  
  - `OPENSLIDE` → directory containing OpenSlide binaries (needed for smooth integration with `openslide-python`)  

### 4.2 config_semseg.py
- Copy from the base file:

```
cp config_semseg.py.base config_semseg.py
```

- Edit `config_semseg.py` and define the class `SelectedConfig`, including the paths.


### 4.3 .paquo.toml
- Copy from the base file:

```
cp .paquo.toml.base .paquo.toml
```

- Edit `.paquo.toml` and set the path to your QuPath executable.

---

## 5. Verification

1. Check that all three files exist and are configured:
   - `definitions.py`
   - `config_semseg.py`
   - `.paquo.toml`

2. You can also verify that everything went smoothly with the `install_check` script:

```
python install_check.py"
```

## ✅ Installation Complete

You are now ready to use NePathTK for multistain multicompartment segmentation and classification workflows in nephropathology.

For details on usage, please refer to the project documentation and examples.

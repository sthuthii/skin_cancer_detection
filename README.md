# Federated Learning for Skin Lesion Classification

## Overview

This project implements a **Federated Learning (FL)** system for skin lesion classification using deep learning. Instead of training on centralized data, multiple clients (simulated hospitals/devices) train locally and share only model updates with a central server.

The system is built using **PyTorch** and **Flower (FL framework)**.

---

## Objectives

* Build a scalable **Federated Learning pipeline**
* Compare **Centralized vs Federated Learning**
* Study impact of **IID vs Non-IID data distribution**
* Improve model performance using **attention mechanisms**
* Visualize model decisions using **Grad-CAM**

---

## Dataset

* Primary dataset: HAM10000
* Optional extension: ISIC Archive

### Important

The dataset is **NOT included in this repository** due to size constraints.

### Download Instructions

1. Download HAM10000 dataset
2. Extract it
3. Place it in:

```
data/HAM10000/
```

Expected structure:

```
data/HAM10000/
├── HAM10000_metadata.csv
├── HAM10000_images_part_1/
├── HAM10000_images_part_2/
```

---

## Project Structure

```
major_project/
│
├── data/                     # (ignored in git)
│   └── HAM10000/
│
├── src/                      # Core ML code
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   ├── validate.py
│   ├── utils.py
│   └── config.py
│
├── fl/                       # Federated Learning
│   ├── simulation.py
│   ├── client.py
│   ├── client_app.py
│   └── server.py
│
├── checkpoints/              # Saved models (ignored)
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Installation

### 1. Clone the repository

```bash
git clone <your-repo-link>
cd major_project
```

### 2. Create virtual environment

```bash
python -m venv venv
venv\Scripts\activate   # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## How to Run

---

### Option 1: Simulation Mode (Recommended)

Runs all clients in a single process.

```bash
python -m fl.simulation
```

✔ Easy debugging
✔ Fast experimentation

---

### Option 2: Multi-Terminal Federated Setup

#### Step 1: Start Server

```bash
python fl/server.py
```

#### Step 2: Start Clients (in separate terminals)

```bash
python fl/client_app.py 0
python fl/client_app.py 1
python fl/client_app.py 2
```

✔ Simulates real federated environment
✔ Can run across multiple systems

---

## Model Architecture

* Backbone: **MobileNetV3**
* Attention mechanism for feature weighting
* Final classification head for 7 classes

---

## Federated Learning Workflow

1. Server initializes global model
2. Clients receive model
3. Each client trains locally on its dataset
4. Clients send updated weights to server
5. Server aggregates (FedAvg)
6. Repeat for multiple rounds

---

## Evaluation Metrics

* **Accuracy**
* **F1 Score (Macro)**
* **AUC (ROC)** ← primary metric

---

## Experiments

### 1. Centralized vs Federated

* Train model on full dataset vs FL setup

### 2. Data Distribution

* IID split
* Non-IID split (label-based)

### 3. Model Variants

* With attention
* Without attention

---

## Visualization

* AUC vs Rounds
* Client-wise performance
* Grad-CAM heatmaps for interpretability

---

## Team Responsibilities

### Model & ML

* Model design
* Training optimization
* Metrics & evaluation
* Grad-CAM

### FL & System

* Federated setup
* Client-server communication
* Data splitting
* Experiment logging

---

## Important Notes

* Dataset is not included in repo
* Use **relative paths only**
* Do not commit:

  * `data/`
  * `checkpoints/`
  * `.pt` files

---

## Key Learnings

* Federated Learning enables privacy-preserving training
* Non-IID data significantly impacts performance
* Lightweight models are essential for distributed systems

---

## Future Work

* Add ISIC Archive
* Implement advanced FL algorithms (FedProx, FedAvgM)
* Deploy on multiple real devices
* Build web interface for predictions

---

## License

This project is for academic and research purposes.

---

## Acknowledgements

* PyTorch
* Flower (Federated Learning Framework)
* HAM10000 Dataset

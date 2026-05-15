# ECG Deep Learning Dashboard

An interactive Dash + Plotly dashboard for ECG arrhythmia classification and myocardial infarction (MI) detection using a 1-D ResNet trained on MIT-BIH and PTB datasets.

---

## Screenshots

| Live Inference | Model Overview |
|---|---|
| ![Live Inference](assets/screenshot_inference.png) | ![Model Overview](assets/screenshot_overview.png) |

| Training Curves | Ablation Study |
|---|---|
| ![Training Curves](assets/screenshot_training.png) | ![Ablation Study](assets/screenshot_ablation.png) |

### Data Pipeline
![Data Pipeline](assets/screenshot_pipeline.png)

---

## Features

| Tab | Description |
|-----|-------------|
| **Live Inference** | Upload a `.npy` beat or `.csv` signal, or generate a synthetic demo beat, then run the model and inspect the waveform + confidence bars |
| **Model Overview** | Architecture diagram (interactive block count), parameter breakdown, transfer-learning setup |
| **Training** | Simulated or loaded training curves; dataset split statistics and class distribution |
| **Ablation Study** | Side-by-side config comparison table and bar chart |
| **Data Pipeline** | Preprocessing flow diagram, AAMI class map, and signal parameters |

---

## Project Structure

```
ecg_dashboard/
├── app.py                  # Entry point — Dash app init, sidebar, root layout
├── callbacks.py            # All Dash callbacks (routing, inference, training, etc.)
├── config.py               # Shared constants (classes, colors, ablation configs)
├── figures.py              # Plotly figure builders
├── layout_helpers.py       # Reusable UI components (card, metric_card, section_header)
├── model.py                # PyTorch model definitions (ECGResNet, ECGResNetTransfer)
├── utils.py                # Signal processing, model loading, synthetic data generators
├── pages/
│   ├── __init__.py
│   ├── inference.py        # Live Inference page layout
│   ├── overview.py         # Model Overview page layout
│   ├── training.py         # Training page layout
│   ├── ablation.py         # Ablation Study page layout
│   └── pipeline.py         # Data Pipeline page layout
├── assets/
│   ├── screenshot_inference.png
│   ├── screenshot_overview.png
│   ├── screenshot_training.png
│   ├── screenshot_ablation.png
│   └── screenshot_pipeline.png
└── requirements.txt
```

---

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/your-username/ecg-dashboard.git
cd ecg-dashboard

# 2. Create and activate a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## Usage

```bash
python app.py
```

Then open **http://127.0.0.1:8050** in your browser.

---

## Loading Real Model Checkpoints

The dashboard works out-of-the-box with randomly-initialised weights for UI demonstration. To get meaningful predictions:

1. Train the arrhythmia model and save a checkpoint (e.g. `output/arrhy_best.pt`).
2. Train the MI transfer model (e.g. `output/transfer_best.pt`).
3. In the **Live Inference** tab, enter the checkpoint paths and click **Load model**.

Expected checkpoint format:

```python
# Arrhythmia
torch.save({"model_state": model.state_dict(), "best_val_acc": acc, "epoch": epoch}, path)

# MI transfer
torch.save({"model_state": model.state_dict(), "best_acc": acc}, path)
```

---

## Model Architecture

```
Input (1×300)
    │
Conv1d k=5, 32ch
    │
ResidualBlock ×N  (each: Conv→Conv→ReLU→MaxPool stride-2)
    │
AdaptiveAvgPool1d → 32-d vector
    │
FC(32→32) → ReLU → FC(32→32) → ReLU → FC(32→num_classes)
```

- **Arrhythmia model**: 5-class (N, S, V, F, Q) trained on MIT-BIH
- **MI model**: 2-class (Normal, MI) fine-tuned on PTB with frozen backbone

---

## Datasets

| Dataset | Task | Classes |
|---------|------|---------|
| [MIT-BIH Arrhythmia](https://physionet.org/content/mitdb/) | Arrhythmia classification | N, S, V, F, Q (AAMI) |
| [PTB Diagnostic ECG](https://physionet.org/content/ptbdb/) | MI detection | Normal, MI |

---

## Dependencies

- [Dash](https://dash.plotly.com/) + [dash-bootstrap-components](https://dash-bootstrap-components.opensource.faculty.ai/)
- [Plotly](https://plotly.com/python/)
- [PyTorch](https://pytorch.org/)
- [SciPy](https://scipy.org/)
- [NumPy](https://numpy.org/)

---

## Paper Targets

| Task | Accuracy |
|------|----------|
| Arrhythmia (MIT-BIH, balanced test) | **93.4%** |
| MI detection (PTB) | **95.9%** |

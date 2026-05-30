# Surface Defect Detection

AI-powered detection of manufacturing defects using deep learning with PyTorch.

## Project Overview

This project implements a surface defect detection system for manufacturing quality control. It includes:
- **Baseline CNN model** for defect classification
- **Transfer Learning models** (ResNet18, ResNet50, VGG16)
- **Grad-CAM visualization** for model interpretability
- **Streamlit web interface** for easy inference

## Classes
- Crack
- Dent
- No Defect
- Scratch

## Project Structure

```
surface_defect/
│
├── Docs/
│   ├── ppt/                          # Presentations
│   │   └── hcl_project_review.pptx
│   ├── problem_statement/             # Problem documentation
│   │   └── problem_statement.docx
│   ├── results/
│   │   └── snapshots/                # Result screenshots
│   └── dataset/                      # Dataset files (CSV, XLSX, etc.)
│
├── main_code/
│   ├── backend/
│   │   ├── src/                      # Core ML pipeline
│   │   │   ├── main.py               # CLI entry point
│   │   │   ├── train.py              # Training logic
│   │   │   ├── evaluate.py           # Evaluation metrics
│   │   │   ├── data_loader.py        # Data handling
│   │   │   ├── preprocess.py         # Preprocessing
│   │   │   ├── gradcam.py            # Model visualization
│   │   │   ├── model_baseline.py     # Baseline CNN
│   │   │   ├── model_transfer.py     # Transfer learning
│   │   │   ├── utils.py              # Utilities
│   │   │   └── __init__.py
│   │   ├── models/                   # Pre-trained weights
│   │   │   ├── baseline_cnn_last.pth
│   │   │   ├── best_model.pth
│   │   │   ├── transfer_resnet18_last.pth
│   │   │   ├── transfer_resnet50_last.pth
│   │   │   └── transfer_vgg16_last.pth
│   │   └── requirements.txt           # Python dependencies
│   │
│   └── frontend/
│       └── app.py                    # Streamlit web UI
│
└── README.md
```

## Installation

```bash
pip install -r main_code/backend/requirements.txt
```

## Usage

### Web Interface (Streamlit)
```bash
cd main_code
streamlit run frontend/app.py
```

### Command Line Interface
```bash
cd main_code/backend
python src/main.py train --data_dir data/ --epochs 30
python src/main.py evaluate --data_dir data/
python src/main.py gradcam --image_dir data/test/ --model_path models/best_model.pth
```

## Models

### Baseline CNN
A custom convolutional neural network trained from scratch.

### Transfer Learning
- ResNet18
- ResNet50
- VGG16

All transfer models are fine-tuned on the surface defect dataset.

## Features

- **Multi-model support** - Compare different architectures
- **Binary & Multi-class modes** - Choose classification granularity
- **Grad-CAM visualization** - Understand model decisions
- **Confidence thresholding** - Control prediction sensitivity
- **Batch processing** - CLI support for evaluating multiple images

## Results

See `Docs/results/snapshots/` for sample predictions and visualizations.

## Requirements

- Python 3.8+
- PyTorch 2.0+
- Streamlit
- OpenCV
- PIL, NumPy, Pandas

See `main_code/backend/requirements.txt` for full list.

## Author

Aravindan V

## License

This project is provided as-is for educational purposes.

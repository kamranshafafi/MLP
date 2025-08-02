# Multi-Layer Perceptron for Kuzushiji-MNIST Classification

A comprehensive study comparing optimization algorithms (Adam vs SGD) for Japanese character recognition using the Kuzushiji-MNIST dataset. This project implements and evaluates MLP architectures with systematic hyperparameter tuning and early stopping strategies.

## 📑 Table of Contents

1. [Results Summary](#-results-summary)
2. [Quick Start](#-quick-start)
   - [Prerequisites](#prerequisites)
   - [Installation](#installation)
   - [Run Complete Pipeline](#run-complete-pipeline)
3. [Project Structure](#-project-structure)
4. [Methodology](#-methodology)
   - [Data Preprocessing](#data-preprocessing)
   - [Hyperparameter Search Space](#hyperparameter-search-space)
   - [Training Strategy](#training-strategy)
5. [Key Findings](#-key-findings)
   - [Optimizer Comparison](#optimizer-comparison)
   - [Hyperparameter Insights](#hyperparameter-insights)
   - [Confusion Analysis](#confusion-analysis)
6. [Individual Script Usage](#-individual-script-usage)
7. [Visualizations](#-visualizations)
8. [Reproducing Results](#-reproducing-results)
9. [Citation](#-citation)
10. [References](#-references)
11. [License](#-license)
12. [Contact](#-Contact)


## 📊 Results Summary

- **Best Model Performance**: 96.42% test accuracy
- **Architecture**: 2-layer MLP (784 → 512 → 256 → 10)
- **Optimizer**: Adam (lr=0.001)
- **Training Time**: ~3 epochs to 95% accuracy

## 🚀 Quick Start

### Prerequisites
```bash
python >= 3.7
pytorch >= 1.9.0
numpy
matplotlib
scikit-learn
seaborn
```

### Installation
```bash
# Clone repository
git clone https://github.com/[username]/mlp-kuzushiji-mnist.git
cd mlp-kuzushiji-mnist

# Install dependencies
pip install torch torchvision numpy matplotlib scikit-learn seaborn
```

### Run Complete Pipeline
```bash
# Make script executable
chmod +x run_all.sh

# Run all experiments
./run_all.sh
```

This will:
1. Download the KMNIST dataset
2. Train baseline model with Adam optimizer
3. Train comparison model with SGD
4. Run hyperparameter tuning (54 configurations)
5. Perform final comparison with optimal settings
6. Generate confusion matrices and analysis plots

## 📁 Project Structure

```
mlp-kuzushiji-mnist/
│
├── Scripts/
│   ├── preparation.py           # Data download and preprocessing
│   ├── mlp_model.py            # MLP architecture and training functions
│   ├── train_baseline.py       # Adam optimizer implementation
│   ├── train_sgd.py            # SGD optimizer implementation
│   ├── hyperparameter_tuning.py # Grid search across configurations
│   ├── final_comparison.py     # Comparative analysis
│   ├── both_confusion_matrices.py # Confusion matrix generation
│   └── run_all.sh              # Automated pipeline script
│
├── Results/
│   ├── baseline_learning_curves.png    # Adam training curves
│   ├── sgd_learning_curves.png        # SGD training curves
│   ├── hyperparameter_analysis.png    # Hyperparameter impact visualization
│   ├── final_comparison_analysis.png  # Comprehensive comparison
│   ├── both_confusion_matrices.png    # Classification patterns
│   └── final_results_summary.txt      # Numerical results
│
├── Models/
│   ├── best_adam_model.pth    # Best Adam checkpoint
│   ├── best_sgd_model.pth     # Best SGD checkpoint
│   ├── final_adam_model.pth   # Final comparison Adam model
│   └── final_sgd_model.pth    # Final comparison SGD model
│
└── Report/
    └── IEEE_Report.pdf        # Comprehensive analysis and findings
```

## 🔬 Methodology

### Data Preprocessing
- **Normalization**: Pixel values scaled to [0, 1]
- **Flattening**: 28×28 images → 784-dimensional vectors
- **Train/Val/Test Split**: 54,000/6,000/10,000 images

### Hyperparameter Search Space
- **Architectures**: [128], [256], [512], [256,128], [512,256], [512,256,128]
- **Learning Rates**: 0.001, 0.01, 0.1
- **Batch Sizes**: 32, 64, 128
- **Total Configurations**: 54

### Training Strategy
- **Early Stopping**: Patience of 10 epochs
- **Maximum Epochs**: 50
- **Validation**: 10% of training data
- **Best Model Selection**: Lowest validation loss

## 📈 Key Findings

### Optimizer Comparison
| Metric | Adam | SGD |
|--------|------|-----|
| Test Accuracy | 96.42% | 95.20% |
| Epochs to 95% | 3 | 8 |
| Best Epoch | 3 | 8 |
| Total Epochs | 14 | 19 |

### Hyperparameter Insights
1. **Learning Rate**: 0.001-0.01 optimal; 0.1 causes instability
2. **Architecture**: 2-layer networks outperform 1 or 3 layers
3. **Batch Size**: Minimal impact (±1% accuracy)

### Confusion Analysis
- **Best Classes**: 0, 3, 9 (>94% accuracy)
- **Most Confused**: Classes 2↔5 (visual similarity)
- **Consistent Patterns**: Both optimizers show similar confusion pairs

## 🛠️ Individual Script Usage

### Data Preparation
```bash
python preparation.py
```
Downloads and preprocesses the KMNIST dataset.

### Train Models
```bash
# Train with Adam optimizer
python train_baseline.py

# Train with SGD optimizer  
python train_sgd.py
```

### Hyperparameter Tuning
```bash
python hyperparameter_tuning.py
```
Tests 54 configurations and saves results to `hyperparameter_results.json`.

### Generate Analysis
```bash
# Final comparison with best hyperparameters
python final_comparison.py

# Create confusion matrices
python both_confusion_matrices.py
```

## 📊 Visualizations

All experiments generate comprehensive visualizations:

1. **Learning Curves**: Training/validation loss and accuracy
2. **Hyperparameter Analysis**: Impact of each parameter on performance
3. **Convergence Comparison**: Adam vs SGD training dynamics
4. **Confusion Matrices**: Classification patterns for both optimizers

## 🎯 Reproducing Results

To reproduce the exact results:

1. Ensure random seeds are set (already included in scripts)
2. Use the same package versions listed above
3. Run on similar hardware (GPU recommended but not required)

## 📝 Citation

If you use this code or methodology, please cite:

```bibtex
@misc{kmnist-mlp-2025,
  author = {Kamran SHafafi},
  title = {Multi-Layer Perceptron Classification of Kuzushiji-MNIST},
  year = {2025},
  url = {https://github.com/kamranshafafi/MLP}
}
```

## 📚 References

- Clanuwat et al., "Deep Learning for Classical Japanese Literature", arXiv:1812.01718, 2018
- [Kuzushiji-MNIST Dataset](https://github.com/rois-codh/kmnist)

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contact
Author: Kamran Shafafi
Email: kamranshafafi@gmail.com
Project Link: https://github.com/kamranshafafi/MLP

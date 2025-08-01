#!/bin/bash

echo "========================================="
echo "Running Complete MLP Pipeline"
echo "========================================="

# Step 1: Download and prepare data
echo -e "\n[Step 1/6] Downloading and preparing data..."
python preparation.py

# Step 2: Train baseline model with Adam
echo -e "\n[Step 2/6] Training baseline model (Adam)..."
python train_baseline.py

# Step 3: Train SGD model
echo -e "\n[Step 3/6] Training SGD model..."
python train_sgd.py

# Step 4: Run hyperparameter tuning
echo -e "\n[Step 4/6] Running hyperparameter tuning (this may take 10-20 minutes)..."
python hyperparameter_tuning.py

# Step 5: Final comparison with best hyperparameters
echo -e "\n[Step 5/6] Running final comparison..."
python final_comparison.py

# Step 6: Generate confusion matrices
echo -e "\n[Step 6/6] Generating confusion matrices..."
python both_confusion_matrices.py

echo -e "\n========================================="
echo "Pipeline Complete!"
echo "========================================="
echo -e "\nGenerated files:"
echo "- Images: *.png"
echo "- Models: *.pth"
echo "- Results: final_results_summary.txt, hyperparameter_results.json"
echo -e "\nAll results ready for report writing!"
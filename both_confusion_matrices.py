import torch
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from preparation import load_data, preprocess
from mlp_model import MLP, prepare_data

# Load data
X_train, y_train, X_test, y_test = load_data()
X_train, X_test = preprocess(X_train, X_test)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Prepare test data
_, _, test_loader = prepare_data(X_train, y_train, X_test, y_test, batch_size=64)

# Function to get predictions
def get_predictions(model_path, hidden_sizes=[512, 256]):
    model = MLP(input_size=784, hidden_sizes=hidden_sizes, num_classes=10).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(y_batch.numpy())
    
    return np.array(all_preds), np.array(all_labels)

# Get predictions for both models
preds_adam, labels = get_predictions('final_adam_model.pth')
preds_sgd, _ = get_predictions('final_sgd_model.pth')

# Create confusion matrices
cm_adam = confusion_matrix(labels, preds_adam)
cm_sgd = confusion_matrix(labels, preds_sgd)

# Plot both confusion matrices
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Adam confusion matrix
sns.heatmap(cm_adam, annot=True, fmt='d', cmap='Blues', ax=ax1)
ax1.set_xlabel('Predicted')
ax1.set_ylabel('Actual')
ax1.set_title('Confusion Matrix - Adam Optimizer')

# SGD confusion matrix
sns.heatmap(cm_sgd, annot=True, fmt='d', cmap='Reds', ax=ax2)
ax2.set_xlabel('Predicted')
ax2.set_ylabel('Actual')
ax2.set_title('Confusion Matrix - SGD Optimizer')

plt.tight_layout()
plt.savefig('both_confusion_matrices.png', dpi=300, bbox_inches='tight')
plt.show()

# Calculate per-class metrics
print("\nPer-Class Accuracy Comparison:")
print("Class | Adam  | SGD   | Difference")
print("------|-------|-------|------------")
for i in range(10):
    adam_acc = cm_adam[i, i] / cm_adam[i].sum()
    sgd_acc = cm_sgd[i, i] / cm_sgd[i].sum()
    diff = adam_acc - sgd_acc
    print(f"  {i}   | {adam_acc:.3f} | {sgd_acc:.3f} | {diff:+.3f}")

# Find most confused pairs
print("\nMost Confused Class Pairs (Adam):")
cm_adam_norm = cm_adam.astype('float') / cm_adam.sum(axis=1)[:, np.newaxis]
np.fill_diagonal(cm_adam_norm, 0)
for _ in range(5):
    i, j = np.unravel_index(cm_adam_norm.argmax(), cm_adam_norm.shape)
    print(f"Class {i} → {j}: {cm_adam[i,j]} samples ({cm_adam_norm[i,j]:.1%})")
    cm_adam_norm[i, j] = 0
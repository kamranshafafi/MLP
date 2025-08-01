import torch
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
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
def get_predictions(model, loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(y_batch.numpy())
    
    return np.array(all_preds), np.array(all_labels)

# Evaluate Adam model
print("Evaluating Adam model...")
model_adam = MLP(input_size=784, hidden_sizes=[256, 128], num_classes=10).to(device)
# Note: You need to save the Adam model in train_baseline.py first
# Add this line before plt.show() in train_baseline.py:
# torch.save(model.state_dict(), 'best_adam_model.pth')

# For now, train a quick Adam model
optimizer = torch.optim.Adam(model_adam.parameters(), lr=0.001)
# ... or load if you saved it

# Evaluate SGD model
print("Evaluating SGD model...")
model_sgd = MLP(input_size=784, hidden_sizes=[256, 128], num_classes=10).to(device)
model_sgd.load_state_dict(torch.load('best_sgd_model.pth'))

preds_sgd, labels = get_predictions(model_sgd, test_loader, device)

# Create confusion matrix
cm = confusion_matrix(labels, preds_sgd)

# Plot confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - SGD Model')
plt.savefig('confusion_matrix_sgd.png', dpi=300, bbox_inches='tight')
plt.show()

# Classification report
print("\nClassification Report - SGD:")
print(classification_report(labels, preds_sgd))

# Class-wise accuracy
class_acc = cm.diagonal() / cm.sum(axis=1)
print("\nPer-class accuracy:")
for i, acc in enumerate(class_acc):
    print(f"Class {i}: {acc:.3f}")
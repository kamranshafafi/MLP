import torch
import torch.nn as nn
from preparation import load_data, preprocess
from mlp_model import MLP, prepare_data, train_epoch, evaluate
import matplotlib.pyplot as plt

# Load data
print("Loading data...")
X_train, y_train, X_test, y_test = load_data()
X_train, X_test = preprocess(X_train, X_test)

# Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Prepare data
train_loader, val_loader, test_loader = prepare_data(
    X_train, y_train, X_test, y_test, val_split=0.1, batch_size=64
)

# Create model with same architecture
model = MLP(input_size=784, hidden_sizes=[256, 128], num_classes=10).to(device)
criterion = nn.CrossEntropyLoss()

# SGD optimizer with momentum
optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

# Training with learning rate scheduling
epochs = 30
train_losses, val_losses = [], []
train_accs, val_accs = [], []
best_val_acc = 0
patience = 5
patience_counter = 0

print("Starting SGD training...")
for epoch in range(epochs):
    train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
    val_loss, val_acc = evaluate(model, val_loader, criterion, device)
    
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_accs.append(train_acc)
    val_accs.append(val_acc)
    
    # Early stopping
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), 'best_sgd_model.pth')
        patience_counter = 0
    else:
        patience_counter += 1
    
    print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
    
    if patience_counter >= patience:
        print(f"Early stopping at epoch {epoch+1}")
        break

# Load best model
model.load_state_dict(torch.load('best_sgd_model.pth'))

# Test evaluation
test_loss, test_acc = evaluate(model, test_loader, criterion, device)
print(f"\nTest Accuracy: {test_acc:.4f}")

# Plot learning curves
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train')
plt.plot(val_losses, label='Validation')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('SGD Learning Curves - Loss')

plt.subplot(1, 2, 2)
plt.plot(train_accs, label='Train')
plt.plot(val_accs, label='Validation')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('SGD Learning Curves - Accuracy')
plt.savefig('sgd_learning_curves.png', dpi=300, bbox_inches='tight')
plt.show()
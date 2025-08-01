import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from preparation import load_data, preprocess
from mlp_model import MLP, prepare_data, train_epoch, evaluate
from sklearn.metrics import confusion_matrix

# Load data
X_train, y_train, X_test, y_test = load_data()
X_train, X_test = preprocess(X_train, X_test)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Best configuration from hyperparameter search
best_config = {
    'hidden_sizes': [512, 256],  # or [256, 128] based on your results
    'batch_size': 64
}

# Prepare data
train_loader, val_loader, test_loader = prepare_data(
    X_train, y_train, X_test, y_test, 
    val_split=0.1, 
    batch_size=best_config['batch_size']
)

# Train both models with early stopping tracking
def train_with_early_stopping(optimizer_type, lr, max_epochs=50):
    # Create model
    model = MLP(input_size=784, 
                hidden_sizes=best_config['hidden_sizes'], 
                num_classes=10).to(device)
    
    criterion = nn.CrossEntropyLoss()
    
    if optimizer_type == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    else:  # sgd
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    
    # Training history
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    
    best_val_loss = float('inf')
    best_epoch = 0
    patience = 10
    patience_counter = 0
    
    print(f"\nTraining {optimizer_type.upper()} model...")
    
    for epoch in range(max_epochs):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            torch.save(model.state_dict(), f'final_{optimizer_type}_model.pth')
            patience_counter = 0
        else:
            patience_counter += 1
        
        if epoch % 5 == 0:
            print(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}")
        
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch} (best was epoch {best_epoch})")
            break
    
    # Load best model
    model.load_state_dict(torch.load(f'final_{optimizer_type}_model.pth'))
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    
    return {
        'model': model,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accs': train_accs,
        'val_accs': val_accs,
        'best_epoch': best_epoch,
        'test_acc': test_acc,
        'total_epochs': len(train_losses)
    }

# Train both models
adam_results = train_with_early_stopping('adam', lr=0.001)
sgd_results = train_with_early_stopping('sgd', lr=0.01)  # Lower LR for SGD based on results

# Visualization
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# 1. Loss curves comparison
ax = axes[0, 0]
ax.plot(adam_results['train_losses'], 'b-', label='Adam Train', alpha=0.7)
ax.plot(adam_results['val_losses'], 'b--', label='Adam Val', alpha=0.7)
ax.plot(sgd_results['train_losses'], 'r-', label='SGD Train', alpha=0.7)
ax.plot(sgd_results['val_losses'], 'r--', label='SGD Val', alpha=0.7)
ax.axvline(adam_results['best_epoch'], color='b', linestyle=':', label=f"Adam best ({adam_results['best_epoch']})")
ax.axvline(sgd_results['best_epoch'], color='r', linestyle=':', label=f"SGD best ({sgd_results['best_epoch']})")
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.set_title('Training Progress Comparison')
ax.legend()
ax.grid(True, alpha=0.3)

# 2. Accuracy curves
ax = axes[0, 1]
ax.plot(adam_results['val_accs'], 'b-', label='Adam', linewidth=2)
ax.plot(sgd_results['val_accs'], 'r-', label='SGD', linewidth=2)
ax.axvline(adam_results['best_epoch'], color='b', linestyle=':', alpha=0.5)
ax.axvline(sgd_results['best_epoch'], color='r', linestyle=':', alpha=0.5)
ax.set_xlabel('Epoch')
ax.set_ylabel('Validation Accuracy')
ax.set_title('Validation Accuracy Over Time')
ax.legend()
ax.grid(True, alpha=0.3)

# 3. Convergence speed
ax = axes[0, 2]
epochs = list(range(min(len(adam_results['val_accs']), len(sgd_results['val_accs']))))
adam_improvement = [adam_results['val_accs'][i] - adam_results['val_accs'][0] for i in epochs]
sgd_improvement = [sgd_results['val_accs'][i] - sgd_results['val_accs'][0] for i in epochs]
ax.plot(epochs, adam_improvement, 'b-', label='Adam', linewidth=2)
ax.plot(epochs, sgd_improvement, 'r-', label='SGD', linewidth=2)
ax.set_xlabel('Epoch')
ax.set_ylabel('Improvement from Start')
ax.set_title('Convergence Speed')
ax.legend()
ax.grid(True, alpha=0.3)

# 4. Early stopping analysis
ax = axes[1, 0]
# Calculate validation loss smoothness (rolling std)
window = 5
adam_std = [np.std(adam_results['val_losses'][max(0,i-window):i+1]) 
            for i in range(len(adam_results['val_losses']))]
sgd_std = [np.std(sgd_results['val_losses'][max(0,i-window):i+1]) 
           for i in range(len(sgd_results['val_losses']))]
ax.plot(adam_std[window:], 'b-', label='Adam', alpha=0.7)
ax.plot(sgd_std[window:], 'r-', label='SGD', alpha=0.7)
ax.set_xlabel('Epoch')
ax.set_ylabel('Validation Loss Variance')
ax.set_title('Training Stability (Lower is Better)')
ax.legend()
ax.grid(True, alpha=0.3)

# 5. Final comparison
ax = axes[1, 1]
metrics = ['Test Accuracy', 'Best Epoch', 'Total Epochs', 'Convergence']
adam_vals = [
    adam_results['test_acc'],
    adam_results['best_epoch'] / adam_results['total_epochs'],  # Normalized
    1 - (adam_results['total_epochs'] / 50),  # Normalized (inverse)
    adam_results['val_accs'][min(10, len(adam_results['val_accs'])-1)]  # Acc at epoch 10
]
sgd_vals = [
    sgd_results['test_acc'],
    sgd_results['best_epoch'] / sgd_results['total_epochs'],
    1 - (sgd_results['total_epochs'] / 50),
    sgd_results['val_accs'][min(10, len(sgd_results['val_accs'])-1)]
]

x = np.arange(len(metrics))
width = 0.35
ax.bar(x - width/2, adam_vals, width, label='Adam', color='blue', alpha=0.7)
ax.bar(x + width/2, sgd_vals, width, label='SGD', color='red', alpha=0.7)
ax.set_xlabel('Metrics')
ax.set_ylabel('Normalized Values')
ax.set_title('Overall Comparison')
ax.set_xticks(x)
ax.set_xticklabels(metrics, rotation=15)
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

# 6. Summary statistics
ax = axes[1, 2]
ax.axis('off')
summary_text = f"""
Final Results Summary:

Adam Optimizer:
- Test Accuracy: {adam_results['test_acc']:.4f}
- Best Epoch: {adam_results['best_epoch']}
- Total Epochs: {adam_results['total_epochs']}
- Early Stop: {'Yes' if adam_results['total_epochs'] < 50 else 'No'}

SGD Optimizer:
- Test Accuracy: {sgd_results['test_acc']:.4f}
- Best Epoch: {sgd_results['best_epoch']}
- Total Epochs: {sgd_results['total_epochs']}
- Early Stop: {'Yes' if sgd_results['total_epochs'] < 50 else 'No'}

Winner: {'Adam' if adam_results['test_acc'] > sgd_results['test_acc'] else 'SGD'}
Difference: {abs(adam_results['test_acc'] - sgd_results['test_acc']):.4f}
"""
ax.text(0.1, 0.9, summary_text, transform=ax.transAxes, 
        fontsize=10, verticalalignment='top', fontfamily='monospace')

plt.tight_layout()
plt.savefig('final_comparison_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# Save summary for report
with open('final_results_summary.txt', 'w') as f:
    f.write(summary_text)
    f.write(f"\n\nDetailed Analysis:\n")
    f.write(f"Adam converged at epoch {adam_results['best_epoch']} with {adam_results['test_acc']:.4f} accuracy\n")
    f.write(f"SGD converged at epoch {sgd_results['best_epoch']} with {sgd_results['test_acc']:.4f} accuracy\n")
    f.write(f"Adam showed {'better' if adam_results['test_acc'] > sgd_results['test_acc'] else 'worse'} generalization\n")
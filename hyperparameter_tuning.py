import torch
import torch.nn as nn
import numpy as np
from preparation import load_data, preprocess
from mlp_model import MLP, prepare_data, train_epoch, evaluate
import matplotlib.pyplot as plt
from itertools import product

# Load data once
X_train, y_train, X_test, y_test = load_data()
X_train, X_test = preprocess(X_train, X_test)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameter grid
hp_grid = {
    'hidden_sizes': [[128], [256], [512], [256, 128], [512, 256], [512, 256, 128]],
    'learning_rate': [0.001, 0.01, 0.1],
    'batch_size': [32, 64, 128]
}

results = []

# Grid search
print("Starting hyperparameter search...")
for hidden, lr, batch in product(hp_grid['hidden_sizes'], 
                                hp_grid['learning_rate'], 
                                hp_grid['batch_size']):
    
    print(f"\nTesting: hidden={hidden}, lr={lr}, batch={batch}")
    
    # Prepare data with current batch size
    train_loader, val_loader, test_loader = prepare_data(
        X_train, y_train, X_test, y_test, val_split=0.1, batch_size=batch
    )
    
    # Create model
    model = MLP(input_size=784, hidden_sizes=hidden, num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Train for fewer epochs for grid search
    best_val_acc = 0
    for epoch in range(10):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
    
    # Test accuracy
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    
    results.append({
        'hidden': str(hidden),
        'lr': lr,
        'batch_size': batch,
        'val_acc': best_val_acc,
        'test_acc': test_acc,
        'params': sum(p.numel() for p in model.parameters())
    })
    
    print(f"Val Acc: {best_val_acc:.4f}, Test Acc: {test_acc:.4f}")

# Sort by validation accuracy
results.sort(key=lambda x: x['val_acc'], reverse=True)

# Print top 5 configurations
print("\n=== Top 5 Configurations ===")
for i, r in enumerate(results[:5]):
    print(f"{i+1}. Hidden: {r['hidden']}, LR: {r['lr']}, Batch: {r['batch_size']}")
    print(f"   Val: {r['val_acc']:.4f}, Test: {r['test_acc']:.4f}, Params: {r['params']:,}")

# Visualize results
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 1. Learning rate impact
ax = axes[0, 0]
for lr in hp_grid['learning_rate']:
    lr_results = [r for r in results if r['lr'] == lr]
    ax.scatter([r['params'] for r in lr_results], 
              [r['test_acc'] for r in lr_results], 
              label=f'LR={lr}', alpha=0.7)
ax.set_xlabel('Number of Parameters')
ax.set_ylabel('Test Accuracy')
ax.set_title('Learning Rate Impact')
ax.legend()

# 2. Architecture depth
ax = axes[0, 1]
depth_acc = {}
for r in results:
    depth = len(eval(r['hidden']))
    if depth not in depth_acc:
        depth_acc[depth] = []
    depth_acc[depth].append(r['test_acc'])

depths = sorted(depth_acc.keys())
means = [np.mean(depth_acc[d]) for d in depths]
stds = [np.std(depth_acc[d]) for d in depths]

ax.errorbar(depths, means, yerr=stds, marker='o', capsize=5)
ax.set_xlabel('Number of Hidden Layers')
ax.set_ylabel('Test Accuracy')
ax.set_title('Impact of Network Depth')
ax.set_xticks(depths)

# 3. Batch size impact
ax = axes[1, 0]
for batch in hp_grid['batch_size']:
    batch_results = [r for r in results if r['batch_size'] == batch]
    vals = [r['val_acc'] for r in batch_results]
    tests = [r['test_acc'] for r in batch_results]
    ax.scatter(vals, tests, label=f'Batch={batch}', alpha=0.7)
ax.set_xlabel('Validation Accuracy')
ax.set_ylabel('Test Accuracy')
ax.set_title('Batch Size Impact')
ax.legend()
ax.plot([0.9, 1.0], [0.9, 1.0], 'k--', alpha=0.3)

# 4. Summary heatmap
ax = axes[1, 1]
# Create a simple summary
arch_names = ['Small', 'Medium', 'Large', 'Deep-2', 'Deep-3', 'Deep-3+']
lr_names = ['0.001', '0.01', '0.1']
summary = np.zeros((len(arch_names), len(lr_names)))

for i, arch in enumerate(hp_grid['hidden_sizes']):
    for j, lr in enumerate(hp_grid['learning_rate']):
        accs = [r['test_acc'] for r in results 
                if r['hidden'] == str(arch) and r['lr'] == lr]
        summary[i, j] = np.mean(accs) if accs else 0

im = ax.imshow(summary, cmap='YlOrRd', aspect='auto')
ax.set_xticks(range(len(lr_names)))
ax.set_xticklabels(lr_names)
ax.set_yticks(range(len(arch_names)))
ax.set_yticklabels(arch_names)
ax.set_xlabel('Learning Rate')
ax.set_ylabel('Architecture')
ax.set_title('Average Test Accuracy')

# Add text annotations
for i in range(len(arch_names)):
    for j in range(len(lr_names)):
        text = ax.text(j, i, f'{summary[i, j]:.3f}',
                      ha="center", va="center", color="black", fontsize=8)

plt.colorbar(im, ax=ax)
plt.tight_layout()
plt.savefig('hyperparameter_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# Save results
import json
with open('hyperparameter_results.json', 'w') as f:
    json.dump(results, f, indent=2)
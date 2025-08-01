import numpy as np
import matplotlib.pyplot as plt
from urllib.request import urlretrieve
import os

# Step 1: Download KMNIST dataset
def download_kmnist():
    base_url = "http://codh.rois.ac.jp/kmnist/dataset/kmnist/"
    files = {
        'train-images': 'kmnist-train-imgs.npz',
        'train-labels': 'kmnist-train-labels.npz', 
        'test-images': 'kmnist-test-imgs.npz',
        'test-labels': 'kmnist-test-labels.npz'
    }
    
    for name, filename in files.items():
        if not os.path.exists(filename):
            print(f"Downloading {name}...")
            urlretrieve(base_url + filename, filename)
    print("Download complete!")

# Step 2: Load data
def load_data():
    X_train = np.load('kmnist-train-imgs.npz')['arr_0']
    y_train = np.load('kmnist-train-labels.npz')['arr_0']
    X_test = np.load('kmnist-test-imgs.npz')['arr_0']
    y_test = np.load('kmnist-test-labels.npz')['arr_0']
    
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    return X_train, y_train, X_test, y_test

# Step 3: Preprocess for MLP
def preprocess(X_train, X_test):
    # Normalize
    X_train = X_train.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0
    
    # Flatten
    X_train = X_train.reshape(-1, 784)
    X_test = X_test.reshape(-1, 784)
    
    return X_train, X_test

# Step 4: Visualize
def show_samples(X, y):
    plt.figure(figsize=(10, 2))
    for i in range(10):
        plt.subplot(2, 5, i+1)
        plt.imshow(X[i].reshape(28, 28), cmap='gray')
        plt.title(f'Label: {y[i]}')
        plt.axis('off')
    plt.show()

# Run all steps
if __name__ == "__main__":
    download_kmnist()
    X_train, y_train, X_test, y_test = load_data()
    X_train, X_test = preprocess(X_train, X_test)
    show_samples(X_train, y_train)
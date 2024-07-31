# load_data.py

import numpy as np
from scipy.io import arff
from sklearn.model_selection import train_test_split

def load_data(file_path):
    data, meta = arff.loadarff(file_path)
    data = np.asarray(data.tolist(), dtype=np.float32)
    X = data[:, :-1]  # Features
    y = data[:, -1].astype(int)  # Labels
    X /= 255.0  # Normalize the data
    return train_test_split(X, y, test_size=0.2, random_state=42)

def visualize_sample(X, y):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 10))
    for i in range(25):
        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(X[i].reshape(28, 28), cmap=plt.cm.binary)
        plt.xlabel(y[i])
    plt.show()

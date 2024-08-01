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
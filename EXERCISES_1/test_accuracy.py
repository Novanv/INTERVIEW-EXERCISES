import numpy as np
import arff
from scipy.io import arff
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from model import *
from load_data import *

# Load data
file_path = 'D:/INTERVIEW_EXERCISES/EXERCISES_1/data/mnist_784.arff'  
X_train, X_test, y_train, y_test = load_data(file_path)

num_classes = 10
model = OneVsAllLogisticRegression(num_classes)

# Load weights
try:
    model.load_weights('D:/INTERVIEW_EXERCISES/EXERCISES_1/weights/mnist_logistic_regression_weights')
    y_pred = model.predict(X_test)
    accuracy = np.mean(y_pred == y_test)
    print(f'Accuracy: {accuracy * 100:.2f}%')
except FileNotFoundError as e:
    print(e)

# Load data
file_path = 'D:/INTERVIEW_EXERCISES/EXERCISES_1/data/mnist_784.arff'  
X_train, X_test, y_train, y_test = load_data(file_path)
# Visualize predictions
visualize_predictions(X_test, y_test, y_pred) # You need use jupyter to view in trai_in_colab_and_tes.ipynb file
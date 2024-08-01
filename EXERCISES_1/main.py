import numpy as np
from model import *
from utils import *

def main():
    # Load data
    file_path = 'EXERCISES_1/data/mnist_784.arff'  
    X_train, X_test, y_train, y_test = load_data(file_path)

    num_classes = 10
    model = OneVsAllLogisticRegression(num_classes)

    # Load weights
    try:
        model.load_weights('EXERCISES_1/weights/mnist_logistic_regression_weights')
        y_pred = model.predict(X_test)
        accuracy = np.mean(y_pred == y_test)
        print(f'Accuracy: {accuracy * 100:.2f}%')
    except FileNotFoundError as e:
        print(e)

    # Load data
    file_path = 'EXERCISES_1/data/mnist_784.arff'  
    X_train, X_test, y_train, y_test = load_data(file_path)
    # Visualize predictions
    visualize_predictions(X_test, y_test, y_pred) # You need use jupyter to view in trai_in_colab_and_tes.ipynb file

if __name__=="__main__":
    main()
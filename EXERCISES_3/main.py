from load_data import load_data
from inference import train_model, evaluate_model
import config

def main():
    file_path = 'data/mnist_784.arff'
    X_train, X_test, y_train, y_test = load_data(file_path)
    

    model = train_model(X_train, y_train)
    
    evaluate_model(model, X_test, y_test)

if __name__ == "__main__":
    main()
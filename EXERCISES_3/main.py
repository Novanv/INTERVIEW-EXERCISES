from utils import load_data
from inference import train, evaluate

def main():
    file_path = 'D:/INTERVIEW-EXERCISES/EXERCISES_3/data/mnist_784.arff'
    X_train, X_test, y_train, y_test = load_data(file_path)
    model = train(X_train, y_train)
    evaluate(model, X_test, y_test)

if __name__ == "__main__":
    main()
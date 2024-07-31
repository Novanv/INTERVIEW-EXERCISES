from load_data import load_data, visualize_sample
from inference import train_model, evaluate_model
import config

def main():
    file_path = 'D:/INTERVIEW_EXERCISES/EXERCISES_3/data/mnist_784.arff'
    X_train, X_test, y_train, y_test = load_data(file_path)
    
    visualize_sample(X_train, y_train)

    model = train_model(X_train, y_train)
    
    evaluate_model(model, X_test, y_test)

if __name__ == "__main__":
    main()

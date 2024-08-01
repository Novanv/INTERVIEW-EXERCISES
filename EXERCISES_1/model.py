import numpy as np

class LogisticRegression:
    def __init__(self, learning_rate=0.1, num_iterations=1000):
        """
        Initialize the LogisticRegression model.

        Parameters:
        learning_rate (float): Learning rate for gradient descent.
        num_iterations (int): Number of iterations for gradient descent.
        """
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations

    def sigmoid(self, z):
        """
        Compute the sigmoid of z.

        Parameters:
        z (ndarray): Input data.

        Returns:
        ndarray: Sigmoid of z.
        """
        return 1 / (1 + np.exp(-z))

    def cost_function(self, h, y):
        """
        Compute the cost function for logistic regression.

        Parameters:
        h (ndarray): Predicted probabilities.
        y (ndarray): Actual labels.

        Returns:
        float: Computed cost.
        """
        m = y.shape[0]
        return (-1 / m) * np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))

    def fit(self, X, y):
        """
        Fit the Logistic Regression model.

        Parameters:
        X (ndarray): Feature matrix.
        y (ndarray): Labels.
        """
        m, n = X.shape
        self.theta = np.zeros((n, 1))  # Initialize weights to zero
        y = y.reshape(m, 1)  # Ensure y is a column vector

        for i in range(self.num_iterations):
            z = np.dot(X, self.theta)  # Linear combination of inputs and weights
            h = self.sigmoid(z)  # Apply sigmoid function to get probabilities
            gradient = np.dot(X.T, (h - y)) / m  # Compute gradient
            self.theta -= self.learning_rate * gradient  # Update weights

            if i % 100 == 0:  # Print cost every 100 iterations
                cost = self.cost_function(h, y)
                print(f'Cost after iteration {i}: {cost}')

    def predict(self, X):
        """
        Predict the probabilities for input data X.

        Parameters:
        X (ndarray): Feature matrix.

        Returns:
        ndarray: Predicted probabilities.
        """
        z = np.dot(X, self.theta)  # Linear combination of inputs and weights
        return self.sigmoid(z)  # Apply sigmoid function to get probabilities

    def save_weights(self, file_path):
        """
        Save the model weights to a file.

        Parameters:
        file_path (str): Path to the file to save weights.
        """
        np.save(file_path, self.theta)

    def load_weights(self, file_path):
        """
        Load the model weights from a file.

        Parameters:
        file_path (str): Path to the file to load weights from.
        """
        self.theta = np.load(file_path)

class OneVsAllLogisticRegression:
    def __init__(self, num_classes, learning_rate=0.1, num_iterations=1000):
        """
        Initialize the OneVsAllLogisticRegression model.

        Parameters:
        num_classes (int): Number of classes.
        learning_rate (float): Learning rate for gradient descent.
        num_iterations (int): Number of iterations for gradient descent.
        """
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.models = []

    def fit(self, X, y):
        """
        Fit the One-vs-All Logistic Regression model.

        Parameters:
        X (ndarray): Feature matrix.
        y (ndarray): Labels.
        """
        for i in range(self.num_classes):
            y_i = np.where(y == i, 1, 0)  # Create binary labels for class i
            model = LogisticRegression(self.learning_rate, self.num_iterations)
            print(f'Training model for class {i}')
            model.fit(X, y_i)  # Train model for class i
            self.models.append(model)  # Store the trained model

    def predict(self, X):
        """
        Predict the class labels for input data X.

        Parameters:
        X (ndarray): Feature matrix.

        Returns:
        ndarray: Predicted class labels.
        """
        predictions = np.zeros((X.shape[0], self.num_classes))
        for i in range(self.num_classes):
            predictions[:, i] = self.models[i].predict(X).ravel()  # Get predictions for class i
        return np.argmax(predictions, axis=1)  # Return the class with the highest probability

    def save_weights(self, file_path_prefix):
        """
        Save the weights of all models to files.

        Parameters:
        file_path_prefix (str): Prefix for the file paths to save weights.
        """
        for i, model in enumerate(self.models):
            model.save_weights(f'{file_path_prefix}_class_{i}.npy')

    def load_weights(self, file_path_prefix):
        """
        Load the weights of all models from files.

        Parameters:
        file_path_prefix (str): Prefix for the file paths to load weights from.
        """
        self.models = []
        for i in range(self.num_classes):
            model = LogisticRegression()
            model.load_weights(f'{file_path_prefix}_class_{i}.npy')
            self.models.append(model)
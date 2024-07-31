import numpy as np

class LogisticRegression:
    def __init__(self, learning_rate=0.1, num_iterations=1000):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def cost_function(self, h, y):
        m = y.shape[0]
        return (-1 / m) * np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))

    def fit(self, X, y):
        m, n = X.shape
        self.theta = np.zeros((n, 1))
        y = y.reshape(m, 1)

        for i in range(self.num_iterations):
            z = np.dot(X, self.theta)
            h = self.sigmoid(z)
            gradient = np.dot(X.T, (h - y)) / m
            self.theta -= self.learning_rate * gradient


            if i % 100 == 0:
                cost = self.cost_function(h, y)
                print(f'Cost after iteration {i}: {cost}')

    def predict(self, X):
        z = np.dot(X, self.theta)
        return self.sigmoid(z)

    def save_weights(self, file_path):
        np.save(file_path, self.theta)

    def load_weights(self, file_path):
        self.theta = np.load(file_path)

class OneVsAllLogisticRegression:
    def __init__(self, num_classes, learning_rate=0.1, num_iterations=1000):
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.models = []

    def fit(self, X, y):
        for i in range(self.num_classes):
            y_i = np.where(y == i, 1, 0)
            model = LogisticRegression(self.learning_rate, self.num_iterations)
            print(f'Training model for class {i}')
            model.fit(X, y_i)
            self.models.append(model)

    def predict(self, X):
        predictions = np.zeros((X.shape[0], self.num_classes))
        for i in range(self.num_classes):
            predictions[:, i] = self.models[i].predict(X).ravel()
        return np.argmax(predictions, axis=1)

    def save_weights(self, file_path_prefix):
        for i, model in enumerate(self.models):
            model.save_weights(f'{file_path_prefix}_class_{i}.npy')

    def load_weights(self, file_path_prefix):
        self.models = []
        for i in range(self.num_classes):
            model = LogisticRegression()
            model.load_weights(f'{file_path_prefix}_class_{i}.npy')
            self.models.append(model)
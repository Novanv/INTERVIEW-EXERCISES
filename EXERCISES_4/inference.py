import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from model import NeuralNetwork
from utils import get_triplets, triplet_loss
import config
import joblib
def train(X_train, y_train, save_path='tripel_weights.npy'):
    """
    Trains the neural network using triplet loss and saves the trained model weights.
    
    Parameters:
    - X_train (numpy.ndarray): Training data of shape (number of examples, input_size).
    - y_train (numpy.ndarray): Labels corresponding to the training data.
    - save_path (str): Path to save the trained model weights (default is 'tripel_weights.npy').

    Returns:
    - model (NeuralNetwork): The trained neural network model.
    """
    model = NeuralNetwork(config.INPUT_SIZE, config.HIDDEN_SIZE, config.OUTPUT_SIZE)
    for epoch in range(config.NUM_EPOCHS):
        triplets = get_triplets(X_train, y_train, config.BATCH_SIZE)
        total_loss = 0
        
        for anchor, positive, negative in triplets:
            # Reshape triplets to match the input shape for the model
            anchor, positive, negative = anchor.reshape(1, -1), positive.reshape(1, -1), negative.reshape(1, -1)
            
            # Forward pass
            anchor_out = model.forward(anchor)
            positive_out = model.forward(positive)
            negative_out = model.forward(negative)

            # Compute triplet loss
            loss = triplet_loss(anchor_out, positive_out, negative_out)
            total_loss += loss

            # Compute gradients for the triplet loss
            pos_dist = np.sum(np.square(anchor_out - positive_out), axis=1)
            neg_dist = np.sum(np.square(anchor_out - negative_out), axis=1)
            condition = (pos_dist - neg_dist + 0.2) > 0
            
            # Gradients for anchor, positive, and negative examples
            d_anchor = (2 * (negative_out - positive_out) * condition[:, np.newaxis])
            d_positive = (-2 * (anchor_out - positive_out) * condition[:, np.newaxis])
            d_negative = (2 * (anchor_out - negative_out) * condition[:, np.newaxis])

            # Backpropagation to update the model weights
            model.backward(anchor, d_anchor, config.LEARNING_RATE)
            model.backward(positive, d_positive, config.LEARNING_RATE)
            model.backward(negative, d_negative, config.LEARNING_RATE)

        # Print the average loss for the epoch
        print(f'Epoch {epoch + 1}/{config.NUM_EPOCHS}, Loss: {total_loss / config.BATCH_SIZE}')
    
    # Save the model weights to a file
    weights = {
        'W1': model.W1,
        'b1': model.b1,
        'W2': model.W2,
        'b2': model.b2
    }
    np.save(save_path, weights)
    return model

def evaluate(model, X_test, y_test):
    """
    Evaluates the performance of the trained model using k-NN classification.
    
    Parameters:
    - model (NeuralNetwork): The trained neural network model.
    - X_test (numpy.ndarray): Test data of shape (number of examples, input_size).
    - y_test (numpy.ndarray): True labels corresponding to the test data.
    
    Prints:
    - Accuracy of the k-NN classifier on the test data.
    """
    embeddings = []
    for x in X_test:
        # Generate embedding for each test example
        embedding = model.forward(x.reshape(1, -1))
        embeddings.append(embedding)
    embeddings = np.array(embeddings).reshape(X_test.shape[0], -1)

    # Use k-NN to classify the embeddings and evaluate performance
    knn = KNeighborsClassifier(n_neighbors=10)
    knn.fit(embeddings, y_test)
    accuracy = knn.score(embeddings, y_test)
    print(f'Accuracy: {accuracy * 100:.2f}%')
    joblib.dump(knn, 'knn_weights.pkl')

def load_weights(model, weight_path='D:/INTERVIEW-EXERCISES/EXERCISES_4/weights/tripel_weights.npy'):
    weights = np.load(weight_path, allow_pickle=True).item()
    model.W1 = weights['W1']
    model.b1 = weights['b1']
    model.W2 = weights['W2']
    model.b2 = weights['b2']

def load_knn_model(model_path='knn_weights.pkl'):
    """
    Loads the k-NN model from a file.
    
    Parameters:
    - model_path (str): Path to the file containing the k-NN model (default is 'knn_weights.pkl').
    
    Returns:
    - knn (KNeighborsClassifier): The loaded k-NN model.
    """
    knn = joblib.load(model_path)
    return knn

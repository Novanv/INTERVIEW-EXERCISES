import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from model import NeuralNetwork
from utils import get_triplets, triplet_loss
import config

def train(X_train, y_train):
    model = NeuralNetwork(config.INPUT_SIZE, config.HIDDEN_SIZE, config.OUTPUT_SIZE)
    for epoch in range(config.NUM_EPOCHS):
        triplets = get_triplets(X_train, y_train, config.BATCH_SIZE)
        total_loss = 0
        for anchor, positive, negative in triplets:
            anchor, positive, negative =  anchor.reshape(1, -1), positive.reshape(1, -1), negative.reshape(1, -1)
            anchor_out = model.forward(anchor)
            positive_out = model.forward(positive)
            negative_out = model.forward(negative)

            loss = triplet_loss(anchor_out, positive_out, negative_out)
            total_loss += loss

            # Gradient Calculation
            pos_dist = np.sum(np.square(anchor_out - positive_out), axis=1)
            neg_dist = np.sum(np.square(anchor_out - negative_out), axis=1)
            condition = (pos_dist - neg_dist + 0.2) > 0
            d_anchor = (2 * (negative_out - positive_out) * condition[:, np.newaxis])
            d_positive = (-2 * (anchor_out - positive_out) * condition[:, np.newaxis])
            d_negative = (2 * (anchor_out - negative_out) * condition[:, np.newaxis])

            # Backpropagation
            model.backward(anchor, d_anchor, config.LEARNING_RATE)
            model.backward(positive, d_positive, config.LEARNING_RATE)
            model.backward(negative, d_negative, config.LEARNING_RATE)

        print(f'Epoch {epoch + 1}/{config.NUM_EPOCHS}, Loss: {total_loss / config.BATCH_SIZE}')
    return model

def evaluate(model, X_test, y_test):
    embeddings = []
    for x in X_test:
        embedding = model.forward(x.reshape(1, -1))
        embeddings.append(embedding)
    embeddings = np.array(embeddings).reshape(X_test.shape[0], -1)

    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(embeddings, y_test)
    accuracy = knn.score(embeddings, y_test)
    print(f'Accuracy: {accuracy * 100:.2f}%')
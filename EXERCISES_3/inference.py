import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from model import NeuralNetwork
from triplet_loss import get_triplets, triplet_loss
import config

def train_model(X_train, y_train):
    model = NeuralNetwork(config.INPUT_SIZE, config.HIDDEN_SIZE, config.OUTPUT_SIZE)
    for epoch in range(config.NUM_EPOCHS):
        triplets = get_triplets(X_train, y_train, config.BATCH_SIZE)
        total_loss = 0
        for anchor, positive, negative in triplets:
            anchor_out = model.forward(anchor.reshape(1, -1))
            positive_out = model.forward(positive.reshape(1, -1))
            negative_out = model.forward(negative.reshape(1, -1))

            loss = triplet_loss(anchor_out, positive_out, negative_out)
            total_loss += loss

            dZ2 = (2 / config.BATCH_SIZE) * (anchor_out - positive_out + negative_out)
            model.backward(anchor.reshape(1, -1), dZ2, config.LEARNING_RATE)

        print(f'Epoch {epoch + 1}/{config.NUM_EPOCHS}, Loss: {total_loss / config.BATCH_SIZE}')
    return model

def evaluate_model(model, X_test, y_test):
    embeddings = []
    for x in X_test:
        embedding = model.forward(x.reshape(1, -1))
        embeddings.append(embedding)
    embeddings = np.array(embeddings).reshape(X_test.shape[0], -1)

    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(embeddings, y_test)
    accuracy = knn.score(embeddings, y_test)
    print(f'Accuracy: {accuracy * 100:.2f}%')

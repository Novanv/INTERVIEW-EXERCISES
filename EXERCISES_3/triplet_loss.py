import numpy as np

def triplet_loss(anchor, positive, negative, alpha=0.2):
    pos_dist = np.sum(np.square(anchor - positive), axis=1)
    neg_dist = np.sum(np.square(anchor - negative), axis=1)
    loss = np.maximum(pos_dist - neg_dist + alpha, 0)
    return np.mean(loss)

def get_triplets(X, y, batch_size):
    triplets = []
    classes = np.unique(y)
    for _ in range(batch_size):
        anchor_class = np.random.choice(classes)
        negative_class = np.random.choice(classes[classes != anchor_class])

        anchor_idx = np.random.choice(np.where(y == anchor_class)[0])
        positive_idx = np.random.choice(np.where(y == anchor_class)[0])
        negative_idx = np.random.choice(np.where(y == negative_class)[0])

        anchor = X[anchor_idx]
        positive = X[positive_idx]
        negative = X[negative_idx]

        triplets.append((anchor, positive, negative))

    return triplets

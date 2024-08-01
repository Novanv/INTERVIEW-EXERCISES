import numpy as np

def triplet_loss(anchor, positive, negative, alpha=0.2):
    """
    Compute the triplet loss.

    Arguments:
    anchor -- numpy array of shape (m, n), embeddings for the anchor images
    positive -- numpy array of shape (m, n), embeddings for the positive images
    negative -- numpy array of shape (m, n), embeddings for the negative images
    alpha -- margin

    Returns:
    loss -- scalar, the triplet loss
    """
    # Compute the L2 distance between anchor and positive
    pos_dist = np.sum(np.square(anchor - positive), axis=1)

    # Compute the L2 distance between anchor and negative
    neg_dist = np.sum(np.square(anchor - negative), axis=1)

    # Compute the triplet loss
    loss = np.maximum(0, pos_dist - neg_dist + alpha)

    return np.mean(loss)

# Example usage
anchor = np.array([[0.5, 0.1], [0.3, 0.4]])
positive = np.array([[0.4, 0.2], [0.3, 0.5]])
negative = np.array([[0.6, 0.9], [0.7, 0.8]])

loss = triplet_loss(anchor, positive, negative)
print("Triplet Loss:", loss)
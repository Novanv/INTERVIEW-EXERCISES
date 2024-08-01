import numpy as np
def extended_triplet_loss(anchor, positives, negatives, alpha=0.2):
    """
    Compute the extended triplet loss with multiple positives and negatives.

    Arguments:
    anchor -- numpy array of shape (m, n), embeddings for the anchor images
    positives -- numpy array of shape (m, k, n), embeddings for the positive images
    negatives -- numpy array of shape (m, l, n), embeddings for the negative images
    alpha -- margin

    Returns:
    loss -- scalar, the extended triplet loss
    """
    # Compute the L2 distance between anchor and all positives, then average
    pos_dist = np.mean(np.sum(np.square(anchor[:, np.newaxis, :] - positives), axis=2), axis=1)

    # Compute the L2 distance between anchor and all negatives, then average
    neg_dist = np.mean(np.sum(np.square(anchor[:, np.newaxis, :] - negatives), axis=2), axis=1)

    # Compute the triplet loss
    loss = np.maximum(0, pos_dist - neg_dist + alpha)

    return np.mean(loss)

# Example usage
anchor = np.array([[0.5, 0.1], [0.3, 0.4]])
positives = np.array([[[0.4, 0.2], [0.45, 0.15]], [[0.3, 0.5], [0.35, 0.45]]])
negatives = np.array([[[0.6, 0.9], [0.7, 0.8], [0.65, 0.85], [0.75, 0.75], [0.8, 0.7]], 
                      [[0.6, 0.9], [0.7, 0.8], [0.65, 0.85], [0.75, 0.75], [0.8, 0.7]]])

loss = extended_triplet_loss(anchor, positives, negatives)
print("Extended Triplet Loss:", loss)

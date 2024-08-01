# Triplet Loss Explanation

<image src = "data/one_sample.png">

- **a** is the anchor sample.
- **p** is the positive sample (same class as anchor).
- **n** is the negative sample (different class than anchor).
- **α** is a margin that is enforced between positive and negative pairs.




# Extended Triplet Loss Explanation

`L(a, P, N) = max(0, (1/|P|) Σ (||f(a) - f(p)||²) - (1/|N|) Σ (||f(a) - f(n)||²) + α)`

- **P** is the set of positive samples.
- **N** is the set of negative samples.
- **|P|** and **|N|** are the number of positive and negative samples, respectively.

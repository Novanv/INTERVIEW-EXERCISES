# Triplet Loss Explanation

<image src = "data/one_sample.png">

- **a** is the anchor sample.
- **p** is the positive sample (same class as anchor).
- **n** is the negative sample (different class than anchor).
- **α** is a margin that is enforced between positive and negative pairs.




# Extended Triplet Loss Explanation

$$\mathcal{L}(a, P, N) = \max \left( 0, \frac{1}{|P|} \sum_{p \in P} \|f(a) - f(p)\|_2^2 - \frac{1}{|N|} \sum_{n \in N} \|f(a) - f(n)\|_2^2 + \alpha \right)
$$

- **P** is the set of positive samples.
- **N** is the set of negative samples.
- **|P|** and **|N|** are the number of positive and negative samples, respectively.
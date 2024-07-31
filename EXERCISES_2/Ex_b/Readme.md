# Extended Triplet Loss Explanation

<image src="Extended_Triplet_Loss.png">

- **P** is the set of positive samples.
- **N** is the set of negative samples.
- **|P|** and **|N|** are the number of positive and negative samples, respectively.

## Example

Given the following data:

- **Anchor**: `[[0.5, 0.1], [0.3, 0.4]]`
- **Positives**: `[[[0.4, 0.2], [0.45, 0.15]], [[0.3, 0.5], [0.35, 0.45]]]`
- **Negatives**: `[[[0.6, 0.9], [0.7, 0.8], [0.65, 0.85], [0.75, 0.75], [0.8, 0.7]], [[0.6, 0.9], [0.7, 0.8], [0.65, 0.85], [0.75, 0.75], [0.8, 0.7]]]`
- **Alpha**: `0.2`

### Distance Calculation

#### Distances between Anchor A1 `[0.5, 0.1]` and Positives:

\[ d([0.5, 0.1], [0.4, 0.2]) = 0.141 \]

\[ d([0.5, 0.1], [0.45, 0.15]) = 0.071 \]

**Average Positive Distance**:

0.141+0.0712=0.106 \frac{0.141 + 0.071}{2} = 0.106 

#### Distances between Anchor A1 `[0.5, 0.1]` and Negatives:

\[ d([0.5, 0.1], [0.6, 0.9]) = 0.806 \]

\[ d([0.5, 0.1], [0.7, 0.8]) = 0.728 \]

\[ d([0.5, 0.1], [0.65, 0.85]) = 0.762 \]

\[ d([0.5, 0.1], [0.75, 0.75]) = 0.694 \]

\[ d([0.5, 0.1], [0.8, 0.7]) = 0.670 \]

**Average Negative Distance**:

0.806+0.728+0.762+0.694+0.6705=0.732 \frac{0.806 + 0.728 + 0.762 + 0.694 + 0.670}{5} = 0.732 

### Triplet Loss Calculation

#### Triplet Loss for A1:

Triplet Loss=max \text{Triplet Loss} = \max(0.106 - 0.732 + 0.2, 0) = \max(-0.426, 0) = 0 

#### Triplet Loss for A2:

Similarly calculated, resulting in:

 \text{Triplet Loss} = \max(0 - 0.5 + 0.2, 0) = \max(-0.3, 0) = 0  \text{Triplet Loss} = \max(0 - 0.5 + 0.2, 0) = \max(-0.3, 0) = 0 

### Total Loss:

 \frac{0 + 0}{2} = 0  \frac{0 + 0}{2} = 0 

### Conclusion

**Extended Triplet Loss**: `0`

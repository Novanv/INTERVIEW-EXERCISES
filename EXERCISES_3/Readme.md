# MNIST Triplet Loss Classification

This project implements a deep learning model for optical character classification using the MNIST dataset, employing the Triplet Loss function for training. The model is implemented using only NumPy, with the goal of providing a hands-on understanding of neural networks and triplet loss.

## Project Structure

- **data**: Contains the MNIST dataset in ARFF format.
- **mnist_triplet_loss.py**: Main script that integrates the entire pipeline (data loading, model training, and evaluation).
- **load_data.py**: Handles data loading, preprocessing, and visualization.
- **model.py**: Defines the neural network architecture and training procedures.
- **triplet_loss.py**: Implements the Triplet Loss function and triplet generation.
- **config.py**: Contains configuration parameters for the project.

## Setup

1. **Install Dependencies**: Ensure you have the required Python libraries installed. You can use `pip` or `conda` for package management. This project requires NumPy, SciPy, and scikit-learn.

    ```bash
    pip install numpy scipy scikit-learn matplotlib
    ```

2. **Prepare Data**: Place your MNIST dataset file (`mnist_784.arff`) in the `data/` directory.

3. **Run the Project**: Execute the main script to run the entire pipeline.

    ```bash
    python mnist_triplet_loss.py
    ```

## Explanation of the Approach

1. **Data Loading and Preprocessing**:
   - The MNIST dataset is loaded from an ARFF file and split into training and test sets.
   - Data is normalized to ensure consistent input scaling.

2. **Model Architecture**:
   - A simple neural network with one hidden layer is used for embedding generation.
   - The forward pass applies ReLU activation and linear transformation to produce embeddings.

3. **Triplet Loss Function**:
   - Triplet Loss aims to ensure that an anchor example is closer to positive examples (same class) than to negative examples (different class) by a margin.
   - Triplets are generated during training, and the loss function penalizes embeddings that do not meet the desired similarity constraints.

4. **Training and Evaluation**:
   - The model is trained using the Triplet Loss function, and embeddings are learned.
   - For evaluation, embeddings are used with a k-Nearest Neighbors (k-NN) classifier to assess classification accuracy.

## Advantages of Deep Learning Approach with Triplet Loss

1. **Feature Learning**: Deep learning models, especially with triplet loss, can learn meaningful feature representations, potentially leading to better generalization on unseen data.
2. **Flexibility**: Neural networks can be easily extended with more layers or different architectures to improve performance.
3. **Robustness**: Triplet Loss can be more robust to noisy data and variations compared to traditional classification methods.

## Disadvantages

1. **Complexity**: Training deep learning models can be computationally expensive and require more resources compared to simpler machine learning models.
2. **Triplet Mining**: Generating effective triplets is crucial and can be challenging. Poor triplet selection can negatively impact model performance.
3. **Overfitting Risk**: With more complex models, there's a risk of overfitting, especially if the dataset is not sufficiently large or diverse.

## Comparison with Previous Machine Learning Method

- **Machine Learning Method**: The previous approach used Logistic Regression with a one-vs-all strategy. This method is straightforward and efficient for smaller datasets but may struggle with more complex data patterns.
- **Deep Learning Method**: The deep learning approach with triplet loss can potentially provide better performance on more complex problems by learning richer feature representations. However, it requires careful handling of hyperparameters and can be more resource-intensive.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- The MNIST dataset is provided by Yann LeCun and the AT&T Labs.
- Special thanks to the contributors and libraries that make this project possible.

import numpy as np
from sklearn.model_selection import train_test_split
import arff
from scipy.io import arff
import cv2
def load_data(file_path):
    data, meta = arff.loadarff(file_path)
    
    # Convert to a structured numpy array
    data = np.asarray(data.tolist(), dtype=np.float32)
    
    X = data[:, :-1]  # Features
    y = data[:, -1].astype(int)  # Labels
    
    X /= 255.0  # Normalize the data
    return train_test_split(X, y, test_size=0.2, random_state=42)

def visualize_predictions(X, y_true, y_pred):
    for i in range(25):
        img = (X[i] * 255).reshape(28, 28).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        
        true_label = y_true[i]
        pred_label = y_pred[i]
        
        text = f'True: {true_label}, Pred: {pred_label}'
        
        # Put text on the image
        cv2.putText(img, text, (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
        
        # Display the image
        cv2.imshow(f'Image {i+1}', img)
        
        # Wait for 500 milliseconds or until 'q' is pressed
        if cv2.waitKey(500) & 0xFF == ord('q'):
            break
    
    # Destroy all windows
    cv2.destroyAllWindows()
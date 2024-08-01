from flask import Flask, request, jsonify, render_template
from PIL import Image
from io import BytesIO
from model import NeuralNetwork
import config
from inference import load_weights, load_knn_model
import numpy as np

app = Flask(__name__)

# Initialize the model
model = NeuralNetwork(config.INPUT_SIZE, config.HIDDEN_SIZE, config.OUTPUT_SIZE)

# Load the trained model weights
load_weights(model, 'weights/tripel_weights.npy')

# Load the k-NN model
knn = load_knn_model('weights/knn_weights.pkl')

@app.route('/')
def index():
    return render_template('templates/index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected for uploading"}), 400
    
    try:
        image = Image.open(BytesIO(file.read())).convert("L")
        image = image.resize((28, 28))  # MNIST images are 28x28 pixels
        image_np = np.array(image).reshape(1, -1) / 255.0  # Normalize the image
        
        # Get the embedding from the model
        embedding = model.forward(image_np)
        
        # Use the k-NN model to predict the class
        embedding = embedding.reshape(1, -1)
        prediction = knn.predict(embedding)
        
        return jsonify({"prediction": int(prediction[0])}), 200
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
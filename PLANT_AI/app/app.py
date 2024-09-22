from flask import Flask, request, jsonify, render_template
import os
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import tensorflow as tf
from werkzeug.utils import secure_filename
import google.generativeai as genai




# Initialize the Flask app
app = Flask(__name__)


# Set up Google Gemini API
os.environ['GOOGLE_API_KEY'] = 'API_KEY' 
genai.configure(api_key=os.environ['GOOGLE_API_KEY'])
model = genai.GenerativeModel('gemini-pro')

system_prompt = "You are an expert botanist specializing in plant infections. Your role is to educate users about various plant diseases, their symptoms, causes, and treatments. When asked about a specific plant infection, provide detailed, accurate information in a clear and concise manner. If you're not sure about something, say so rather than providing incorrect information."

def get_gemini_response(prompt):
    messages = [{"role": "user", "content": system_prompt},
                {"role": "user", "content": prompt}
                ]
    response = model.generate_content(prompt)
    return response.text



# Directory to store uploaded images
UPLOAD_FOLDER = r'C:\Users\karan\Python ML\PLANT_AI\app\uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)



valSet = tf.keras.utils.image_dataset_from_directory(
    r'C:\Users\karan\Downloads\Plant Data\New Plant Diseases Dataset(Augmented)\Plant Dataset\valid',
    labels="inferred",
    label_mode="categorical",
    class_names=None,
    color_mode="rgb",
    batch_size=32,
    image_size=(128, 128),
    shuffle=True,
    seed=None,
    validation_split=None,
    subset=None,
    interpolation="bilinear",
    follow_links=False,
    crop_to_aspect_ratio=False,

)

# Load the pre-trained model
tfmodel = tf.keras.models.load_model(r'C:\Users\karan\Python ML\PLANT_AI\trained_model.keras')  
className = valSet.class_names  


# Route for the home page
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle image uploads and prediction
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file:
        # Save the uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Preprocess the image
        img = load_img(filepath, target_size=(128, 128))  # Resize to match model's input size
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        
        # Make prediction
        predictions = tfmodel.predict(img_array)
        result_index = np.argmax(predictions)  # Find the index of the highest probability
        predicted_class = className[result_index]  # Get the corresponding class name
        confidence_score = predictions[0][result_index]  # Probability of the predicted class
          # Return the predicted class and confidence score as JSON
        return jsonify({
            'prediction': predicted_class,
            'confidence': round(float(confidence_score) * 100, 2)  # Convert to percentage and round
        })
    
# Route to handle chatbot requests
@app.route('/chatbot', methods=['POST'])
def chatbot():
    user_message = request.json.get('message')
    if not user_message:
        return jsonify({'error': 'No message provided'}), 400
    
    bot_response = get_gemini_response(user_message)
    return jsonify({'response': bot_response})

if __name__ == '__main__':
    app.run(debug=True)

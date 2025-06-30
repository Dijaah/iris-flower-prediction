from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas

app = Flask(__name__)
# Load model and label encoder
model = pickle.load(open('wine_type_model.pkl', 'rb'))
label_encoder = pickle.load(open('label_encoder.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get form inputs
    features = [float(request.form[feature]) for feature in request.form]
    final_features = np.array([features])

    # Predict wine type
    prediction = model.predict(final_features)[0]
    wine_type = label_encoder.inverse_transform([prediction])[0].capitalize()

    return render_template('result.html', prediction_text=f"The wine is predicted to be: {wine_type} wine.")

if __name__ == "__main__":
    app.run(debug=True)

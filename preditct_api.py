# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error
from flask import Flask, request, jsonify, render_template

# Step 1: Load the dataset
print("Loading dataset...")
data = pd.read_csv('Train_rev1.csv')  # Use Kaggle dataset downloaded locally 
print(f"Dataset loaded with {data.shape[0]} rows and {data.shape[1]} columns.")

# Step 2: Preprocess the data
print("Preprocessing data...")
data['Title'] = data['Title'].fillna('')
data['FullDescription'] = data['FullDescription'].fillna('')
data['LocationRaw'] = data['LocationRaw'].fillna('Unknown')
data['Category'] = data['Category'].fillna('Unknown')

# Target variable
data = data[data['SalaryNormalized'] > 0]  # Remove zero or invalid salaries
print(f"Filtered dataset to {data.shape[0]} rows with valid salaries.")

# Step 3: Feature Engineering
print("Combining text features...")
text_features = data['Title'] + ' ' + data['FullDescription'] + ' ' + data['LocationRaw'] + ' ' + data['Category']
target = data['SalaryNormalized']
print("Text features and target variable prepared.")

# Step 4: Split the data into train and test sets
print("Splitting dataset into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(text_features, target, test_size=0.2, random_state=42)
print(f"Training set: {X_train.shape[0]} samples, Testing set: {X_test.shape[0]} samples.")

# Step 5: Build the model pipeline
print("Building the model pipeline...")
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=10000)),  # Convert text to numerical vectors
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))  # Regressor
])
print("Pipeline created.")

# Step 6: Train the model
print("Training the model...")
pipeline.fit(X_train, y_train)
print("Model training completed.")

# Step 7: Evaluate the model
print("Evaluating the model...")
y_pred = pipeline.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Absolute Error: {mae}")

# Step 8: Build the Flask API
print("Starting Flask API...")
app = Flask(__name__, template_folder='templates')

# Route to serve the frontend
@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')    

@app.route('/predict/salary/<posting_id>', methods=['GET'])
def predict_salary(posting_id):
    # Example Lever job postings data (mock data for simplicity)
    job_postings = {
        "0b976101-6047-47a5-bc9b-d4ff4578652c": "Remote Data Quality Specialist job description in Toronto, Data Quality, Remote",
        "3d0fc59b-70f1-4d6d-9384-2bfb0fa03a0a": "Head of Customer Success for SaaS company in Vancouver, Leadership, Customer Success",
        "0bb9479c-36a5-4a2f-ad47-6e5ca2fb37ec": "Research Intern, AI Research, Machine Learning Intern in Montreal"
    }

    if posting_id not in job_postings:
        return jsonify({"error": "Job posting ID not found."}), 404

    # Preprocess job description
    job_description = job_postings[posting_id]
    print(f"Predicting salary for job posting ID: {posting_id}...")

    # Predict salary
    predicted_salary = pipeline.predict([job_description])[0]
    print(f"Predicted salary: {predicted_salary}")

    return jsonify({
        "job_posting_id": posting_id,
        "predicted_salary": round(predicted_salary, 2)
    })

if __name__ == '__main__':
    print("Flask app is running...")
    app.run(debug=True)

# **Salary Predictor API**


https://github.com/user-attachments/assets/a2a6642e-f751-4a62-9563-97214e1e6bfd


## **Overview**

The Salary Predictor API is a Flask-based application that predicts salaries for job listings based on their job posting ID. Using machine learning, the app is trained on a dataset of job titles, descriptions, and corresponding salaries. The project also includes a simple frontend to interact with the API.

---

## **Features**

- **Salary Prediction**: Predicts the salary of a job posting based on job descriptions and titles.
- **API Integration**: REST API that accepts job posting IDs and returns predicted salaries.
- **Frontend Interface**: User-friendly interface to input job posting IDs and fetch predicted salaries.
- **Model Pipeline**: Uses `TfidfVectorizer` for text processing and a `RandomForestRegressor` for salary prediction.

---

## **Technologies Used**

- **Backend**: Python, Flask
- **Frontend**: HTML, CSS, JavaScript
- **Machine Learning**: Scikit-learn
- **Deployment**: Currently training the model locally but intend to deployt it on Render

---

## **Usage**

### **1. Dataset**
Ensure the training dataset (`Train_rev1.csv`) is present in the project directory. You can update the file path in the script if needed.

### **2. Running Locally**
Run the following command to start the Flask app:
```bash
python app.py
```

Access the app in your browser at: `http://127.0.0.1:5000`

### **3. Frontend**
The frontend can be accessed via the root endpoint (`/`). Enter the job posting ID to fetch the predicted salary.

---

## **API Endpoints**

### **Predict Salary**
**GET** `/predict/salary/<posting_id>`

- **Input**: Job posting ID
- **Output**: Predicted salary for the job

**Example:**
```bash
curl http://127.0.0.1:5000/predict/salary/0b976101-6047-47a5-bc9b-d4ff4578652c
```

**Response:**
```json
{
    "job_posting_id": "0b976101-6047-47a5-bc9b-d4ff4578652c",
    "predicted_salary": 75000
}
```

---

## **Folder Structure**

```
salary-predictor/
├── predict_api.py                # Main Flask application
├── templates/            # HTML templates
│   └── index.html        # Frontend
├── Train_rev1.csv        # Dataset
└── README.md             # Project documentation
```

---

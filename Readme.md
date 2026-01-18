# Project 6 – Wine Cultivar Origin Prediction System

## Project Overview

You are required to develop a **Wine Cultivar Origin Prediction System** using a machine learning algorithm. The system predicts the cultivar (origin/class) of wine based on its chemical properties using the *Wine Dataset* (UCI / sklearn Wine dataset).

The dataset contains 13 numerical chemical features measured from wine samples derived from three different cultivars. For this project, only the following features may be used:

- `alcohol`  
- `malic_acid`  
- `ash`  
- `alcalinity_of_ash`  
- `magnesium`  
- `total_phenols`  
- `flavanoids`  
- `color_intensity`  
- `hue`  
- `od280/od315_of_diluted_wines`  
- `proline`  
- `cultivar` (target variable / class label)  

> 👉 To build the predictive model, select any six (6) input features from the list above (excluding **cultivar**, which is the target variable).

---

## PART A — Model Development

**File:** `model_development.py` or `model_building.ipynb`

You are required to:

- Load the Wine dataset  
- Perform data preprocessing, including:  
  - Handling missing values (if any)  
  - Feature selection  
  - Feature scaling (mandatory due to varying feature ranges)  

- Implement any one of the following machine learning algorithms:  
  - Logistic Regression  
  - Random Forest Classifier  
  - Support Vector Machine (SVM)  
  - K-arest Neighbours (KNN)  
  - Naïve Bayes  
  - Neural Network  
  - LightGBM  

- Train the model using the dataset  
- Evaluate the model using appropriate multiclass classification metrics:  
  - Accuracy  
  - Precision, Recall, and F1-score (macro or weighted)  
  - Classification report  

- Save the trained model to disk using an appropriate method (e.g., Joblib or Pickle)

---

## PART B — Web GUI Application

**Files:** `app.py` and `index.html` (if applicable)

Develop a simple Web-based Graphical User Interface (GUI) that:

- Loads the saved trained multiclass classification model  
- Allows users to input wine chemical properties  
- Passes the input data to the model  
- Displays the predicted wine cultivar/origin (e.g., Cultivar 1, 2, or 3)

---

## Permitted Technologies / Stack

- Flask + HTML/CSS  
- Streamlit  
- FastAPI  
- Django (not recommended)  
- Gradio  

---

## PART C — GitHub Submission

Upload the entire project to GitHub using the structure below:

/WineCultivar_Project_yourName_matricNo/
|
|- app.py
|- requirements.txt
|
|- /model/
| |- model_building.ipynb
| |- wine_cultivar_model.pkl
|
|- /static/
| |- style.css (optional, if applicable)
|
|- /templates/
|- index.html (if applicable)


---

## PART D — Deployment Instructions

Deploy the Web GUI using any one of the following platforms:

- Render.com  
- PythonAnywhere.com  
- Streamlit Cloud  
- Vercel  

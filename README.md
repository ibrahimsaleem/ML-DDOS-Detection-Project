# ML-DDOS-Detection-Project

A machine learning-based approach for detecting Distributed Denial of Service (DDoS) attacks. This repository is developed for the **Data Science for Cybersecurity** course as a group project by **Team Olympians**.

---

## Table of Contents
1. [Overview](#overview)  
2. [Dataset](#dataset)  
3. [Project Timeline & Tasks](#project-timeline--tasks)  
4. [Project Structure](#project-structure)  
5. [Installation](#installation)  
6. [Usage](#usage)  
7. [Future Enhancements](#future-enhancements)  
8. [Team](#team)  
9. [License](#license)  
10. [Acknowledgements](#acknowledgements)  

---

## Overview
This project aims to demonstrate how data science and machine learning techniques can be applied to detect DDoS attacks in network traffic. By performing data preprocessing, feature engineering, model training, and evaluation, we showcase a pipeline that identifies malicious patterns indicative of DDoS behavior.

**Key Objectives:**
- Understand and analyze network traffic data through Exploratory Data Analysis (EDA).
- Clean and preprocess the data to handle missing values and outliers.
- Select and engineer features relevant to DDoS attack detection.
- Address class imbalance using techniques such as SMOTE.
- Train, tune, and evaluate multiple machine learning models.
- Propose future improvements, including advanced deep learning methods.

---

## Dataset
- **Name**: DDoS2020 (Prairie View A&M University)  
- **Source**: [PVAMU-DDoS-2020.csv](http://pvamu1.s3-website-us-east-1.amazonaws.com/data/PVAMU-DDoS-2020.csv)  
- **Description**: The dataset includes network traffic records labeled to indicate normal vs. DDoS traffic, suitable for building classification models.

---

## Project Timeline & Tasks
Below is a summary of our project plan, tasks, responsible team members, and completion dates.

| **Sr No** | **Task**                                     | **Team Member** | **Timeline**  |
|-----------|----------------------------------------------|-----------------|---------------|
| 1         | Exploratory Data Analysis (EDA)             | 1               | 3/30/2025     |
| 2         | Data Cleaning                               | 2               | 3/30/2025     |
| 3         | Feature Selection & Extraction              | 3               | 4/2/2025      |
| 4         | Handling Data Imbalance                     | 4               | 4/5/2025      |
| 5         | Splitting Data & Normalizing Data           | 4               | 4/5/2025      |
| 6         | Training 4 Basic Models                     | All             | 4/8/2025      |
| 7         | Training 4 Advanced Models                  | All             | 4/8/2025      |
| 8         | Hyperparameter Tuning & Cross-Validation    | 3               | 4/10/2025     |
| 9         | Model Evaluation on Test Data               | All             | 4/12/2025     |
| 10        | Conclusion & Future Enhancements            | All             | 4/12/2025     |
| 11        | Presentation                                | 1 & 2           | 4/19/2025     |
| 12        | Report                                      | 3 & 4           | 4/19/2025     |

### Task Descriptions

1. **Exploratory Data Analysis (EDA)**  
   - Load dataset, visualize distributions, identify outliers, correlations, and initial insights.

2. **Data Cleaning**  
   - Handle missing values (drop or impute), remove duplicates, and address outliers using appropriate methods.

3. **Feature Selection & Extraction**  
   - Use techniques like correlation matrices, SelectKBest, or Lasso to identify and retain the most informative features.

4. **Handling Data Imbalance**  
   - Use SMOTE or similar techniques to address imbalance in class labels.

5. **Splitting & Normalizing Data**  
   - Split into training and test sets (80/20 split). Normalize/standardize numerical features for model compatibility.

6. **Training 4 Basic Models**  
   - Logistic Regression, Decision Tree, K-Nearest Neighbors (KNN), and Naïve Bayes.

7. **Training 4 Advanced Models**  
   - Random Forest, Gradient Boosting (XGBoost), Support Vector Machine (SVM), and Neural Networks (MLP).

8. **Hyperparameter Tuning & Cross-Validation**  
   - Perform GridSearchCV or RandomizedSearchCV with 5-fold CV to optimize model parameters and avoid overfitting.

9. **Model Evaluation on Test Data**  
   - Evaluate performance using Accuracy, Precision, Recall, F1-score, ROC-AUC, and Confusion Matrix.  
   - Optionally, use SHAP or LIME for model explainability.

10. **Conclusion & Future Enhancements**  
    - Summarize findings, compare model performances, and propose improvements (e.g., deep learning approaches, anomaly detection with autoencoders, or text-based analysis with BERT/GPT models).

11. **Presentation**  
    - Prepare a comprehensive project presentation showcasing methodologies, results, and key insights.

12. **Report**  
    - Compile a detailed report summarizing the entire project, methodologies, outcomes, and lessons learned.

---

## Project Structure
ML-DDOS-Detection-Project/
├── data/
│   ├── raw/                 # Original dataset
│   └── processed/           # Cleaned and transformed data
├── notebooks/               # Jupyter notebooks for EDA, experiments, analysis
├── src/                     # Source code for data processing and model training
│   ├── preprocessing.py     # Scripts for cleaning and preparing data
│   ├── feature_engineering.py
│   ├── model_training.py    # Scripts to train both basic and advanced models
│   ├── hyperparameter.py    # Hyperparameter tuning and cross-validation
│   └── utils.py             # Utility functions
├── results/                 # Model outputs, evaluation metrics, plots
├── requirements.txt         # Python dependencies
└── README.md                # Project documentation

---

## Installation
1. **Clone the repository**:  
   `git clone https://github.com/yourusername/ML-DDOS-Detection-Project.git`

2. **Navigate to the project directory**:  
   `cd ML-DDOS-Detection-Project`

3. **Create a virtual environment**:  
   `python -m venv venv`

4. **Activate the virtual environment**:  
   - **Windows**: `venv\Scripts\activate`  
   - **macOS/Unix**: `source venv/bin/activate`

5. **Install dependencies**:  
   `pip install -r requirements.txt`

---

## Usage
1. **Data Preprocessing**  
   Run the preprocessing script to clean and prepare the dataset:  
   `python src/preprocessing.py`

2. **Feature Engineering**  
   Generate new features or select top features:  
   `python src/feature_engineering.py`

3. **Model Training**  
   Train the basic and advanced models:  
   `python src/model_training.py`

4. **Hyperparameter Tuning**  
   Run hyperparameter tuning and cross-validation:  
   `python src/hyperparameter.py`

5. **Evaluation & Visualization**  
   Evaluate the trained models on the test set and generate plots. Refer to the `notebooks/` directory or `results/` folder for any additional scripts or detailed analysis.

---

## Future Enhancements
- **Deep Learning Approaches**: Implement LSTM or CNN for time-series based detection.  
- **Anomaly Detection**: Use autoencoders for unsupervised anomaly detection.  
- **Text-Based Analysis**: Integrate BERT/GPT models for analyzing text-based attack logs.

---

## Team

- **Team Name**: Team Olympians  
- **Total Members**: 4  

### Members
1. **Danindu Gammanpilage**  

2. **Mohammad Ibrahim Saleem**  

3. **Simran Khaparde**  

4. **Suvarna Aglave** (Team Leader)  

---

## License
This project is licensed under the [MIT License](LICENSE).

---

## Acknowledgements
- **Course**: Data Science for Cybersecurity  
- **Instructors & Mentors**: For guidance and feedback  
- **Open-Source Community**: For providing essential libraries and resources  
- **Prairie View A&M University**: For the DDoS2020 dataset  

Feel free to open an issue or pull request for any improvements or clarifications!

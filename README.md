#  Ecommerce Purchase Intent Classification

## Problem Statement
To predict whether an online visitor will complete a purchase using machine learning classification models based on their browsing behavior and session characteristics.

---

##  Dataset Description
The **Online Shoppers Purchasing Intention Dataset** contains **12,330 user sessions** with **18 behavioral and technical features** collected from a real e-commerce website.

Each session captures user interaction details such as:
- Number of pages visited and time spent
- Bounce rate and exit rate
- Traffic source and visitor type
- Operating system, browser, and region
- Special days and weekend information

###  Target Variable
- **Revenue**
  - `1` → Purchase completed
  - `0` → No purchase

---

##  Machine Learning Models Used
The following classification models were trained and evaluated:

- Logistic Regression  
- Decision Tree  
- K-Nearest Neighbors (KNN)  
- Naive Bayes  
- Random Forest  
- XGBoost  

---

## Evaluation Metrics
Due to class imbalance, multiple evaluation metrics were used:

- Accuracy  
- AUC (Area Under ROC Curve)  
- Precision  
- Recall  
- F1 Score  
- MCC (Matthews Correlation Coefficient)  

---

## Model Performance Comparison

| Model               | Accuracy | AUC    | Precision | Recall | F1 Score | MCC    |
|--------------------|----------|--------|-----------|--------|----------|--------|
| Logistic Regression | 0.8832 | 0.8653 | 0.7640 | 0.3560 | 0.4857 | 0.4696 |
| Decision Tree       | 0.8532 | 0.7293 | 0.5250 | 0.5497 | 0.5371 | 0.4501 |
| KNN                 | 0.8783 | 0.7990 | 0.6990 | 0.3770 | 0.4898 | 0.4540 |
| Naive Bayes         | 0.7794 | 0.8020 | 0.3802 | 0.6728 | 0.4858 | 0.3826 |
| Random Forest       | 0.9011 | 0.9185 | 0.7331 | 0.5681 | 0.6401 | 0.5902 |
| XGBoost             | 0.8893 | 0.9161 | 0.6698 | 0.5628 | 0.6117 | 0.5505 |

---

## Model Observations

| Model | Observation |
|------|------------|
| Logistic Regression | Performs well due to approximate linear separability of features |
| Decision Tree | Tends to overfit without pruning |
| KNN | Highly sensitive to feature scaling |
| Naive Bayes | Fast and simple but assumes feature independence |
| Random Forest | Handles non-linearity effectively and performs consistently |
| XGBoost | Best overall balance across all evaluation metrics |

---

## Streamlit Web Application
An interactive **Streamlit web application** is developed to:

- Select any trained machine learning model
- Upload a test CSV file for prediction
- View predicted purchase intent
- Display prediction counts
- Show classification report and confusion matrix 
- Download a sample test CSV file for quick testing

---

## Project Structure

```text
ecommerce-purchase-intent-classifier/
│
├── app.py                    # Streamlit application
├── requirements.txt          # Project dependencies
├── model/
│   └── saved_model.pkl       # Trained models, scaler, and metrics
├── sample_test_data.csv      # Sample test dataset
└── README.md

```
---

## ▶️ How to Run the Application Locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

Conclusion

Ensemble learning models such as Random Forest and XGBoost outperform traditional classifiers by effectively capturing complex, non-linear user behavior patterns.
The deployed Streamlit application enables real-time testing and evaluation of purchase intent predictions.

Author

Soumya Jha

import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef

#Load dataset
df = pd.read_csv("data/online_shoppers_intention.csv")

#Encoding columns - categorical
cat_cols = ['Month', 'VisitorType','Weekend']
encoder = LabelEncoder()
for col in cat_cols:
    df[col] = encoder.fit_transform(df[col])

#Target variable
df['Revenue'] = df['Revenue'].astype(int)

X = df.drop('Revenue', axis=1)
Y = df['Revenue']

#Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y)

#Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#Models
models = {
    'Logistic Regression' : LogisticRegression(max_iter=1000),
    'Decision Tree' : DecisionTreeClassifier(random_state=42),
    'KNN' : KNeighborsClassifier(n_neighbors=7),
    'Naive Bayes': GaussianNB(),
    'Random Forest': RandomForestClassifier(n_estimators=200, random_state=42),
    'XGBoost': XGBClassifier(
        eval_mteric='logloss',
        use_label_encoder=False,
        random_state=42
    )

}

metrics = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics[name] = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'AUC': roc_auc_score(y_test, y_prob),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1 Score': f1_score(y_test, y_pred),
        'MCC': matthews_corrcoef(y_test, y_pred)
    }

#Saving
with open('model/saved_model.pkl', 'wb') as f:
    pickle.dump((models, scaler, metrics), f)

print("Training complete. Model metrics: \n")
print(pd.DataFrame(metrics).T)
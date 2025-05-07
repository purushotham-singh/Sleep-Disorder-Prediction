# Sleep Disorder Prediction: Data Science Project

## Project Overview

The goal of this data science project is to analyze various lifestyle and medical factors to predict the occurrence and type of sleep disorders individuals may experience. Sleep disorders, such as **Insomnia** and **Sleep Apnea**, can have serious consequences for one's health and overall well-being. By identifying individuals who are at risk, we can provide timely interventions and treatments to improve their sleep quality and, consequently, their health.

In this project, we focus on a dataset that includes variables like **age**, **BMI**, **physical activity**, **sleep duration**, and **blood pressure**, among others. Our aim is to use these features to build a model that can predict the likelihood of different types of sleep disorders, thereby offering a data-driven approach to improving sleep health.

## Dataset Description

The dataset used for this analysis is the **Sleep Health and Lifestyle Dataset**, which contains information about 400 individuals, with 13 columns representing various attributes related to sleep and daily habits. Below is an overview of the dataset's key features:

### Key Features:

#### 1. Comprehensive Sleep Metrics

This includes details like sleep duration, sleep quality, and other factors that may influence an individual's sleep patterns.

#### 2. Lifestyle Factors

This section provides insights into factors such as physical activity, stress levels, and BMI, which are likely to have an impact on sleep health.

#### 3. Cardiovascular Health

Key cardiovascular measurements such as blood pressure and heart rate are included, as these health indicators are often linked to sleep disorders.

#### 4. Sleep Disorder Analysis

The main objective of the project is to identify the presence or absence of sleep disorders. The **Sleep Disorder** column classifies individuals into one of the following categories:

* **None**: No sleep disorder is detected.
* **Insomnia**: Difficulty falling or staying asleep, leading to poor sleep quality.
* **Sleep Apnea**: Interrupted breathing during sleep, which results in fragmented sleep patterns.

### Data Dictionary

| **Column Name**     | **Description**                                                        |
| ------------------- | ---------------------------------------------------------------------- |
| `Person_ID`         | Unique identifier for each person                                      |
| `Gender`            | Gender of the individual (Male/Female)                                 |
| `Age`               | Age of the individual (in years)                                       |
| `Occupation`        | Occupation of the individual                                           |
| `Sleep_duration`    | Hours of sleep per night                                               |
| `Quality_of_sleep`  | Subjective rating of sleep quality (scale from 1 to 10)                |
| `Physical_activity` | Level of physical activity (Low/Medium/High)                           |
| `Stress_Level`      | Subjective rating of stress level (scale from 1 to 10)                 |
| `BMI_category`      | Body Mass Index (BMI) category (Underweight/Normal/Overweight/Obesity) |
| `Blood_pressure`    | Blood pressure (systolic/diastolic)                                    |
| `Heart_rate`        | Heart rate (beats per minute)                                          |
| `Daily_Steps`       | Number of steps taken per day                                          |
| `Sleep_disorder`    | Type of sleep disorder (None, Insomnia, Sleep Apnea)                   |

## Preprocessing

1. **Load & Clean Data**

   ```python
   import pandas as pd
   df = pd.read_csv('Sleep_health_and_lifestyle_dataset.csv')
   df.drop('Person ID', axis=1, inplace=True, errors='ignore')
   df['Sleep Disorder'] = df['Sleep Disorder'].fillna('None')
   df.dropna(subset=['Age','Blood Pressure','BMI Category'], inplace=True)
   ```
2. **Blood Pressure Split**

   ```python
   bp = df['Blood Pressure'].str.split('/', expand=True).astype(int)
   df['systolic_bp'], df['diastolic_bp'] = bp[0], bp[1]
   df.drop('Blood Pressure', axis=1, inplace=True)
   ```
3. **Clean BMI Categories**

   ```python
   df['BMI Category'] = df['BMI Category'].str.replace('Weight','').str.strip()
   ```

## Feature Preparation

* **Target Encoding** with `LabelEncoder` on `Sleep Disorder`.
* **Numeric Features**: `['Age', 'systolic_bp', 'diastolic_bp']` scaled with `StandardScaler`.
* **Categorical Features**: One-hot encoded with `OneHotEncoder(handle_unknown='ignore')`.

```python
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
le = LabelEncoder()
y = le.fit_transform(df['Sleep Disorder'])
X = df.drop('Sleep Disorder', axis=1)
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), ['Age','systolic_bp','diastolic_bp']),
    ('cat', OneHotEncoder(handle_unknown='ignore'), X.columns.difference(['Age','systolic_bp','diastolic_bp']))
])
```

## Modeling Pipeline

We build unified pipelines for four classifiers:

* **Logistic Regression** (`clf__C` grid: `[0.01, 0.1, 1, 10]`)
* **K-Nearest Neighbors** (`clf__n_neighbors`: `[3,5,7]`)
* **Support Vector Machine** (`clf__C`: `[0.1,1,10]`, `clf__kernel`: `['linear','rbf']`)
* **Decision Tree** (`clf__max_depth`: `[None,5,10]`)

Each pipeline:

```python
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

def make_pipe(clf):
    return Pipeline([
        ('prep', preprocessor),
        ('select', SelectKBest(f_classif, k=10)),
        ('clf', clf)
    ])
models = {
    'logreg': (make_pipe(LogisticRegression(max_iter=1000)), {'clf__C':[0.01,0.1,1,10]}),
    'knn':    (make_pipe(KNeighborsClassifier()), {'clf__n_neighbors':[3,5,7]}),
    'svm':    (make_pipe(SVC()), {'clf__C':[0.1,1,10],'clf__kernel':['linear','rbf']}),
    'dt':     (make_pipe(DecisionTreeClassifier()), {'clf__max_depth':[None,5,10]})
}
```

## Training & Evaluation

1. **Train/Test Split** (80/20 stratified).
2. **GridSearchCV** (5-fold, accuracy scoring).
3. **Metrics**: accuracy score, classification report, confusion matrix plots.
4. **Select Best Model** and save.

```python
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import joblib

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
best_models = {}
for name, (pipe, params) in models.items():
    gs = GridSearchCV(pipe, params, cv=5, n_jobs=-1, scoring='accuracy')
    gs.fit(X_train, y_train)
    y_pred = gs.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(name, gs.best_params_, acc)
    best_models[name] = (gs.best_estimator_, acc)
# Pick and save
best_name, (best_model, best_acc) = max(best_models.items(), key=lambda x: x[1][1])
joblib.dump(best_model, 'sleep_disorder_model.joblib')
```

## Running in Google Colab

1. Upload the dataset and notebook to Colab.
2. Mount Google Drive:

   ```python
   from google.colab import drive
   ```

drive.mount('/content/drive')

```
3. Adjust `DATA_PATH` to your Drive location and run cells sequentially.

## Impact and Objective

This project provides insights into how lifestyle and health metrics affect sleep disorders. The final model can help healthcare professionals identify at-risk individuals and guide interventions to improve sleep quality and overall well-being.

```

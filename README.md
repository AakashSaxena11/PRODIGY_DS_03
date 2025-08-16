# PRODIGY_DS_03

# Bank Marketing Campaign Prediction
  
*Visualization of the optimized decision tree*

## 📌 Business Objective
Predict whether a customer will subscribe to a term deposit (target variable `y`) based on:
- Demographic features (age, job, education)
- Financial features (balance, loans)
- Campaign metadata (contact duration, previous attempts)

**Why it matters**: Banks can achieve **5x higher conversion rates** by targeting high-probability customers.

## 🛠️ Technical Implementation
```python
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(max_depth=4, min_samples_split=20)
model.fit(X_train, y_train)  # 89.2% test accuracy

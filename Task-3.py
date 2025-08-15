#!/usr/bin/env python
# coding: utf-8

# In[12]:


# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt

# Load the dataset with proper formatting
try:
    # Try loading with semicolon separator (common for bank datasets)
    file_path = r"C:\Users\aakas\OneDrive\PRODIGY INFOTECH\Task-3\Bank_Cleaned - bank-full.csv"
    data = pd.read_csv(file_path, sep=';')
    
    # If still single column, try comma separator
    if data.shape[1] == 1:
        data = pd.read_csv(file_path, sep=',')
        
    # If still single column, try tab separator
    if data.shape[1] == 1:
        data = pd.read_csv(file_path, sep='\t')
        
except Exception as e:
    print(f"Error loading file: {e}")
    exit()

# Verify data loaded correctly
print("Dataset shape:", data.shape)
print("\nFirst 5 rows:")
print(data.head())

# Data Preprocessing
# Convert categorical variables to numerical using Label Encoding
label_encoders = {}
categorical_cols = ['job', 'marital', 'education', 'default', 'housing', 'loan', 
                   'contact', 'month', 'poutcome', 'y']

for col in categorical_cols:
    try:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])
        label_encoders[col] = le
    except KeyError:
        print(f"Warning: Column '{col}' not found in dataset. Available columns: {list(data.columns)}")
        continue

# Split data into features (X) and target (y)
X = data.drop('y', axis=1)
y = data['y']

# Split data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the Decision Tree classifier
dt_classifier = DecisionTreeClassifier(
    max_depth=4,  # Limit tree depth for better visualization
    min_samples_split=20,
    random_state=42
)
dt_classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = dt_classifier.predict(X_test)

# Evaluate the model
print("\nModel Evaluation:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Visualize the decision tree
plt.figure(figsize=(20,10))
plot_tree(dt_classifier, 
          feature_names=X.columns, 
          class_names=['No', 'Yes'], 
          filled=True, 
          rounded=True,
          proportion=True)
plt.title("Decision Tree Visualization")
plt.show()

# Display text representation of the tree
tree_rules = export_text(dt_classifier, feature_names=list(X.columns))
print("\nDecision Tree Rules:")
print(tree_rules)

# Feature importance
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': dt_classifier.feature_importances_
}).sort_values('Importance', ascending=False)

print("\nFeature Importance:")
print(feature_importance)

# Plot feature importance
plt.figure(figsize=(10,6))
plt.barh(feature_importance['Feature'], feature_importance['Importance'])
plt.xlabel('Importance')
plt.title('Feature Importance')
plt.show()


# In[ ]:





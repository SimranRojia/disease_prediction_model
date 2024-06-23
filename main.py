import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, roc_curve, roc_auc_score, precision_recall_curve, auc, confusion_matrix
from sklearn.preprocessing import StandardScaler


# Step 1: Load the dataset
file_path = 'heart.csv'  # Replace with the actual path to your heart.csv file
heart_data = pd.read_csv(file_path)

# Step 2: Data preprocessing
# Handle missing data (if any)
# For illustration, assume there are no missing values in this dataset

# Separate features and target variable
X = heart_data.drop('target', axis=1)
y = heart_data['target']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Feature scaling (if necessary)
# For Decision Trees, feature scaling is not mandatory, but we'll demonstrate it for completeness
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 4: Initialize and train the model
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train_scaled, y_train)

# Step 5: Evaluate the model
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Step 6: Predict probabilities and binary predictions
y_probabilities = model.predict_proba(X_test_scaled)
y_pred_prob = y_probabilities[:, 1]  # Probability of class 1 (disease)
y_pred_binary = (y_pred_prob >= 0.5).astype(int)  # Apply threshold of 0.5

# Print an example with detailed information
example_index = 0
example_features = X_test.iloc[example_index]  # Get features of the example
example_actual_label = y_test.iloc[example_index]  # Get actual label of the example

print("\nExample Prediction:")
print(f"Actual Label: {example_actual_label}")
print(f"Predicted Probability of Disease: {y_pred_prob[example_index]:.4f}")
print(f"Predicted Class: {y_pred_binary[example_index]}")



# Step 8: Generate visualizations

# Histogram of predicted probabilities
plt.figure(figsize=(10, 6))
plt.hist(y_pred_prob, bins=20, edgecolor='black')
plt.xlabel('Predicted Probability of Disease')
plt.ylabel('Frequency')
plt.title('Histogram of Predicted Probabilities')
plt.show()

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
roc_auc = roc_auc_score(y_test, y_pred_prob)

plt.figure(figsize=(8, 8))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()

# Precision-Recall Curve
precision, recall, _ = precision_recall_curve(y_test, y_pred_prob)
pr_auc = auc(recall, precision)

plt.figure(figsize=(8, 8))
plt.plot(recall, precision, color='green', lw=2, label=f'Precision-Recall curve (AUC = {pr_auc:.2f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc='best')
plt.show()

# Confusion Matrix Heatmap
cm = confusion_matrix(y_test, y_pred_binary)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, cmap='Blues', fmt='g', cbar=False)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()

# Step 9: Optional - Visualize the Decision Tree (for interpretability)
plt.figure(figsize=(20, 10))
plot_tree(model, filled=True, feature_names=X.columns, class_names=['No Disease', 'Disease'])
plt.title("Decision Tree Visualization")
plt.show()

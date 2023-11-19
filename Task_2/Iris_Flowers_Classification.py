# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# Load the dataset
iris_data = pd.read_csv("IRIS.csv")

# Data exploration and visualization
sns.pairplot(iris_data, hue='species')
plt.suptitle('Pairplot of Iris Dataset')
plt.show()

# Correlation heatmap
correlation_matrix = iris_data.drop(['species'], axis=1).corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

# Data preprocessing
X = iris_data.drop(['species'], axis=1)
y = iris_data['species']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=32)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train and evaluate multiple classifiers
classifiers = {
    'Logistic Regression': LogisticRegression(),
    'K-Nearest Neighbors': KNeighborsClassifier(),
    'Support Vector Machine': SVC(),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier()
}

best_accuracy = 0
best_model_name = ''

for name, classifier in classifiers.items():
    classifier.fit(X_train_scaled, y_train)
    y_pred = classifier.predict(X_test_scaled)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred) * 100
    print(f'\n{name} Accuracy: {accuracy:.4f}')
    print(f'Classification Report for {name}:\n{classification_report(y_test, y_pred)}')

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, cmap='Blues', fmt='g', xticklabels=iris_data['species'].unique(),
                yticklabels=iris_data['species'].unique())
    plt.title(f'Confusion Matrix for {name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

    # Update best model information
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model_name = name

# Print the best model information
print(f'\nBest Model: {best_model_name} with Accuracy: {best_accuracy:.4f}')

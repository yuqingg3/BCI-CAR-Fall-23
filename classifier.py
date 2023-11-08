import pandas as pd
from scipy import stats
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score



# Load your dataset (replace 'your_dataset.csv' with the actual file path)
#df = pd.read_csv('../BCI-CAR-Fall-23/data.csv', delimiter='\t')
df = pd.read_csv('data.csv', delimiter='\t')

# Group the data by category
grouped_data = df.groupby(df.iloc[:, 0])

# Initialize an empty DataFrame to store the cleaned data
cleaned_data_list = []
# Loop through each category and remove outliers using the Z-Score method
for category, group in grouped_data:
    z_scores = stats.zscore(group.iloc[:, 1:])
    z_scores_abs = np.abs(z_scores)
    no_outliers = group[(z_scores_abs < 3).all(axis=1)]  # Adjust the threshold as needed
    
    # Check if 'no_outliers' is not empty before concatenating
    if not no_outliers.empty:
        # Append the cleaned data for this category to the cleaned_data DataFrame
        cleaned_data_list.append(no_outliers)
        
cleaned_data = pd.concat(cleaned_data_list)

# Extract features and labels from the cleaned data
X = cleaned_data.iloc[:, 1:].values
y = np.array(cleaned_data.iloc[:, 0].values, dtype='int64')


# Split the cleaned data into a training set and a test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.preprocessing import StandardScaler

# Standardize features to have mean=0 and variance=1
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)



# Initialize the KNN classifier (you can specify the number of neighbors with the 'n_neighbors' parameter)
knn = KNeighborsClassifier(n_neighbors=5, weights='distance')  # You can adjust the number of neighbors as needed

# Train the knn on the training data

#Regular
knn.fit(X_train, y_train)

#Train on scaled data
#knn.fit(X_train_scaled, y_train)



# Predict using the trained knn

# Regular 
y_pred = knn.predict(X_test)

#Scaled knn
#y_pred = knn.predict(X_test_scaled)



# Evaluate the knn's accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# K FOLD CROSS VALIDATION
k = 10
scores = cross_val_score(knn, X, y, cv=k)

# Print the accuracy scores for each fold
for i, score in enumerate(scores):
    print(f"Fold {i + 1} Accuracy: {score:.2f}")

# Calculate and print the mean accuracy across all folds
mean_accuracy = scores.mean()
print(f"Mean Accuracy: {mean_accuracy:.2f}")


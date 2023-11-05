import pandas as pd
from scipy import stats
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier


# Load your dataset (replace 'your_dataset.csv' with the actual file path)
df = pd.read_csv('data.csv', delimiter='\t')

summary_stats = df.iloc[:,0].describe()
print(summary_stats)

unique_values = df.iloc[:,0].unique()
print(unique_values)

df.iloc[:,0] = df.iloc[:,0].astype(int)

# Group the data by category
grouped_data = df.groupby(df.iloc[:, 0])

# Initialize an empty DataFrame to store the cleaned data
cleaned_data = pd.DataFrame(columns=df.columns)

# Loop through each category and remove outliers using the Z-Score method
for category, group in grouped_data:
    z_scores = stats.zscore(group.iloc[:, 1:])
    z_scores_abs = np.abs(z_scores)
    no_outliers = group[(z_scores_abs < 3).all(axis=1)]  # Adjust the threshold as needed

    # Append the cleaned data for this category to the cleaned_data DataFrame
    cleaned_data = pd.concat([cleaned_data, no_outliers])

# Extract features and labels from the cleaned data
X = cleaned_data.iloc[:, 1:].values
y = cleaned_data.iloc[:, 0].values

print (y)

# Split the cleaned data into a training set and a test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Initialize the KNN classifier (you can specify the number of neighbors with the 'n_neighbors' parameter)
model = KNeighborsClassifier(n_neighbors=5, weights='distance')  # You can adjust the number of neighbors as needed

# Train the model on the training data
model.fit(X_train, y_train)

# Predict using the trained model
y_pred = model.predict(X_test)

# Evaluate the model's accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

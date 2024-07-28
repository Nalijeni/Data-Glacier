import pandas as pd
from sklearn.preprocessing import (StandardScaler)
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle

# Load the csv file
df = pd.read_csv("Crop_recommendation.csv")
print(df.head())

# Check null values
null_values = df.isnull().sum()
print(null_values)
total_null_values = df.isnull().sum().sum()
print(f'Total number of null values in the DataFrame: {total_null_values}')

# Select independent and dependent variable
X = df[["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]]
y = df["label"]

# Split the dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=50)

# Feature scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test= sc.transform(X_test)

# Instantiate the model
classifier = RandomForestClassifier()

# Fit the model
classifier.fit(X_train, y_train)

# Make pickle file of our model
pickle.dump(classifier, open("Crop-Prediction.pkl", "wb"))
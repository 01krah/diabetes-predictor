import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import joblib

# Load cleaned data
df = pd.read_csv("C:/Users/krahm/Downloads/587/587phase4/src/phase4/cleaned_data.csv")

# Pick features and target
X = df[["HBA1C_LEVEL", "BLOOD_GLUCOSE_LEVEL", "AGE"]]
y = df["DIABETES"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train Decision Tree
clf = DecisionTreeClassifier(max_depth=5, random_state=42)
clf.fit(X_train, y_train)

# Save the trained model to a file
joblib.dump(clf, "diabetes_model.pkl")

print("Model trained and saved as 'diabetes_model.pkl'")

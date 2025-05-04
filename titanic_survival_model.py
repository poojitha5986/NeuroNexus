# Step 1: Install required packages
!pip install -q kaggle
import os
import json

# Step 2: Set your Kaggle API credentials here
kaggle_token = {
    "username": "your_kaggle_username",  # Replace with your Kaggle username
    "key": "your_kaggle_api_key"         # Replace with your Kaggle API key
}

# Save the API token
os.makedirs(os.path.expanduser("~/.kaggle"), exist_ok=True)
with open(os.path.expanduser("~/.kaggle/kaggle.json"), "w") as f:
    json.dump(kaggle_token, f)
os.chmod(os.path.expanduser("~/.kaggle/kaggle.json"), 0o600)

# Step 3: Download Titanic dataset
!kaggle competitions download -c titanic
!unzip -o titanic.zip -d titanic_data

# Step 4: Imports
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Step 5: Load training data
train_df = pd.read_csv("titanic_data/train.csv")

# Step 6: Preprocess training data
train_df['Age'].fillna(train_df['Age'].median(), inplace=True)
train_df['Embarked'].fillna(train_df['Embarked'].mode()[0], inplace=True)
train_df.drop(['Cabin', 'Name', 'Ticket'], axis=1, inplace=True)

le_sex = LabelEncoder()
le_embarked = LabelEncoder()
train_df['Sex'] = le_sex.fit_transform(train_df['Sex'])
train_df['Embarked'] = le_embarked.fit_transform(train_df['Embarked'])

X = train_df.drop(['Survived', 'PassengerId'], axis=1)
y = train_df['Survived']

# Step 7: Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Step 8: Load and preprocess test data
test_df = pd.read_csv("titanic_data/test.csv")
test_passenger_ids = test_df['PassengerId']

# Preprocess same as training
test_df['Age'].fillna(train_df['Age'].median(), inplace=True)
test_df['Fare'].fillna(train_df['Fare'].median(), inplace=True)
test_df['Embarked'].fillna(train_df['Embarked'].mode()[0], inplace=True)
test_df.drop(['Cabin', 'Name', 'Ticket'], axis=1, inplace=True)

# Encode using same encoders
test_df['Sex'] = le_sex.transform(test_df['Sex'])
test_df['Embarked'] = le_embarked.transform(test_df['Embarked'])

X_test_final = test_df.drop(['PassengerId'], axis=1)

# Step 9: Predict survival
predictions = model.predict(X_test_final)

# Step 10: Display predictions
for pid, pred in zip(test_passenger_ids, predictions):
    status = "Survived ✅" if pred == 1 else "Did NOT survive ❌"
    print(f"PassengerId: {pid} - Prediction: {pred} → {status}")

# Optional: Save to CSV for Kaggle submission
submission_df = pd.DataFrame({
    'PassengerId': test_passenger_ids,
    'Survived': predictions
})
submission_df.to_csv("submission.csv", index=False)
print("\n✅ Predictions saved to 'submission.csv'")

# Plot: How many passengers survived vs. did not survive (with integer count labels)
plt.figure(figsize=(6,4))
ax = sns.countplot(data=train_df, x='Survived', palette='Set2')

# Add integer count labels on top of each bar
for p in ax.patches:
    height = int(p.get_height())  # Convert to integer
    ax.annotate(f'{height}', 
                (p.get_x() + p.get_width() / 2., height), 
                ha='center', va='bottom', fontsize=12, color='black')

# Set labels and title
plt.title("Survival Count on Titanic")
plt.xlabel("Passenger")
plt.ylabel("Number of Passengers")
plt.xticks([0, 1], ['Did NOT Survive', 'Survived'])
plt.tight_layout()
plt.show()
plt.savefig("survival_count_plot.png")

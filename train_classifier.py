import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

print("Loading dataset from data.pickle...")
with open('./data.pickle', 'rb') as f:
    data_dict = pickle.load(f)

data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

print(f"Dataset shape: {data.shape}")
print(f"Total classes: {len(np.unique(labels))}")

# Stratified split to ensure equal representation for all signs
x_train, x_test, y_train, y_test = train_test_split(
    data,
    labels,
    test_size=0.2,
    shuffle=True,
    stratify=labels,
    random_state=42
)

print("\nTraining optimized Random Forest Classifier...")
model = RandomForestClassifier(
    n_estimators=150,
    max_depth=25,
    min_samples_split=4,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

model.fit(x_train, y_train)

y_predict = model.predict(x_test)
score = accuracy_score(y_predict, y_test)

print(f"\nTest Accuracy: {score * 100:.2f}%")
print("\nClassification Report:\n")
print(classification_report(y_test, y_predict))

with open('model.p', 'wb') as f:
    pickle.dump({'model': model}, f)

print("Updated universal model saved successfully as 'model.p'!")
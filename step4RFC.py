import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
data_dict = pickle.load(open('./data_padded.pickle', 'rb'))
data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

# Visualize data
num_classes = len(np.unique(labels))
num_features = data.shape[1]

print(f"Number of samples: {data.shape[0]}")
print(f"Number of features: {num_features}")
print(f"Number of classes: {num_classes}")

# Plot a histogram of the class distribution
plt.figure(figsize=(8, 6))
plt.hist(labels, bins=num_classes, edgecolor='black')
plt.xticks(range(num_classes), np.unique(labels))
plt.xlabel('Class')
plt.ylabel('Count')
plt.title('Class Distribution')
plt.show()

# Split data into train and test sets
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# Train the model
model = RandomForestClassifier()
model.fit(x_train, y_train)

# Make predictions and calculate accuracy
y_predict = model.predict(x_test)
score = accuracy_score(y_test, y_predict)
print('Random Forest Classifier: {}% of samples were classified correctly!'.format(score * 100))

# Print classification report
print("Classification Report:")
print(classification_report(y_test, y_predict))

# Plot Heat map with seaborn
cm = confusion_matrix(y_test, y_predict)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xticks(np.arange(num_classes) + 0.5, np.unique(labels))
plt.yticks(np.arange(num_classes) + 0.5, np.unique(labels))
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Heat Map')
plt.show()

# Save the model
with open('model.p', 'wb') as f:
    pickle.dump({'model': model}, f)



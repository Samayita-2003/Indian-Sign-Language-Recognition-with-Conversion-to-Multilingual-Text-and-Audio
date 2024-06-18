import pickle

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import matplotlib.pyplot as plt

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

x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# Using KNN classifier
model = KNeighborsClassifier(n_neighbors=3)  # You can specify the number of neighbors here

model.fit(x_train, y_train)

y_predict = model.predict(x_test)

score = accuracy_score(y_predict, y_test)

print('KNN Classifier: {}% of samples were classified correctly !'.format(score * 100))

# Print classification report
print("Classification Report:")
print(classification_report(y_test, y_predict))

# Plot heat map
cm = confusion_matrix(y_test, y_predict)
plt.figure(figsize=(8, 6))
plt.imshow(cm, cmap='Blues')
plt.colorbar()
plt.xticks(range(len(np.unique(labels))), np.unique(labels))
plt.yticks(range(len(np.unique(labels))), np.unique(labels))
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Heat Map')
plt.show()

# Saving the model
with open('KNNmodel.p', 'wb') as f:
    pickle.dump({'model1': model}, f)
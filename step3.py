import pickle

# Load data from 'data.pickle' file
data_dict = pickle.load(open('data_padded.pickle', 'rb'))

# Extract data and labels from the dictionary
data = data_dict['data']
labels = data_dict['labels']

# Print the lengths of data and labels
print(f"Number of samples (data): {len(data)}")
print(f"Number of labels: {len(labels)}")

# Check the length of each data sample
for i, sample in enumerate(data):
    print(f"Length of data sample {i + 1}: {len(sample)}")

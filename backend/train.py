# dataset from https://figshare.com/articles/dataset/data_json/7376747


import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the CSV file
data = pd.read_csv('comments.csv')

# Create the binary labels based on the 'funny' column
data['is_funny'] = data['funny'].apply(lambda x: 1 if x > 0 else 0)

# Split the data into training and testing sets
train_data, test_data, train_labels, test_labels = train_test_split(
    data['text'], data['is_funny'], test_size=0.2, random_state=42)

# Create a TF-IDF vectorizer
vectorizer = TfidfVectorizer()

# Fit and transform the training data
train_features = vectorizer.fit_transform(train_data)

# Transform the testing data
test_features = vectorizer.transform(test_data)

# Create the TensorFlow model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(train_features.shape[1],)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(train_features.toarray(), train_labels, epochs=10, batch_size=32)

# Evaluate the model
loss, accuracy = model.evaluate(test_features.toarray(), test_labels)
print(f'Test Loss: {loss:.4f}')
print(f'Test Accuracy: {accuracy:.4f}')
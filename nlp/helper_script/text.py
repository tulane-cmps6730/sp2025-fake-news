import pandas as pd
import os

# Define the directory where features are saved
data_dir = 'nlp/data/preprocessed'

# Load the preprocessed data
train_data = pd.read_csv(os.path.join(data_dir, 'train.csv'))
test_data = pd.read_csv(os.path.join(data_dir, 'test.csv'))

# Check the shape of the data
print("Train Data Shape:", train_data.shape)
print("Test Data Shape:", test_data.shape)

# Optionally, inspect a sample of the data
print("Sample Train Data:", train_data.head())

# Verify Data Types
print("\nData Types in Train Data:")
print(train_data.dtypes)

# Convert processed_text to string if not already
train_data['processed_text'] = train_data['processed_text'].astype(str)

# Inspect for Missing or NaN Values
print("\nMissing Values in Train Data:")
print(train_data.isnull().sum())

# Check for any empty strings in processed_text
empty_strings = (train_data['processed_text'] == '').sum()
print(f"Empty strings in 'processed_text': {empty_strings}")

# Sample texts for further inspection
sample_texts = train_data['processed_text'].head(5).tolist()
print("\nSample Processed Texts:")
print(sample_texts)

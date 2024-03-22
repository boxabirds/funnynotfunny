# this takes the comments csv and creates a train and test set, 90/10 ratio, making sure the data is both evenly split and balanced by undersampling. 
import pandas as pd
from sklearn.model_selection import train_test_split
import argparse
from pathlib import Path

# Set up command line argument parsing
parser = argparse.ArgumentParser(description='Preprocess and split the dataset')
parser.add_argument('--dataset', type=str, default='datasets/comments.csv', help='Path to the input dataset CSV file')
parser.add_argument('--create-splits', action='store_true', help='Should train/test splits be created')
parser.add_argument('--test-size', type=float, default=0.1, help='If create-splits then how big should the test size be (1.0 = all, default 0.1)')
args = parser.parse_args()

# Read the CSV file
dataset_path = Path(args.dataset)
df = pd.read_csv(dataset_path)

# Transform the data
df['funny_binary'] = df['funny'].apply(lambda x: 1 if x > 0 else 0)
transformed_df = df[['funny_binary', 'text']]

# Balance the dataset
funny_0_count = transformed_df[transformed_df['funny_binary'] == 0].shape[0]
funny_1_count = transformed_df[transformed_df['funny_binary'] == 1].shape[0]
min_count = min(funny_0_count, funny_1_count)

funny_0_subset = transformed_df[transformed_df['funny_binary'] == 0].sample(n=min_count, random_state=42)
funny_1_subset = transformed_df[transformed_df['funny_binary'] == 1].sample(n=min_count, random_state=42)
balanced_df = pd.concat([funny_0_subset, funny_1_subset])

if args.create_splits:
    # Split the balanced dataset into train and test sets
    train_df, test_df = train_test_split(balanced_df, test_size=args.test_size, random_state=42, stratify=balanced_df['funny_binary'])
else:
    # Use the entire balanced dataset as the train set
    train_df = balanced_df
    test_df = None

# Print the count and distribution of funny=0 and funny=1 for each dataset
print("Train set:")
print(train_df['funny_binary'].value_counts())
print("Distribution:")
print(train_df['funny_binary'].value_counts(normalize=True))
print("---")

if test_df is not None:
    print("Test set:")
    print(test_df['funny_binary'].value_counts())
    print("Distribution:")
    print(test_df['funny_binary'].value_counts(normalize=True))
    print("---")

# Save the resulting files
output_dir = dataset_path.parent
train_file = output_dir / f"{dataset_path.stem}-train.csv"
train_df.to_csv(train_file, index=False)
print(f"Train set saved as: {train_file}")

if test_df is not None:
    test_file = output_dir / f"{dataset_path.stem}-test.csv"
    test_df.to_csv(test_file, index=False)
    print(f"Test set saved as: {test_file}")

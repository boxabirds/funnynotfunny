import csv
import argparse

def clean_csv(input_file, output_file):
    removed_count = 0

    with open(input_file, mode='r', encoding='utf-8') as infile, open(output_file, mode='w', encoding='utf-8', newline='') as outfile:
        reader = csv.DictReader(infile)
        writer = csv.DictWriter(outfile, fieldnames=reader.fieldnames)
        
        writer.writeheader()
        
        for row in reader:
            if not row["text"].strip():
                print(f"Problematic row: {row}")
                removed_count += 1
            else:
                writer.writerow(row)
    
    print(f"Total problematic rows removed: {removed_count}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean a CSV file by removing rows with empty 'text' fields.")
    parser.add_argument('--csv', type=str, default='datasets/comments-train.csv', help='Input CSV file path')
    parser.add_argument('--output', type=str, default='datasets/comments-cleaned-train.csv', help='Output CSV file path')
    
    args = parser.parse_args()
    
    clean_csv(args.csv, args.output)

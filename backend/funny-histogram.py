import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import ticker

# Read the CSV file into a pandas DataFrame
df = pd.read_csv('datasets/comments.csv')

# Get the total number of rows
total_rows = len(df)

# Get the number of rows with at least one funny vote
rows_with_funny_votes = df[df['funny'] > 0].shape[0]

print(f"Total rows: {total_rows:,}")  
print(f"Rows with at least one funny vote: {rows_with_funny_votes:,}")

# Get the range of non-zero funny vote values
funny_counts = df['funny'].value_counts()
funny_range = range(1, funny_counts.index.max()+1)

# Tally the frequency of each number of funny votes, excluding 0
funny_counts = funny_counts.reindex(funny_range, fill_value=0)

print("\nFunny Vote Counts:")
pd.set_option('display.max_rows', None)
print(funny_counts)

# Create a histogram
fig, ax = plt.subplots(figsize=(10,6))
ax.bar(funny_range, funny_counts, width=0.8, color='#86bf91', zorder=2)

# Despine
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)

# Switch off ticks
ax.tick_params(axis="both", which="both", bottom="off", top="off", labelbottom="on", left="off", right="off", labelleft="on")

# Draw horizontal axis lines
vals = ax.get_yticks()
for tick in vals:
    ax.axhline(y=tick, linestyle='dashed', alpha=0.4, color='#eeeeee', zorder=1)

# Set x-axis label
ax.set_xlabel("Number of Funny Votes (excl. 0)", labelpad=20, weight='bold', size=12)

# Set y-axis label 
ax.set_ylabel("Frequency", labelpad=20, weight='bold', size=12)

# Format y-axis ticks
ax.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,g}'))

plt.xticks(funny_range)
plt.show()

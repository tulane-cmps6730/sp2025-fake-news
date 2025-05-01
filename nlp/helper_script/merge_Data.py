import pandas as pd
from pathlib import Path

data_dir = Path("nlp/data/News _dataset")  # folder name contains a space and underscore

# Load CSV files
fake = pd.read_csv(data_dir / "Fake.csv")
true = pd.read_csv(data_dir / "True.csv")

# Concatenate and label
df = pd.concat([fake, true], ignore_index=True)

# Add simple claim column; here we use the headline/title
df["claim"] = df["title"]

# Ensure there is a `text` column (rename if the body column is something else)
# If already 'text', no change.

# Save merged file one level up (nlp/data/merged.csv)
out_path = data_dir.parent / "merged.csv"
df.to_csv(out_path, index=False)
print("saved", out_path)
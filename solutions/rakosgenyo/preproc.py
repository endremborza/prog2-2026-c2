import pandas as pd
import numpy as np

df = pd.read_csv("input.csv")

# 1. Automatically grab ONLY the boolean columns (these are your genres)
genre_cols = df.select_dtypes(include=['bool']).columns.tolist()

# 2. Create the 'genre' column: a list of column names where the value is True
df["genre"] = df[genre_cols].apply(lambda row: row.index[row].tolist(), axis=1)

# 3. Build your cleaned dataframe exactly as you specified
cleaned_np = df[["imdb_id", "title", "x", "y", "genre", "year"]].to_numpy()


np.save("clean.npy", cleaned_np)

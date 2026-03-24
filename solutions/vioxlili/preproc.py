import pandas as pd
import numpy as np

GENRES = [
    "drama", "history", "musical", "comedy", "music", "romance",
    "family", "adventure", "fantasy", "mystery", "thriller", "sci_fi",
    "biography", "crime", "horror", "documentary", "western", "action",
    "war", "animation", "sport", "film_noir"
]

df = pd.read_csv("input.csv")

keep_cols = ["x", "y", "year", "title", "imdb_id"] + GENRES
df = df[keep_cols]

# Compact dtypes
df["x"] = df["x"].astype(np.float32, copy=False)
df["y"] = df["y"].astype(np.float32, copy=False)
df["year"] = df["year"].astype(np.int16, copy=False)
df["title"] = df["title"].astype(str)
df["imdb_id"] = df["imdb_id"].astype(str)

# szamma kodolja a stringet
title_codes, title_uniques = pd.factorize(df["title"], sort=False)
imdb_codes, imdb_uniques = pd.factorize(df["imdb_id"], sort=False)

df["title_idx"] = title_codes.astype(np.int32, copy=False)
df["imdb_idx"] = imdb_codes.astype(np.int32, copy=False)

# Sort once globally by year, then genre subsets preserve year order
df = df.sort_values("year", kind="mergesort").reset_index(drop=True)

save_dict = {
    "titles": np.asarray(title_uniques, dtype=f"U{max(1, title_uniques.astype(str).str.len().max())}"),
    "imdb_ids": np.asarray(imdb_uniques, dtype=f"U{max(1, imdb_uniques.astype(str).str.len().max())}"),
}

for g in GENRES:
    mask = df[g].to_numpy(dtype=bool, copy=False)
    sub = df.loc[mask, ["x", "y", "year", "title_idx", "imdb_idx"]]

    save_dict[f"{g}__x"] = np.ascontiguousarray(sub["x"].to_numpy(dtype=np.float32, copy=False))
    save_dict[f"{g}__y"] = np.ascontiguousarray(sub["y"].to_numpy(dtype=np.float32, copy=False))
    save_dict[f"{g}__year"] = np.ascontiguousarray(sub["year"].to_numpy(dtype=np.int16, copy=False))
    save_dict[f"{g}__title_idx"] = np.ascontiguousarray(sub["title_idx"].to_numpy(dtype=np.int32, copy=False))
    save_dict[f"{g}__imdb_idx"] = np.ascontiguousarray(sub["imdb_idx"].to_numpy(dtype=np.int32, copy=False))

# Use uncompressed savez: faster write/load than compressed for this use case
np.savez("genres_data.npz", **save_dict)
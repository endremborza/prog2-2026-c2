import pandas as pd


df = pd.read_csv("input.csv")
query_df = pd.read_csv("query.csv")

out = []
for _, row in query_df.iterrows():
    filtered = df.loc[
        df[row["genre"]]
        & (df["year"] >= row["min_year"])
        & (df["year"] <= row["max_year"]),
        ["x", "y"],
    ]
    dists = ((filtered - row[["x", "y"]]) ** 2).sum(axis=1) ** 0.5
    closest_idx = dists.idxmin()
    out.append(df.loc[closest_idx, ["year", "title", "imdb_id"]].to_dict())

pd.DataFrame(out).to_csv("out.csv", index=False)

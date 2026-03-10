import csv
import numpy as np
from pykdtree.kdtree import KDTree
from collections import defaultdict
########################
#Parameter
p=100
########################




# 1. Load the preprocessed array
clean = np.load("clean.npy", allow_pickle=True)

imdb_ids = clean[:, 0]
titles = clean[:, 1]
coords = np.ascontiguousarray(clean[:, 2:4], dtype=np.float32) 
genres_list = clean[:, 4]
years = clean[:, 5].astype(int)

# 2. Read queries first to find exactly which genres we need
queries = []
required_genres = set()

with open("query.csv", mode="r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        queries.append(row)
        required_genres.add(row["genre"])

# 3. Group indices ONLY for the genres we actually need
genre_to_indices = defaultdict(list)
for i, g_list in enumerate(genres_list):
    if isinstance(g_list, list):
        for g in g_list:
            # OPTIMIZATION: Skip building lists for unqueried genres
            if g in required_genres: 
                genre_to_indices[g].append(i)

# 4. Build the KDTree for the required genres
genre_trees = {}
genre_data_map = {} 

for g, indices in genre_to_indices.items():
    idx_array = np.array(indices, dtype=int)
    genre_data_map[g] = idx_array
    genre_trees[g] = KDTree(coords[idx_array])

# 5. Process the loaded queries
out = []
for row in queries:
    q_genre = row["genre"]
    q_min = int(row["min_year"])
    q_max = int(row["max_year"])
    qx, qy = float(row["x"]), float(row["y"])
    
    # Edge case: If the requested genre doesn't exist in our data
    if q_genre not in genre_trees:
        continue
        
    tree = genre_trees[q_genre]
    orig_indices = genre_data_map[q_genre]
    q_pt = np.array([[qx, qy]], dtype=np.float32)
    
    INITIAL_K = min(p, len(orig_indices))
    dists, idxs = tree.query(q_pt, k=INITIAL_K)
    idxs = np.atleast_1d(idxs).flatten()
    
    found = False
    
    # Pass 1: Nearest 100
    for idx in idxs:
        real_idx = orig_indices[idx]
        if q_min <= years[real_idx] <= q_max:
            out.append({
                "year": years[real_idx],
                "title": titles[real_idx],
                "imdb_id": imdb_ids[real_idx]
            })
            found = True
            break
            
    # Pass 2: Fallback to querying all items in this genre tree
    if not found and len(orig_indices) > INITIAL_K:
        dists, idxs = tree.query(q_pt, k=len(orig_indices))
        idxs = np.atleast_1d(idxs).flatten()
        for idx in idxs:
            real_idx = orig_indices[idx]
            if q_min <= years[real_idx] <= q_max:
                out.append({
                    "year": years[real_idx],
                    "title": titles[real_idx],
                    "imdb_id": imdb_ids[real_idx]
                })
                break

# 6. Write the results
with open("out.csv", mode="w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["year", "title", "imdb_id"])
    writer.writeheader()
    writer.writerows(out)
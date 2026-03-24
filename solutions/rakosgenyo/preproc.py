import flask
import csv
import numpy as np
import pandas as pd
from pykdtree.kdtree import KDTree

app = flask.Flask(__name__)

# =========================
# 1. SETUP: ONE TREE PER GENRE
# =========================
df = pd.read_csv("input.csv")
genre_cols = df.select_dtypes(include=["bool"]).columns.tolist()

imdb_ids = df["imdb_id"].values
titles = df["title"].values
coords = np.ascontiguousarray(df[["x", "y"]].values, dtype=np.float32)
years = df["year"].values.astype(np.int32)

genre_data = {}

for g in genre_cols:
    g_mask = df[g].values
    g_idx = np.where(g_mask)[0]
    
    if len(g_idx) == 0:
        continue
        
    # We build just ONE tree per genre, containing all years
    genre_data[g] = {
        'tree': KDTree(coords[g_idx]),
        'idx': g_idx,
        'years': years[g_idx],
        'coords': coords[g_idx]
    }

print("Vectorized Genre Trees Prepared.")

# =========================
# 2. FAST QUERY ENDPOINT
# =========================
@app.route("/ping")
def ping():
    # --- ZERO-PANDAS READ ---
    q_genres, q_mins, q_maxs, q_xs, q_ys = [], [], [], [], []
    
    with open("query.csv", "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        
        idx_g = header.index("genre")
        idx_min = header.index("min_year")
        idx_max = header.index("max_year")
        idx_x = header.index("x")
        idx_y = header.index("y")
        
        for row in reader:
            q_genres.append(row[idx_g])
            q_mins.append(int(row[idx_min]))
            q_maxs.append(int(row[idx_max]))
            q_xs.append(float(row[idx_x]))
            q_ys.append(float(row[idx_y]))

    q_genres = np.array(q_genres, dtype=object)
    q_mins = np.array(q_mins, dtype=np.int32)
    q_maxs = np.array(q_maxs, dtype=np.int32)
    q_xs = np.array(q_xs, dtype=np.float32)
    q_ys = np.array(q_ys, dtype=np.float32)
    
    n_queries = len(q_genres)
    
    out_years = np.full(n_queries, "", dtype=object)
    out_titles = np.full(n_queries, "", dtype=object)
    out_ids = np.full(n_queries, "", dtype=object)

    # --- THE LOOP KILLER: VECTORIZED MATH ---
    # np.unique finds the ~20 genres. We only loop 20 times total!
    for g in np.unique(q_genres):
        if g not in genre_data:
            continue
            
        # 1. Grab EVERY query for this genre at once
        g_mask = (q_genres == g)
        q_idx = np.where(g_mask)[0] 
        
        pts = np.column_stack((q_xs[g_mask], q_ys[g_mask]))
        mins = q_mins[g_mask]
        maxs = q_maxs[g_mask]
        
        g_info = genre_data[g]
        tree = g_info['tree']
        g_yrs = g_info['years']
        g_ids = g_info['idx']
        g_coords = g_info['coords']
        
        # 2. ONE SINGLE TOLL BOOTH: Ask C for the 128 nearest movies 
        # for ALL of these queries simultaneously
        k = min(128, len(g_ids))
        dist, nn = tree.query(pts, k=k)
        
        if k == 1:
            nn = nn[:, np.newaxis]
            
        yrs = g_yrs[nn]
        
        # 3. Vectorized Year Filter (Runs in C/NumPy, zero Python loops)
        valid = (yrs >= mins[:, None]) & (yrs <= maxs[:, None])
        has_match = valid.any(axis=1)
        
        # Slot the successes into the output arrays
        if has_match.any():
            match_col = valid[has_match].argmax(axis=1)
            win_q_idx = q_idx[has_match]
            win_nn_idx = nn[has_match, match_col]
            win_global = g_ids[win_nn_idx]
            
            out_years[win_q_idx] = imdb_ids[win_global] # Wait, this should be years
            out_years[win_q_idx] = years[win_global]
            out_titles[win_q_idx] = titles[win_global]
            out_ids[win_q_idx] = imdb_ids[win_global]

        # 4. INSTANT BRUTE FORCE for the < 1% that missed the 128-net
        failed = ~has_match
        if failed.any():
            fail_q_idx = q_idx[failed]
            fail_pts = pts[failed]
            fail_mins = mins[failed]
            fail_maxs = maxs[failed]
            
            for i in range(len(fail_q_idx)):
                ymask = (g_yrs >= fail_mins[i]) & (g_yrs <= fail_maxs[i])
                if ymask.any():
                    valid_c = g_coords[ymask]
                    valid_global = g_ids[ymask]
                    
                    dsq = (valid_c[:,0] - fail_pts[i,0])**2 + (valid_c[:,1] - fail_pts[i,1])**2
                    best = valid_global[dsq.argmin()]
                    
                    target = fail_q_idx[i]
                    out_years[target] = years[best]
                    out_titles[target] = titles[best]
                    out_ids[target] = imdb_ids[best]

    # --- ZERO-PANDAS WRITE ---
    with open("out.csv", "w", encoding="utf-8", newline="") as f:
        f.write("year,title,imdb_id\n")
        for i in range(n_queries):
            if out_ids[i] == "":
                f.write(",,\n")
            else:
                t = out_titles[i]
                if ',' in t or '"' in t:
                    t = f'"{t.replace(chr(34), chr(34)+chr(34))}"'
                f.write(f"{out_years[i]},{t},{out_ids[i]}\n")

    return "OK\n"

@app.route("/")
def ok():
    return "OK\n"

@app.route("/shutdown")
def shutdown():
    import os
    os._exit(0)

if __name__ == "__main__":
    app.run(port=5678)
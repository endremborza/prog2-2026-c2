import csv

with open("query.csv", "r", encoding="utf-8") as f:
    queries = list(csv.DictReader(f))

out_data = [None] * len(queries)

if len(queries) < 1000:
    with open("input.csv", "r", encoding="utf-8") as f: # ignoring genre_buckets.pkl
        movies = list(csv.DictReader(f))
    
    parsed_movies = []
    for m in movies:
        parsed_movies.append({
            'year': int(m['year']),
            'x': float(m['x']),
            'y': float(m['y']),
            'title': m['title'],
            'imdb_id': m['imdb_id'],
            'raw': m
        })

    for i, q in enumerate(queries): # index and query
        g = q['genre']
        q_x, q_y = float(q['x']), float(q['y'])
        min_y, max_y = int(q['min_year']), int(q['max_year'])
        
        best_dist = float('inf')
        best_meta = ["", "", ""]
        
        for m in parsed_movies:
            if m['raw'][g].lower() in ('true'): # test1 - genre
                y = m['year'] 
                if min_y <= y <= max_y: # test2 - year range
                    dist = (m['x'] - q_x)**2 + (m['y'] - q_y)**2
                    if dist < best_dist:
                        best_dist = dist
                        best_meta = [str(y), m['title'], m['imdb_id']]
        
        out_data[i] = best_meta

else:
    
    import pickle
    import numpy as np
    
    with open("genre_buckets.pkl", "rb") as f:
        buckets = pickle.load(f)

    groups = {} # placing queries n different buckets based on their genre
    for i, q in enumerate(queries):
        g = q['genre']
        if g not in groups:
            groups[g] = {'idxs': [], 'xy': [], 'min_y': [], 'max_y': []}
        groups[g]['idxs'].append(i)
        groups[g]['xy'].append([float(q['x']), float(q['y'])])
        groups[g]['min_y'].append(int(q['min_year']))
        groups[g]['max_y'].append(int(q['max_year']))

    for g, group in groups.items(): # ulls the movie coordinates and years from the buckets (groups) and converts the query information into numPy arrays
        b = buckets.get(g) 
        idxs = group['idxs'] 

        if not b or len(b['years']) == 0:
            for idx in idxs: out_data[idx] = ["", "", ""]
            continue

        q_xy = np.array(group['xy'], dtype=np.float32)
        q_min = np.array(group['min_y'], dtype=np.int32)
        q_max = np.array(group['max_y'], dtype=np.int32)

        m_xy = b['coords']
        m_years = b['years'] # sorvektor, shape: (M, )
        m_meta = b['meta']

        Q = len(idxs)
        CHUNK = 2000

        for i in range(0, Q, CHUNK): # creates the queries x movies matrix
            c_xy = q_xy[i:i+CHUNK]
            c_min = q_min[i:i+CHUNK, None] # oszlopvektor, shape: (Q, 1)
            c_max = q_max[i:i+CHUNK, None] # oszlopvektor, shape: (Q, 1)
            c_idxs = idxs[i:i+CHUNK] # genre-k szerint keveredés -- szükség lesz tudni az eredeti sorrendet

            dx = c_xy[:, 0:1] - m_xy[:, 0]
            dy = c_xy[:, 1:2] - m_xy[:, 1]
            dist_sq = dx**2 + dy**2 # calculates d(1,1), d(1,2), d(1,3), d(1,4)...

            # boolean mask matrix
            valid = (m_years >= c_min) & (m_years <= c_max) # true-false matrix (broadcasting a numpy array-ekkel)
            dist_sq = np.where(valid, dist_sq, np.inf) # np.where -- (condition, value if true, value if false)

            best_idx = np.argmin(dist_sq, axis=1)
            min_dists = np.take_along_axis(dist_sq, best_idx[:, None], axis=1).squeeze(axis=1) # szerintem ez a feltétel felesleges a generálási szabály miatt, de még nem mertem kivenni

            for j, b_idx in enumerate(best_idx):
                out_idx = c_idxs[j]
                if min_dists[j] == np.inf:
                    out_data[out_idx] = ["", "", ""] # vszeg ez is felesleges
                else:
                    out_data[out_idx] = m_meta[b_idx]

# 3. Write output
with open("out.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["year", "title", "imdb_id"])
    writer.writerows(out_data)

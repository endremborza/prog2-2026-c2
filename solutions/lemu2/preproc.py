import csv
import pickle
import numpy as np

with open("input.csv", "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    rows = list(reader)

non_genre_cols = {'year', 'x', 'y', 'title', 'imdb_id'}
genres = [col for col in reader.fieldnames if col not in non_genre_cols]

buckets = {}
for g in genres:
    g_rows = [r for r in rows if r[g].lower() in ('true')] # nézzem meg, hogyan van jelölve
    
    buckets[g] = {
        'years': np.array([int(r['year']) for r in g_rows], dtype=np.int32),
        'coords': np.array([[float(r['x']), float(r['y'])] for r in g_rows], dtype=np.float32),
        'meta': np.array([[r['year'], r['title'], r['imdb_id']] for r in g_rows])
    }

with open("genre_buckets.pkl", "wb") as f: # wb = write binary
    pickle.dump(buckets, f)

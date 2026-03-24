import csv

GENRES = [
    "drama", "history", "musical", "comedy", "music", "romance",
    "family", "adventure", "fantasy", "mystery", "thriller", "sci_fi",
    "biography", "crime", "horror", "documentary", "western", "action",
    "war", "animation", "sport", "film_noir"
]

with open("query.csv", "r", encoding="utf-8") as f:
    raw_queries = list(csv.DictReader(f))

n = len(raw_queries)

if n <= 100:
    parsed_queries = [
        (
            row["genre"].strip().lower(),
            int(row["min_year"]),
            int(row["max_year"]),
            float(row["x"]),
            float(row["y"]),
        )
        for row in raw_queries
    ]

    needed_genres = {q[0] for q in parsed_queries}
    movies_by_genre = {g: [] for g in needed_genres}

    with open("input.csv", "r", encoding="utf-8") as f:
        for m in csv.DictReader(f):
            year = int(m["year"])
            x = float(m["x"])
            y = float(m["y"])
            title = m["title"]
            imdb_id = m["imdb_id"]

            movie = (year, x, y, title, imdb_id)

            for g in needed_genres:
                if m[g] == "True":
                    movies_by_genre[g].append(movie)

    for movies in movies_by_genre.values():
        movies.sort(key=lambda movie: movie[0])

    out_rows = [["", "", ""] for _ in range(n)]

    for i, (genre, min_year, max_year, qx, qy) in enumerate(parsed_queries):
        candidates = movies_by_genre.get(genre)
        if not candidates:
            continue

        best_dist = float("inf")
        best_row = None

        for year, x, y, title, imdb_id in candidates:
            if year < min_year:
                continue
            if year > max_year:
                break

            dx = x - qx
            dy = y - qy
            dist = dx * dx + dy * dy

            if dist < best_dist:
                best_dist = dist
                best_row = [str(year), title, imdb_id]

        if best_row is not None:
            out_rows[i] = best_row

    small_mode = True

else:
    import numpy as np

    loaded = np.load("genres_data.npz", allow_pickle=False)

    titles = loaded["titles"]
    imdb_ids = loaded["imdb_ids"]

    loaded_files = set(loaded.files)

    data = {}
    for g in GENRES:
        year_key = f"{g}__year"
        if year_key in loaded_files:
            data[g] = (
                loaded[year_key],
                loaded[f"{g}__x"],
                loaded[f"{g}__y"],
                loaded[f"{g}__title_idx"],
                loaded[f"{g}__imdb_idx"],
            )

    parsed_queries = [
        (
            row["genre"].strip().lower(),
            int(row["min_year"]),
            int(row["max_year"]),
            float(row["x"]),
            float(row["y"]),
        )
        for row in raw_queries
    ]

    out_year = np.empty(n, dtype=np.int16)
    out_title = np.empty(n, dtype=np.int32)
    out_imdb = np.empty(n, dtype=np.int32)
    valid = np.zeros(n, dtype=bool)

    for i, (genre, min_year, max_year, qx, qy) in enumerate(parsed_queries):
        entry = data.get(genre)
        if entry is None:
            continue

        years, xs, ys, title_idx, imdb_idx = entry

        lo = np.searchsorted(years, min_year, side="left")
        hi = np.searchsorted(years, max_year, side="right")

        if lo >= hi:
            continue

        x_slice = xs[lo:hi]
        y_slice = ys[lo:hi]
        dx = x_slice - qx
        dy = y_slice - qy
        best = lo + int(np.argmin(dx * dx + dy * dy))

        valid[i] = True
        out_year[i] = years[best]
        out_title[i] = title_idx[best]
        out_imdb[i] = imdb_idx[best]

    small_mode = False

with open("out.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["year", "title", "imdb_id"])

    if small_mode:
        writer.writerows(out_rows)
    else:
        for i in range(n):
            if valid[i]:
                writer.writerow([
                    int(out_year[i]),
                    str(titles[out_title[i]]),
                    str(imdb_ids[out_imdb[i]]),
                ])
            else:
                writer.writerow(["", "", ""])

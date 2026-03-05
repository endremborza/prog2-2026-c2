import argparse
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd

DF_URL = "http://tmp-borza-public-cyx.s3.amazonaws.com/imdb-comp.csv.gz"

RUNS_DIR = Path("runs/run-logs")
SOLUTIONS_DIR = Path("solutions")
TEST_DATA_PATH = Path("full-df.csv.gz")

GENRE_COLS = [
    "drama",
    "history",
    "musical",
    "comedy",
    "music",
    "romance",
    "family",
    "adventure",
    "fantasy",
    "mystery",
    "thriller",
    "sci_fi",
    "biography",
    "crime",
    "horror",
    "documentary",
    "western",
    "action",
    "war",
    "animation",
    "sport",
    "film_noir",
]

output_cols = ["year", "title", "imdb_id"]


def load_test_df() -> pd.DataFrame:
    if TEST_DATA_PATH.exists():
        return pd.read_csv(TEST_DATA_PATH)
    df = pd.read_csv(DF_URL)
    df.to_csv(TEST_DATA_PATH, index=False)
    return df


class SolutionRunner:
    def __init__(
        self,
        solution: str,
        in_n: int = 1_000,
        q_n: int = 10,
        comparison: str = "",
        seed: int = 742,
    ):
        self.solution = solution
        self.in_n = in_n
        self.q_n = q_n
        self.comparison = comparison
        self.seed = seed
        self.s_path = SOLUTIONS_DIR / solution
        self.rng = np.random.RandomState(seed)
        self.in_p, self.q_p, self.out_p = map(
            self.s_path.joinpath, ["input.csv", "query.csv", "out.csv"]
        )
        self.test_df = load_test_df()
        self.input_df = None
        self.query_df = None

    def call(self, comm: str) -> float:
        start = time.time()
        subprocess.call(["make", comm], cwd=self.s_path.as_posix())
        return time.time() - start

    def dump_input(self) -> None:
        self.input_df = self.test_df.sample(self.in_n, random_state=self.rng)
        self.input_df.to_csv(self.in_p, index=False)

    def dump_query(self) -> None:
        noise_scale = self.input_df[["x", "y"]].std().mean() * 2
        genre_subsets = {g: self.input_df[self.input_df[g]] for g in GENRE_COLS}
        valid_genres = [g for g, subset in genre_subsets.items() if not subset.empty]

        queries = []
        while len(queries) < self.q_n:
            genre = self.rng.choice(valid_genres)
            genre_subset = genre_subsets[genre]
            anchor = genre_subset.sample(1, random_state=self.rng).iloc[0]
            min_year = int(anchor["year"]) - int(self.rng.randint(1, 30))
            max_year = int(anchor["year"]) + int(self.rng.randint(1, 30))

            year_mask = (genre_subset["year"] >= min_year) & (
                genre_subset["year"] <= max_year
            )
            subset = genre_subset[year_mask]
            if subset.empty:
                continue
            point = subset[["x", "y"]].sample(1, random_state=self.rng).iloc[0]
            queries.append(
                {
                    "genre": genre,
                    "min_year": min_year,
                    "max_year": max_year,
                    "x": float(point["x"]) + self.rng.normal(0, noise_scale),
                    "y": float(point["y"]) + self.rng.normal(0, noise_scale),
                }
            )

        self.query_df = pd.DataFrame(queries)
        self.query_df.to_csv(self.q_p, index=False)

    def validate_output(self, out_df: pd.DataFrame) -> None:
        for i, (_, out_row) in enumerate(out_df.iterrows()):
            query_row = self.query_df.iloc[i]
            movie_vs = self.input_df[self.input_df["imdb_id"] == out_row["imdb_id"]]
            assert not movie_vs.empty, (
                f"query {i}: movie {out_row['imdb_id']} not found in input"
            )
            movie = movie_vs.iloc[0]
            genre = query_row["genre"]
            assert movie[genre], (
                f"query {i}: movie {out_row['title']} does not have genre {genre}"
            )
            assert query_row["min_year"] <= movie["year"] <= query_row["max_year"], (
                f"query {i}: movie year {movie['year']} not in range [{query_row['min_year']}, {query_row['max_year']}]"
            )

    def run(self) -> list[str] | None:
        assert self.s_path.exists(), f"solution not found: {self.s_path}"

        logs = [f"inputs: {self.in_n}", f"queries: {self.q_n}"]
        for comm, prep in [
            ("setup", lambda: None),
            ("preproc", self.dump_input),
            ("compute", self.dump_query),
        ]:
            prep()
            logs.append(f"{comm}: {self.call(comm):.6f}")
        self.in_p.unlink()
        self.q_p.unlink()
        self.call("cleanup")

        try:
            out_df = pd.read_csv(self.out_p)
        except Exception:
            print(f"ERROR: could not read {self.out_p}")
            return None

        assert out_df.columns.tolist() == output_cols, (
            f"columns {out_df.columns.tolist()} != {output_cols}"
        )
        assert out_df.shape[0] == self.q_n, (
            f"output length {out_df.shape[0]} != {self.q_n}"
        )

        self.validate_output(out_df)

        if self.comparison:
            runner = SolutionRunner(
                self.comparison,
                self.in_n,
                self.q_n,
                seed=self.seed,
            )
            runner.run()
            comp_df = pd.read_csv(SOLUTIONS_DIR / self.comparison / self.out_p.name)
            misses = (comp_df != out_df).any(axis=1)
            if misses.any():
                print("Mismatches found:")
                print(f"{self.solution}:\n{out_df[misses]}")
                print(f"{self.comparison}:\n{comp_df[misses]}")

        logstr = "\n".join(logs)
        RUNS_DIR.mkdir(parents=True, exist_ok=True)
        (RUNS_DIR / f"{time.time():.6f}-{self.solution}").write_text(logstr)
        print(f"\nsuccess! solution: {self.solution}")
        print(logstr)
        return logs


def main(
    solution: str,
    in_n: int = 1_000,
    q_n: int = 10,
    comparison: str = "",
    seed: int = 742,
) -> list[str] | None:
    runner = SolutionRunner(solution, in_n, q_n, comparison, seed)
    return runner.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a single challenge-2 solution")
    parser.add_argument("solution")
    parser.add_argument("--compare", default="", metavar="SOLUTION")
    parser.add_argument("--in-n", type=int, default=1_000)
    parser.add_argument("--q-n", type=int, default=10)
    parser.add_argument("--seed", type=int, default=742)
    args = parser.parse_args()
    main(
        args.solution,
        in_n=args.in_n,
        q_n=args.q_n,
        comparison=args.compare,
        seed=args.seed,
    )

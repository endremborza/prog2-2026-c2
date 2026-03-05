import time

from single_run import SOLUTIONS_DIR, main

SIZES = [
    (1_000, 10),
    (5_000, 50),
    (10_000, 100),
    (10_000, 1_000),
    (10_000, 100_000),
]

BENCHMARK_SOLUTION = "baboon"


def getseed() -> int:
    return int(time.time() * 123) // 17 % 0x1000000


class Runner:
    def __init__(self) -> None:
        self.solutions = [s.name for s in SOLUTIONS_DIR.iterdir() if s.is_dir()]
        self.valid_solutions: list[str] = []

    def validate(self) -> None:
        seed = getseed()
        in_n, q_n = SIZES[0]
        main(BENCHMARK_SOLUTION, in_n, q_n, seed=seed)
        for s in self.solutions:
            try:
                main(s, in_n, q_n, seed=seed)
                self.valid_solutions.append(s)
            except Exception as e:
                print(f"failed {s}: {e}")

    def run(self) -> None:
        for in_size, q_size in SIZES:
            seed = getseed()
            for s in self.valid_solutions:
                try:
                    main(s, in_size, q_size, seed=seed)
                except Exception as e:
                    print(f"{s} failed: {e}")


if __name__ == "__main__":
    runner = Runner()
    runner.validate()
    runner.run()

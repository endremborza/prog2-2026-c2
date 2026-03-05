import datetime as dt
import re
import time
from pathlib import Path

import pandas as pd

limit = 60 * 60 * 3


if __name__ == "__main__":
    recs = []
    for f in Path("runs/run-logs").iterdir():
        try:
            tss, sol = re.findall(r"([\d|\.]+)-(.*)", f.name)[0]
        except IndexError:
            continue
        if float(tss) > (time.time() - limit):
            rec = {"solution": sol}
            for l in f.read_text().strip().split("\t"):
                k, v = l.split(":")
                if "ms" in v:
                    v = v.replace("ms", "")
                rec[k.strip()] = float(v.strip())
            recs.append(rec)

    lines = [f"# {dt.date.today().isoformat()}"]
    gcols = ["inputs", "queries"]
    for (ni, nq), gdf in pd.DataFrame(recs).groupby(gcols, sort=True):
        lines.append(f"## Inputs: {int(ni)}, Queries {int(nq)}")
        lines.append(
            gdf.drop(gcols, axis=1)
            .sort_values("compute")
            .round(2)
            .drop_duplicates("solution")
            .to_markdown(index=False)
        )

    Path("runs/README.md").write_text("\n\n".join(lines))

import numpy as np
import pandas as pd
from pathlib import Path


def main():
    src = Path("locations_master.csv")
    dst = Path("locations_master_anonymized.csv")

    if not src.exists():
        raise FileNotFoundError(f"{src} not found in current directory.")

    df = pd.read_csv(src)

    if "TRT ID" not in df.columns:
        raise ValueError("Column 'TRT ID' not found in locations_master.csv.")

    n = len(df)
    if n == 0:
        raise ValueError("locations_master.csv is empty.")

    # Create unique 6-digit IDs that look random
    max_unique = 900_000  # from 100000 to 999999 inclusive
    if n > max_unique:
        raise ValueError(
            f"Too many rows ({n}) to assign unique 6-digit IDs."
        )

    base = 100_000
    # Generate a pool of candidate IDs, shuffle, then take first n
    pool = np.arange(base, base + max_unique, dtype=int)
    rng = np.random.default_rng()
    rng.shuffle(pool)
    new_ids = pool[:n]

    df["TRT ID"] = new_ids
    df.to_csv(dst, index=False)

    print(f"Wrote anonymized file: {dst}")
    print("Sample of new TRT ID values:")
    print(df['TRT ID'].head())


if __name__ == "__main__":
    main()

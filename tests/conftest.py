import os

os.environ.setdefault("PYDANTIC_AI_NO_BANNER", "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402


def make_df(n: int = 300, n_classes: int = 2, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    a = rng.normal(0, 1, n)
    b = a + rng.normal(0, 0.1, n)  # |r(a, b)| ~ 0.99
    c = rng.normal(0, 1, n)
    d = rng.normal(0, 1, n)
    color = rng.choice(["red", "blue"], n)
    z = 1.5 * a - c + (color == "red") * 0.8 + rng.normal(0, 0.5, n)
    if n_classes == 2:
        y = np.where(z > 0, "yes", "no")
    else:
        y = np.select([z < -0.7, z < 0.7], ["low", "mid"], "high")
    return pd.DataFrame({"a": a, "b": b, "c": c, "d": d, "color": color, "y": y})


@pytest.fixture
def df():
    return make_df()


@pytest.fixture
def csv_path(tmp_path, df):
    path = tmp_path / "data.csv"
    df.to_csv(path, index=False)
    return path

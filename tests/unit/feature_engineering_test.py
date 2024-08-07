import pandas as pd

from pet_adoption.feature_engineering import numerical_cols_as_float


def test_numerical_cols_as_float():
    df = pd.DataFrame({"int_col": [1, 2, 3], "float_col": [1.0, 2.0, 3.0], "str_col": ["a", "b", "c"]})
    df = numerical_cols_as_float(df)
    assert df["int_col"].dtype == float
    assert df["float_col"].dtype == float
    assert df["str_col"].dtype == object

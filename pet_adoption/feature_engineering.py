import pandas as pd


def numerical_cols_as_float(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cast numerical columns to float type.

    :param df: initial dataframe
    :return: dataframe with numerical columns casted to float type
    """
    numerical_columns = df.select_dtypes(include=["int64", "float64"]).columns
    df[numerical_columns] = df[numerical_columns].astype(float)
    return df

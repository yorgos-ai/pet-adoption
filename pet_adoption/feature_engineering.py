import pandas as pd


def object_cols_as_category(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cast object columns as category type.

    :param df: initial dataframe
    :return: dataframe with object columns casted to category type
    """
    object_columns = df.select_dtypes(include=["object"]).columns
    df[object_columns] = df[object_columns].astype("category")
    return df


def numerical_cols_as_float(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cast numerical columns to float type.

    :param df: initial dataframe
    :return: dataframe with numerical columns casted to float type
    """
    numerical_columns = df.select_dtypes(include=["int64", "float64"]).columns
    df[numerical_columns] = df[numerical_columns].astype(float)
    return df

import pandas as pd
from catboost import CatBoostClassifier
from typing import List, Tuple, Optional
from sklearn.model_selection import train_test_split


def object_type_to_category(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cast object columns as category type.

    :param df: initial dataframe
    :return: dataframe with object columns casted to category type
    """
    object_columns = df.select_dtypes(include=["object"]).columns
    df[object_columns] = df[object_columns].astype("category")
    return df


def stratified_split(
    df: pd.DataFrame,
    target_col: str,
    random_state: int,
    frac_train: float = 0.6,
    frac_val: float = 0.2,
    frac_test: float = 0.2,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Splits a DataFrame into train, validation, and test sets, stratified on the target column.

    :param df: initial dataframe
    :param target_col: the name fo the target column
    :param frac_train: the fraction of data in the train set, defaults to 0.6
    :param frac_val: the fraction of data in the validation set, defaults to 0.2
    :param frac_test: the fraction of data in the test set, defaults to 0.2
    :param random_state: the random seed, defaults to None
    :return: _description_
    """
    if frac_train + frac_val + frac_test != 1.0:
        raise ValueError("Fractions must sum to 1")

    # Perform initial split into train/temp and validate/test
    df_train, df_temp = train_test_split(
        df, stratify=df[target_col], test_size=(1.0 - frac_train), random_state=random_state
    )

    # Further split temp into val/test
    relative_frac_test = frac_test / (frac_val + frac_test)
    df_val, df_test = train_test_split(
        df_temp, stratify=df_temp[target_col], test_size=relative_frac_test, random_state=random_state
    )

    return df_train, df_val, df_test


def train_model(
    df_train: pd.DataFrame,
    df_val: Optional[pd.DataFrame],
    target: str,
    num_features: List[str],
    cat_features: List[str],
    random_state: int,
) -> CatBoostClassifier:
    """
    Train a CatBoost classifier on the training data. Optionally, you can pass a validation dataframe.

    :param df_train: the training dataset
    :param df_val: the validation dataset
    :param target: the name of the target column
    :param num_features: the list of numerical features
    :param cat_features: the list of categorical features
    :param random_state: the random seed
    :return: a fitted CatBoost classifier
    """
    model = CatBoostClassifier(
        iterations=1000,
        learning_rate=0.1,
        depth=6,
        cat_features=cat_features,
        verbose=200,
        random_state=random_state,
    )

    X_train = df_train[num_features + cat_features]
    y_train = df_train[target]

    if df_val is not None:
        X_val = df_val[num_features + cat_features]
        y_val = df_val[target]

        model.fit(X_train, y_train, eval_set=(X_val, y_val))

    else:
        model.fit(X_train, y_train)

    return model

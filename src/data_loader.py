from io import StringIO

import boto3
import pandas as pd


def read_csv_from_s3(bucket_name: str, file_key: str) -> pd.DataFrame:
    """
    Read a CSV file from an S3 bucket into a Pandas DataFrame.

    :param bucket_name: the name of the S3 bucket
    :param file_key: the full path to the csv file
    :return: a dataframe containing the CSV data
    """
    # Create an S3 resource
    s3 = boto3.resource("s3")

    # Get the object from the bucket
    obj = s3.Object(bucket_name, file_key)
    file_content = obj.get()["Body"].read().decode("utf-8")

    # Read the CSV data into a DataFrame
    df = pd.read_csv(StringIO(file_content))
    return df

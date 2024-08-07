import pandas as pd
from pandas.testing import assert_frame_equal

from pet_adoption.flows.tasks import preprocess_data, read_data, read_data_from_s3, store_data_in_s3


def test_read_data_from_s3(create_bucket):
    s3 = create_bucket
    # mock uploading a CSV file to the S3 bucket
    file_key = "test-file.csv"
    test_data = pd.DataFrame({"col1": [1, 2, 3], "col2": [4, 5, 6]})
    csv_string = test_data.to_csv(index=False)
    s3.Object("test_bucket", file_key).put(Body=csv_string)
    # Call the read_data_from_s3 task
    df = read_data_from_s3("test_bucket", file_key)
    # Check if the returned DataFrame matches the test data
    assert_frame_equal(df, test_data)


def test_store_data_in_s3(create_bucket):
    create_bucket
    # mock data
    file_key = "test-file.csv"
    test_data = pd.DataFrame({"col1": [1, 2, 3], "col2": [4, 5, 6]})
    # Call the store_data_in_s3 task
    store_data_in_s3(test_data, "test_bucket", file_key)
    # Read the data from S3
    mocked_data = read_data_from_s3("test_bucket", file_key)
    # Check if the returned DataFrame matches the test data
    assert_frame_equal(test_data, mocked_data)


def test_read_data():
    df = read_data()
    assert df.shape[0] == 2007
    assert df.shape[1] == 13


def test_preprocess_data():
    data = pd.DataFrame({"col1": [1, 2, 3], "col2": [4, 5, 6], "col3": ["dog", "cat", "dog"]})

    proccessed_data = preprocess_data(data)
    assert proccessed_data["col1"].dtype == float
    assert proccessed_data["col2"].dtype == float

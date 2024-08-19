import boto3
import pytest
from moto import mock_s3


@pytest.fixture
def s3_client():
    with mock_s3():
        conn = boto3.resource("s3", region_name="us-east-1")
        yield conn


@pytest.fixture
def bucket_name():
    return "my-test-bucket"


@pytest.fixture
def s3_bucket(s3_client, bucket_name):
    s3_client.create_bucket(Bucket=bucket_name)
    yield

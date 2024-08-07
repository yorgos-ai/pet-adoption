import os

import boto3
import pytest
from moto import mock_aws


@pytest.fixture(scope="function")
def aws_credentials():
    """Mocked AWS Credentials for moto."""
    os.environ["AWS_ACCESS_KEY_ID"] = "testing"
    os.environ["AWS_SECRET_ACCESS_KEY"] = "testing"
    os.environ["AWS_SECURITY_TOKEN"] = "testing"
    os.environ["AWS_SESSION_TOKEN"] = "testing"
    os.environ["AWS_DEFAULT_REGION"] = "eu-west-1"


@pytest.fixture(scope="function")
def s3_client(aws_credentials):
    """Pytest fixture that creates the recipes bucket in
    the fake moto AWS account

    Yields a fake boto3 s3 client
    """
    with mock_aws():
        s3_client = boto3.resource("s3", region_name="eu-west-1")
        yield s3_client


@pytest.fixture
def create_bucket(s3_client):
    location = {"LocationConstraint": "eu-west-1"}
    s3_client.create_bucket(Bucket="test_bucket", CreateBucketConfiguration=location)
    yield s3_client


# def upload_csv(s3_bucket):
#     """Pytest fixture that mocks uploading a CSV file to the S3 bucket"""
#     def _upload_csv(file_name, content):
#         s3 = boto3.client("s3")
#         s3.put_object(Bucket="test_bucket", Key=file_name, Body=content)

# return _upload_csv

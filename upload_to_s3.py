import boto3

from logger_config import logger


def upload_file_to_s3(bucket_name, object_name, directory_name):
    # Create an S3 client
    s3 = boto3.client("s3", region_name="eu-central-1")
    # Define the S3 bucket and the file name you want to upload
    s3.put_object(Bucket=bucket_name, Key=f'{directory_name}/')
    s3_object_key = f'{directory_name}/{object_name}'
    # Upload the file to S3
    s3.upload_file(object_name, bucket_name, s3_object_key)
    logger.info(f"File '{object_name}' uploaded to '{bucket_name}/{s3_object_key}'")

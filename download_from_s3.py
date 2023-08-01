import boto3
from logger_config import logger
def download_file_from_s3(bucket_name, object_name, local_file_path):
    s3_client = boto3.client("s3",
                      region_name="eu-central-1")
    try:
        logger.info(f"Downloading file from bucket '{bucket_name}' with object '{object_name}' and storing to path '{local_file_path}'")
        s3_client.download_file(bucket_name, object_name, local_file_path)
        return True
    except Exception as e:
        logger.info(f"Error downloading file '{object_name}' from bucket '{bucket_name}': {e}")
        return False
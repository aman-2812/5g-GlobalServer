import boto3
from logger_config import logger
from datetime import datetime

# Get today's date in the format YYYY-MM-DD
current_datetime = datetime.now()

# Format the date and time as a string (YYYY-MM-DD)
formatted_datetime = current_datetime.strftime('%d-%m-%Y')

def upload_file_to_s3(name):
    # Set your AWS credentials (if not set via environment variables or IAM role)
    region_name = 'eu-central-1'
    directory_name = f'{formatted_datetime}'
    # Create an S3 client
    s3 = boto3.client('s3', region_name=region_name)
    # Define the S3 bucket and the file name you want to upload
    bucket_name = 'fra-5g-nw-global'
    s3.put_object(Bucket=bucket_name, Key=f'{directory_name}/')
    file_name = name  # Change this to your local file path
    s3_object_key = f'{directory_name}/{name}'
    # Upload the file to S3
    s3.upload_file(file_name, bucket_name, s3_object_key)
    logger.info(f"File '{file_name}' uploaded to '{bucket_name}/{s3_object_key}'")

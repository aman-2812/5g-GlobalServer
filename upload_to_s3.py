import boto3
def upload_file_to_s3(name):
    # Set your AWS credentials (if not set via environment variables or IAM role)
    region_name = 'eu-central-1'
    # Create an S3 client
    s3 = boto3.client('s3', region_name=region_name)
    # Define the S3 bucket and the file name you want to upload
    bucket_name = 'fra-5g-nw-global'
    file_name = name  # Change this to your local file path
    # Specify the key (object name) under which the file will be saved in the S3 bucket
    object_key = name  # Change this to your desired S3 object key
    # Upload the file to S3
    s3.upload_file(file_name, bucket_name, object_key)
    print(f"File '{file_name}' uploaded to '{bucket_name}/{object_key}'")

import boto3, os
from dotenv import load_dotenv
load_dotenv("server/.env")

s3 = boto3.client(
    "s3",
    endpoint_url=os.environ["R2_ENDPOINT_URL"],
    aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
    aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
    region_name="auto",
)

# Upload object nhỏ để test
s3.put_object(Bucket=os.environ["R2_BUCKET_NAME"], Key="test/ping.txt", Body=b"ok")
print("Upload OK")

# List để verify
r = s3.list_objects_v2(Bucket=os.environ["R2_BUCKET_NAME"], Prefix="test/")
print(r["Contents"])

# Xóa
s3.delete_object(Bucket=os.environ["R2_BUCKET_NAME"], Key="test/ping.txt")
print("Cleanup OK")

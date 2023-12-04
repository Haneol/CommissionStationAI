import boto3
import requests

s3_client = boto3.client('s3')


def read_file_from_s3(bucket_name, file_key):
    response = s3_client.get_object(Bucket=bucket_name, Key=file_key)
    return response['Body'].read().decode('utf-8')


def parse_s3_link(s3_url):
    parts = s3_url.split('/')
    bucket_name = parts[2].split('.')[0]
    key = '/'.join(parts[3:])

    return bucket_name, key


def read_text_file(url):
    response = requests.get(url)
    if response.status_code == 200:
        return response.text
    else:
        return "Error: Unable to access the file at the provided URL."

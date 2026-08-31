# from google.cloud import storage


# PROJECT_ID = "project-b2f1328b-69cb-4843-a87"
# BUCKET_NAME = "hcl-gcp-agent-data"
# FILE_NAME = "orders_clean.csv"


# client = storage.Client(project=PROJECT_ID)

# bucket = client.bucket(BUCKET_NAME)

# blob = bucket.blob(FILE_NAME)

# blob.upload_from_filename("orders.csv")

# print("File uploaded successfully!")

from google.cloud import bigquery


PROJECT_ID = "project-b2f1328b-69cb-4843-a87"

client = bigquery.Client(project=PROJECT_ID)

table_id = "hcl_demo.orders_from_gcs"

job_config = bigquery.LoadJobConfig(
    source_format=bigquery.SourceFormat.CSV,
    skip_leading_rows=1,
    autodetect=True,
)


uri = "gs://hcl-gcp-agent-data/orders_clean.csv"


load_job = client.load_table_from_uri(
    uri,
    table_id,
    job_config=job_config,
)

load_job.result()

print("Data loaded into BigQuery!")
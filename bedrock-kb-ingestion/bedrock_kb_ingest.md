# Amazon Bedrock Knowledge Base Document Ingestion

This script helps you ingest large numbers of documents into an Amazon Bedrock Knowledge Base, handling the 1000-document limit per API call by automatically batching your documents.

## Prerequisites

- AWS CLI configured with appropriate permissions
- Python 3.6+
- Boto3 library
- An existing Amazon Bedrock Knowledge Base
- Documents stored in an S3 bucket

## Installation

```bash
pip install boto3
```

## Usage

```bash
python bedrock_kb_ingest.py --knowledge-base-id your-kb-id --bucket your-s3-bucket --prefix documents/
```

### Arguments

- `--knowledge-base-id`: (Required) Your Bedrock Knowledge Base ID
- `--bucket`: (Required) S3 bucket containing your documents
- `--prefix`: (Optional) S3 prefix/folder containing your documents
- `--region`: (Optional) AWS region (default: us-east-1)
- `--wait`: (Optional) Wait for each batch to complete before starting the next

## Example

To ingest all documents from an S3 bucket and wait for each batch to complete:

```bash
python bedrock_kb_ingest.py \
  --knowledge-base-id 1a2b3c4d \
  --bucket my-document-bucket \
  --prefix knowledge-base-docs/ \
  --region us-west-2 \
  --wait
```

## Monitoring

The script logs the progress of each batch ingestion. You can also check the status of ingestion jobs in the AWS Console or using the AWS CLI:

```bash
aws bedrock-agent get-knowledge-base-ingestion-job \
  --knowledge-base-id your-kb-id \
  --ingestion-job-id your-job-id
```

## Notes

- The script handles pagination for S3 objects, so it works with any number of documents
- Each batch contains up to 25 documents as per API limitations
- The script automatically skips folder objects in S3

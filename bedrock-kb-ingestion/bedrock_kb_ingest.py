#!/usr/bin/env python3
"""
Script to ingest documents into an Amazon Bedrock Knowledge Base.
Handles batching of documents (25 per API call) for large document sets.
"""

import boto3
import argparse
import time
import logging
import json
from botocore.exceptions import ClientError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def list_s3_objects(bucket, prefix=''):
    """List all objects in an S3 bucket with the given prefix."""
    s3_client = boto3.client('s3')
    paginator = s3_client.get_paginator('list_objects_v2')
    
    all_objects = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        if 'Contents' in page:
            all_objects.extend([obj['Key'] for obj in page['Contents']])
    
    return all_objects

def batch_documents(s3_objects, bucket, batch_size=25):
    """Create batches of S3 document references."""
    batches = []
    current_batch = []
    
    for obj_key in s3_objects:
        if len(current_batch) >= batch_size:
            batches.append(current_batch)
            current_batch = []
        
        # Skip folders or empty objects
        if obj_key.endswith('/'):
            continue
            
        current_batch.append({
            'content': {
                'dataSourceType': 'S3',
                's3': {
                    's3Location': {
                        'uri': f"s3://{bucket}/{obj_key}"
                    }
                }
            }
        })
    
    # Add the last batch if it's not empty
    if current_batch:
        batches.append(current_batch)
    
    return batches

def ingest_documents_batch(bedrock_agent_client, knowledge_base_id, data_source_id, documents):
    """Ingest a batch of documents into the knowledge base."""
    try:
        response = bedrock_agent_client.ingest_knowledge_base_documents(
            knowledgeBaseId=knowledge_base_id,
            dataSourceId=data_source_id,
            documents=documents
        )
        
        # Debug the response
        logger.debug(f"API Response: {json.dumps(response, default=str)}")
        
        # Check if ingestionJobId exists in the response
        if 'ingestionJobId' in response:
            return response['ingestionJobId']
        elif 'jobId' in response:
            return response['jobId']
        else:
            # If neither key exists, log the response and return a placeholder
            logger.warning(f"Unexpected response format: {json.dumps(response, default=str)}")
            return f"unknown-job-{time.time()}"
            
    except ClientError as e:
        logger.error(f"Error ingesting documents: {e}")
        raise

def check_ingestion_status(bedrock_agent_client, knowledge_base_id, data_source_id, ingestion_job_id):
    """Check the status of an ingestion job."""
    try:
        # Skip status check for placeholder job IDs
        if ingestion_job_id.startswith("unknown-job-"):
            logger.warning(f"Skipping status check for unknown job ID: {ingestion_job_id}")
            return "UNKNOWN"
            
        try:
            response = bedrock_agent_client.get_ingestion_job(
                knowledgeBaseId=knowledge_base_id,
                dataSourceId=data_source_id,
                ingestionJobId=ingestion_job_id
            )
            return response['status']
        except ClientError as e:
            if "ResourceNotFoundException" in str(e):
                # Try alternative API if the first one fails
                try:
                    response = bedrock_agent_client.list_ingestion_jobs(
                        knowledgeBaseId=knowledge_base_id,
                        dataSourceId=data_source_id,
                        filters=[{'field': 'jobId', 'value': ingestion_job_id}]
                    )
                    if response.get('ingestionJobs'):
                        return response['ingestionJobs'][0]['status']
                    else:
                        return "NOT_FOUND"
                except Exception as inner_e:
                    logger.error(f"Error in alternative status check: {inner_e}")
                    return "ERROR"
            else:
                raise
    except Exception as e:
        logger.error(f"Error checking ingestion status: {e}")
        return "ERROR"

def filter_metadata_files(s3_objects):
    """Filter out metadata files to avoid duplicate ingestion."""
    return [obj for obj in s3_objects if not obj.endswith('.metadata.json')]

def main():
    parser = argparse.ArgumentParser(description='Ingest documents into Amazon Bedrock Knowledge Base')
    parser.add_argument('--knowledge-base-id', required=True, help='Knowledge Base ID')
    parser.add_argument('--data-source-id', required=True, help='Data Source ID')
    parser.add_argument('--bucket', required=True, help='S3 bucket containing documents')
    parser.add_argument('--prefix', default='', help='S3 prefix (folder) containing documents')
    parser.add_argument('--region', default='us-east-1', help='AWS region')
    parser.add_argument('--wait', action='store_true', help='Wait for each batch to complete before starting next')
    parser.add_argument('--debug', action='store_true', help='Print debug information')
    parser.add_argument('--batch-size', type=int, default=25, help='Number of documents per batch (max 25)')
    parser.add_argument('--skip-metadata', action='store_true', help='Skip .metadata.json files')
    args = parser.parse_args()

    # Set debug level if requested
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    # Initialize Bedrock Agent client
    bedrock_agent_client = boto3.client('bedrock-agent', region_name=args.region)
    
    # List all objects in the S3 bucket/prefix
    logger.info(f"Listing objects in s3://{args.bucket}/{args.prefix}")
    s3_objects = list_s3_objects(args.bucket, args.prefix)
    
    # Filter out metadata files if requested
    if args.skip_metadata:
        original_count = len(s3_objects)
        s3_objects = filter_metadata_files(s3_objects)
        logger.info(f"Filtered out {original_count - len(s3_objects)} metadata files")
    
    logger.info(f"Found {len(s3_objects)} objects in S3")
    
    # Ensure batch size doesn't exceed API limit
    batch_size = min(args.batch_size, 25)
    if args.batch_size > 25:
        logger.warning(f"Requested batch size {args.batch_size} exceeds API limit. Using maximum of 25.")
    
    # Create batches of documents
    document_batches = batch_documents(s3_objects, args.bucket, batch_size)
    logger.info(f"Created {len(document_batches)} batches of documents (max {batch_size} per batch)")
    
    # Debug: Print the structure of the first document if requested
    if args.debug and document_batches and document_batches[0]:
        logger.debug(f"First document structure: {json.dumps(document_batches[0][0], indent=2)}")
    
    # Process each batch
    ingestion_jobs = []
    for i, batch in enumerate(document_batches):
        logger.info(f"Processing batch {i+1}/{len(document_batches)} with {len(batch)} documents")
        
        try:
            job_id = ingest_documents_batch(
                bedrock_agent_client, 
                args.knowledge_base_id,
                args.data_source_id,
                batch
            )
            ingestion_jobs.append(job_id)
            logger.info(f"Started ingestion job {job_id} for batch {i+1}")
            
            if args.wait:
                logger.info(f"Waiting for batch {i+1} to complete...")
                status = "IN_PROGRESS"
                while status in ["IN_PROGRESS", "QUEUED", "PENDING"]:
                    time.sleep(30)  # Check every 30 seconds
                    status = check_ingestion_status(
                        bedrock_agent_client, 
                        args.knowledge_base_id, 
                        args.data_source_id,
                        job_id
                    )
                    logger.info(f"Batch {i+1} status: {status}")
                
                if status not in ["COMPLETE", "COMPLETED", "SUCCESS"]:
                    logger.warning(f"Batch {i+1} finished with status: {status}")
            else:
                # Add a small delay between batches to avoid throttling
                time.sleep(2)
        except Exception as e:
            logger.error(f"Error processing batch {i+1}: {e}")
            if args.debug:
                import traceback
                logger.debug(traceback.format_exc())
    
    logger.info(f"Started {len(ingestion_jobs)} ingestion jobs")
    for i, job_id in enumerate(ingestion_jobs):
        logger.info(f"Batch {i+1}: Job ID {job_id}")
    
    logger.info("Document ingestion process initiated successfully")

if __name__ == "__main__":
    main()

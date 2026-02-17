#!/usr/bin/env python3
"""
Alibaba Cluster Trace Dataset Extraction and Preprocessing Script
"""

import os
import tarfile
import pandas as pd
import logging
from typing import List

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler('dataset_extraction.log'),
        logging.StreamHandler()
    ]
)

def extract_tarball(tarball_path: str, extract_path: str) -> None:
    """
    Extract the tarball to a specified directory
    """
    try:
        # Create extraction directory if it doesn't exist
        os.makedirs(extract_path, exist_ok=True)
        
        # Extract the tarball
        with tarfile.open(tarball_path, 'r:gz') as tar:
            tar.extractall(path=extract_path)
        
        logging.info(f"Successfully extracted {tarball_path} to {extract_path}")
    except Exception as e:
        logging.error(f"Error extracting tarball: {e}")
        raise

def list_extracted_files(extract_path: str) -> List[str]:
    """
    List all files in the extracted directory
    """
    try:
        files = []
        for root, _, filenames in os.walk(extract_path):
            for filename in filenames:
                files.append(os.path.join(root, filename))
        
        logging.info(f"Found {len(files)} files in {extract_path}")
        return files
    except Exception as e:
        logging.error(f"Error listing files: {e}")
        return []

def process_batch_task_data(file_path: str, output_path: str) -> pd.DataFrame:
    """
    Process batch task data and prepare for analysis
    """
    try:
        # Read the CSV file with different parsing strategies
        # Try various delimiter and header options
        parsing_strategies = [
            {'sep': ',', 'header': 0},     # Standard CSV
            {'sep': '\t', 'header': None}, # Tab-separated, no header
            {'sep': ',', 'header': None},  # Comma-separated, no header
        ]
        
        df = None
        for strategy in parsing_strategies:
            try:
                df = pd.read_csv(file_path, **strategy)
                
                # Print full diagnostic information
                print("DataFrame Info:")
                print(df.info())
                print("\nFirst few rows:")
                print(df.head())
                print("\nColumn names:")
                print(list(df.columns))
                
                break
            except Exception as e:
                print(f"Failed parsing strategy {strategy}: {e}")
                continue
        
        if df is None:
            logging.error("Could not parse the CSV file with any strategy")
            return pd.DataFrame()
        
        # Advanced column detection and mapping
        def detect_resource_columns(df):
            # Strategy to identify resource-related columns
            potential_columns = {
                'cpu_request': None,
                'cpu_usage': None,
                'memory_request': None,
                'memory_usage': None
            }
            
            # Look for columns that might represent resources
            for col in df.columns:
                # Convert column to lowercase for easier matching
                col_name = str(col).lower()
                
                # Check for CPU-related columns
                if 'cpu' in col_name:
                    if 'request' in col_name:
                        potential_columns['cpu_request'] = col
                    elif 'usage' in col_name:
                        potential_columns['cpu_usage'] = col
                
                # Check for memory-related columns
                if 'memory' in col_name:
                    if 'request' in col_name:
                        potential_columns['memory_request'] = col
                    elif 'usage' in col_name:
                        potential_columns['memory_usage'] = col
            
            return potential_columns
        
        # Detect resource columns
        resource_columns = detect_resource_columns(df)
        
        # Filter out None values
        valid_columns = {k: v for k, v in resource_columns.items() if v is not None}
        
        print("\nDetected Resource Columns:")
        print(valid_columns)
        
        if not valid_columns:
            logging.error("No resource-related columns found!")
            return pd.DataFrame()
        
        # Select detected columns
        df_processed = df[[col for col in valid_columns.values()]]
        
        # Rename columns to standard names
        df_processed.columns = list(valid_columns.keys())
        
        # Basic type conversion and cleaning
        for col in df_processed.columns:
            df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
        
        # Remove rows with negative or invalid values
        df_processed = df_processed[
            (df_processed >= 0).all(axis=1)
        ]
        
        # Save processed data
        df_processed.to_csv(output_path, index=False)
        
        logging.info(f"Processed data saved to {output_path}")
        logging.info(f"Processed data shape: {df_processed.shape}")
        
        return df_processed
    
    except Exception as e:
        logging.error(f"Error processing batch task data: {e}")
        return pd.DataFrame()

def main():
    """
    Main function to orchestrate dataset extraction and preprocessing
    """
    # Paths - adjust these to match your file locations
    TARBALL_PATH = 'data/raw/trace_201708.tgz'
    EXTRACT_PATH = 'data/raw/clusterdata2018'
    
    # Try multiple potential output files
    POTENTIAL_BATCH_TASK_FILES = [
        'data/raw/clusterdata2018/batch_task.csv',
        'data/raw/clusterdata2018/container_usage.csv',
        'data/raw/clusterdata2018/server_usage.csv'
    ]
    
    PROCESSED_OUTPUT_PATH = 'data/processed/batch_task_processed.csv'
    
    try:
        # Step 1: Extract tarball
        extract_tarball(TARBALL_PATH, EXTRACT_PATH)
        
        # Step 2: List extracted files
        extracted_files = list_extracted_files(EXTRACT_PATH)
        
        # Try processing files until successful
        for batch_task_file in POTENTIAL_BATCH_TASK_FILES:
            if os.path.exists(batch_task_file):
                print(f"\nAttempting to process: {batch_task_file}")
                result = process_batch_task_data(batch_task_file, PROCESSED_OUTPUT_PATH)
                
                if not result.empty:
                    logging.info(f"Successfully processed {batch_task_file}")
                    break
            else:
                print(f"File not found: {batch_task_file}")
        
        logging.info("Dataset extraction and preprocessing completed.")
    
    except Exception as e:
        logging.error(f"Dataset processing failed: {e}")

if __name__ == '__main__':
    main()
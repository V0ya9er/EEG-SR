import sys
import os
import torch
import numpy as np

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.datamodule import EEGDataModule

def debug_targets():
    print("Initializing DataModule...")
    dm = EEGDataModule(
        dataset_name="BNCI2014_001",
        subject_ids=[1],
        batch_size=4,
        num_workers=0
    )
    
    print("Preparing data...")
    dm.prepare_data()
    
    print("Setting up data...")
    dm.setup()
    
    print("Checking Train Set Targets...")
    # Access the underlying dataset
    train_set = dm.train_set
    
    # Fetch a few samples
    print(f"Train set size: {len(train_set)}")
    
    # Check first 5 samples
    for i in range(5):
        sample = train_set[i]
        # Braindecode WindowsDataset returns (x, y, ind)
        x, y, ind = sample
        print(f"Sample {i}: Target (y) = {y}, Type = {type(y)}")
        
    # Check if target_transform is present on the underlying dataset
    # train_set is a Subset, so we need to access .dataset
    underlying_ds = train_set.dataset
    
    # underlying_ds is likely a BaseConcatDataset
    print(f"Underlying dataset type: {type(underlying_ds)}")
    
    if hasattr(underlying_ds, 'datasets'):
        print(f"Number of concatenated datasets: {len(underlying_ds.datasets)}")
        first_ds = underlying_ds.datasets[0]
        print(f"First dataset type: {type(first_ds)}")
        if hasattr(first_ds, 'target_transform'):
            print(f"First dataset target_transform: {first_ds.target_transform}")
        else:
            print("First dataset has no target_transform attribute")
            
        # Check raw targets in y
        if hasattr(first_ds, 'y'):
            print(f"First dataset y (first 5): {first_ds.y[:5]}")
    
if __name__ == "__main__":
    debug_targets()
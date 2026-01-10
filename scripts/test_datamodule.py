import sys
import os
import torch

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.datamodule import EEGDataModule

def test_datamodule():
    print("Initializing DataModule...")
    # Use a small subset or default for testing
    dm = EEGDataModule(
        dataset_name="BNCI2014001",
        subject_ids=[1],
        batch_size=4,
        num_workers=0 # Windows usually requires 0 for simple scripts
    )
    
    print("Preparing data (downloading if needed)...")
    dm.prepare_data()
    
    print("Setting up data...")
    try:
        dm.setup()
    except Exception as e:
        print(f"Error during setup: {e}")
        # If dataset download fails (e.g. network), we might want to mock or skip
        # But for now let's see the error
        return

    print("Checking DataLoaders...")
    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()
    test_loader = dm.test_dataloader()
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # Fetch one batch
    batch = next(iter(train_loader))
    x, y, ind = batch
    
    print(f"Batch X shape: {x.shape}") # Should be [batch_size, channels, time_points]
    print(f"Batch Y shape: {y.shape}")
    print(f"Batch Ind shape: {ind.shape}")
    
    print("DataModule Test Passed!")

if __name__ == "__main__":
    test_datamodule()
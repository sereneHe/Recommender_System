import os
from pathlib import Path
import pytest
import torch
from torch.utils.data import Dataset
from loguru import logger
from mlo_group_project.data import BreastCancerData

# Define project root dynamically to ensure paths work on any machine
PROJECT_ROOT = Path(__file__).resolve().parents[1]

#Init and raw data test 
def test_data_dataset_is_created_correctly_from_raw_csv():
    
    logger.debug(f"\nPytest is running from: {os.getcwd()}")
    raw_path = PROJECT_ROOT / "data" / "raw" / "bcw.csv"
    
    if not raw_path.exists():
        pytest.skip(f"Raw data file not found at {raw_path}. Cannot test dataset creation.")

    dataset = BreastCancerData(Path(raw_path))
    assert isinstance(dataset, Dataset)

#Dimension test
def test_data_sample_file_structure_and_dimensions_are_correct():
    
    data_path = PROJECT_ROOT / "tests" / "sample_data.pt"
    
    #If sample is not there raise error
    if not data_path.exists():
        #This can changed to skip if needed
        pytest.fail(f"Sample data missing at {data_path}")

    images, targets = torch.load(data_path)

    # Shape check
    assert images.ndim == 2
    assert images.shape == (5, 30)
    assert images.shape[1] == 30
    assert images.shape[0] == targets.shape[0]
    assert targets.shape == (5,)

    # Type check
    assert isinstance(images, torch.Tensor)
    assert isinstance(targets, torch.Tensor)

#Processed Data test 
def test_processed_data_files_exist_and_contain_tensors_with_thirty_features():
    
    train_path = PROJECT_ROOT / "data" / "processed" / "train.pt"
    test_path = PROJECT_ROOT / "data" / "processed" / "test.pt"

    # Skip if data hasn't been generated yet
    if not train_path.exists() or not test_path.exists():
        pytest.skip(f"Data files not found at {train_path}. Please run 'make data' first.")

    # Load data
    train_x, train_y = torch.load(train_path)
    test_x, test_y = torch.load(test_path)

    # Check 1: Training data features
    assert train_x.shape[1] == 30, \
        f"Expected 30 training features (columns), but got {train_x.shape[1]}"

    # Check 2: Test data features
    assert test_x.shape[1] == 30, \
        f"Expected 30 test features (columns), but got {test_x.shape[1]}"

    # Check 3: Labels should match row counts
    assert train_x.shape[0] == train_y.shape[0], \
        "Mismatch between training features and labels count"

#Normalised data test
def test_training_data_is_scaled_between_zero_and_one_using_minmax_scaler():
    
    train_path = PROJECT_ROOT / "data" / "processed" / "train.pt"

    if not train_path.exists():
        pytest.skip("Processed training data file not found.")

    train_x, _ = torch.load(train_path)

    # Check Min: Should be >= 0.0
    min_val = train_x.min().item()
    assert min_val >= 0.0, f"MinMaxScaler failed? Found value less than 0: {min_val}"

    # Check Max: Should be <= 1.0 (with small float tolerance)
    max_val = train_x.max().item()
    assert max_val <= 1.0001, f"MinMaxScaler failed? Found value greater than 1: {max_val}"


#Leak check
def test_no_data_leakage_between_train_and_test_sets():
    
    train_path = PROJECT_ROOT / "data" / "processed" / "train.pt"
    test_path = PROJECT_ROOT / "data" / "processed" / "test.pt"

    if not train_path.exists():
        pytest.skip("Data missing")

    train_x, _ = torch.load(train_path)
    test_x, _ = torch.load(test_path)

    #Convert tensors to sets of tuples for fast comparison and round to 4 
    train_set = set(tuple(x.round(decimals=4).tolist()) for x in train_x)
    test_set = set(tuple(x.round(decimals=4).tolist()) for x in test_x)

    #The intersection should be empty
    common_samples = train_set.intersection(test_set)
    assert len(common_samples) == 0, f"Data Leakage! Found {len(common_samples)} samples that are in both Train and Test sets."

#Balance variety among classes test
def test_targets_contain_both_benign_and_malignant_classes():
    
    train_path = PROJECT_ROOT / "data" / "processed" / "train.pt"
    if not train_path.exists():
        pytest.skip("Data missing")
        
    _, train_y = torch.load(train_path)
    
    unique_classes = torch.unique(train_y)
    
    #We expect class 0 (Benign) and class 1 (Malignant) to both exist
    assert 0 in unique_classes, "Training data is missing Class 0 (Benign)!"
    assert 1 in unique_classes, "Training data is missing Class 1 (Malignant)!"

import yaml

from src.data_loader import DatasetBatchManager, DatasetWindowTimeManager


def test_load_config():
    with open("./config/default.yaml", "r") as file:
        config = yaml.safe_load(file)
    assert "batch_data_manager" in config
    batch_config = config["batch_data_manager"]
    for k, v in batch_config.items():
        assert "dataset_path" in v
        assert "dataset_type" in v
        assert "batch_range" in v
        assert "initial_fraction" in v

    assert "window_frame_data_manager" in config
    window_frame_config = config["window_frame_data_manager"]
    for k, v in window_frame_config.items():
        assert "dataset_path" in v
        assert "dataset_type" in v
        assert "window_size" in v
        assert "step_size" in v
        assert "initial_fraction" in v

def test_batch_loader():
    batch_loader = DatasetBatchManager()
    batch_loader.get_dataset(
        dataset_path="./dataset/CollegeMsg.txt",
        dataset_type="college_msg",
        batch_range=1e-3,
        initial_fraction=0.85,
    )

def test_windowframe_loader():    
    frame_loader = DatasetWindowTimeManager()
    frame_loader.get_dataset(
        dataset_path="./dataset/CollegeMsg.txt",
        dataset_type="college_msg",
        window_size=30,
        step_size=1,
        initial_fraction=0.75,
    )

import yaml
from pathlib import Path


def create_data_yaml():
    # Define path where the YAML will be saved
    data_dir = Path("Helmet detection")
    data_dir.mkdir(parents=True, exist_ok=True)  # Ensure the folder exists

    # Define content of the YAML file
    data = {
        'train': str((data_dir / 'dataset/images/train').resolve()),  # Full path to training images
        'val': str((data_dir / 'dataset/images/val').resolve()),      # Full path to validation images
        'nc': 2,                                               # Number of classes
        'names': ['helmet', 'no_helmet']            # Class names
    }

    # Save it as a YAML file
    with open('data.yaml', 'w') as f:
        yaml.dump(data, f)

    print("✅ data.yaml created at Helmet detection/data.yaml")

create_data_yaml()



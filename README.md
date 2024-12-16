# ABBCA ML Materials

This repository provides tools to convert datasets exported from LabelStudio into the COCO format using Python. It streamlines the process of dataset preparation for machine learning pipelines.

## Features
- Convert LabelStudio datasets to COCO format.
- Specify input classes, data sources, and output directories.
- Supports dataset partitioning (e.g., train/test splits).
- Configurable via command-line arguments.

## Requirements
- Python 3.11+
- Libraries: `argparse`, `os`, and other dependencies (install via `requirements.txt`).

## Installation
Clone the repository and install dependencies:
```bash
git clone https://github.com/Leael/abbca-ml-materials.git
cd abbca-ml-materials
pip install -r requirements.txt
```

## Usage
Run the script with the required arguments:
```bash
python labelstudio-coco.py --input-classes vehicle,animal \
                          --input-data labelstudio-default:100,200,500 \
                          --output-dir ./converted-datasets \
                          --output-type coco \
                          --partition train:0.7,validation:0.3 \
                          --token my_labelstudio_token
```

### Command-Line Arguments

| Argument         | Required | Description                                                                                                   |
|------------------|----------|---------------------------------------------------------------------------------------------------------------|
| `--input-classes` | Yes      | Comma-separated list of classes to be used (e.g., `person_body,person_head`).                                 |
| `--input-data`    | No       | Comma-separated input data sources (e.g., `labelstudio-default:0,2,3`). Defaults to an empty string.            |
| `--output-dir`    | No       | Path to save the converted dataset. Defaults to the current working directory.                                |
| `--output-type`   | Yes      | Format of the output dataset (e.g., `coco`).                                             |
| `--partition`     | No       | Key-value pairs for dataset partitions (e.g., `test:0.2,train:0.8`).                                         |
| `--token`         | Yes      | Authentication token for LabelStudio.                                                                        |

### Example
Convert a LabelStudio dataset into COCO format, specifying classes and partitions:
```bash
python labelstudio-coco.py --input-classes vehicle,animal \
                          --input-data labelstudio-default:100,200,500 \
                          --output-dir ./converted-datasets \
                          --output-type coco \
                          --partition train:0.7,validation:0.3 \
                          --token my_labelstudio_token
```

## Directory Structure
```
abbca-ml-materials/
├── labelstudio-coco.py # Main script for dataset conversion
├── labelstudio.py      # Main script for dataset conversion
├── requirements.txt    # Required Python dependencies
├── README.md           # Project documentation
```


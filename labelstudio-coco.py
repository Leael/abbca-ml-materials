import argparse
import os
from labelstudio import LabelStudio

label_studio = LabelStudio()


def parse_partition(partition: str) -> dict:
    """
    Parses a partition string into a dictionary.
    The partition string should be in the format "key1:value1,key2:value2,..."
    where each value is a float. The sum of all values must equal 1.
    Args:
        partition (str): The partition string to parse.
    Returns:
        dict: A dictionary where the keys are the partition names and the values are the partition values.
    Raises:
        Exception: If the partition string is invalid, if a value is not a number, or if the total partition value does not equal 1.
    """
    dict_partition: dict = {}
    total_val = 0

    if partition is None:
        dict_partition["default"] = 1
        return dict_partition
    
    for p in partition.split(','):
        pair = p.split(':')
        
        if len(pair) != 2:
            raise Exception("Invalid partition parameter")
        
        key = pair[0]
        
        try:
            val = float(pair[1])
        except ValueError:
            raise Exception("Partition value part should be a number")
        
        total_val += val
        
        dict_partition[key] = val
        
    if total_val != 1:
        raise Exception("Total partition value must be equal to 1")
    
    return dict_partition


def main(input_classes, input_data, output_dir, output_type, partition, token):
    input_classes = input_classes.split(",")

    for int_data in input_data:
        partition_dict = parse_partition(partition)
        label_studio.prepare_data(input_datas=int_data, output_dir=output_dir, token=token)
        label_studio.convert_to_coco(int_data, output_dir, partition_dict, input_classes)


def parse_args():
    """
    Parses command-line arguments.

    Returns:
        argparse.Namespace: Parsed command-line arguments.

    Arguments:
        --input-classes (str): Classes to be used. Example: --classes person_body,person_head. (required)
        --input-data (str): Input data separated by comma. Example: --input-data labelstudio-spark:0,2,3. (optional, default: "")
        --output-dir (str): Output path of the converted dataset. (optional, default: current working directory)
        --output-type (str): Destination of dataset. Example: coco, tfrecord, satrn. (required)
        --partition (str): Key value pair for partitions separated by comma and pairs separated by semicolon. Example: --partition test:0.2,train:0.8. (optional)
        --token (str): Labelstudio token. (required)
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-classes", required=True, help="Classes to be used. i.e --classes person_body,person_head")
    parser.add_argument('--input-data', default="", required=False, nargs='+', help='Input data separated by comma. i.e --input-data labelstudio-spark:0,2,3')
    parser.add_argument('--output-dir', default=os.getcwd(), help='Output path of the converted dataset')
    parser.add_argument('--output-type', default="", required=True, help='Destination of dataset. i.e. coco, tfrecord, satrn')
    parser.add_argument("--partition", help="Key value pair for partitions separated by comma and pairs separated by semicolon. i.e --partition test:0.2,train:0.8")
    parser.add_argument("--token", default="", required=True, help="Labelstudio token")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(**vars(args))

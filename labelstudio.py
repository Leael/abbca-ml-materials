import os
import string
import threading
import uuid
import requests
import random
import numpy as np
from tqdm import tqdm
from threading import Thread
# from google.cloud import storage
from PIL import Image

threadLimiter = threading.BoundedSemaphore(20)
# # Initialise a client
# storage_client = storage.Client("cloud storage name") # change this to your cloud storage name
# # Create a bucket object for our bucket
# bucket = storage_client.get_bucket("bucket name") # change this to your bucket name


class LabelStudio():
    def __init__(self):
        self.captures = {}
        self.workers = []

    def download_images(self, img_url, file_name, sub_bucket):
        """
        Downloads an image from a given URL and saves it to a specified file path.

        Args:
            img_url (str): The URL of the image to be downloaded.
            file_name (str): The local file path where the image will be saved.
            sub_bucket (str): The name of the sub-bucket in the cloud storage.

        Raises:
            Exception: If there is an error during the download process.

        Note:
            This method checks if the file already exists before attempting to download it.
            It also ensures that the directory structure for the file path exists.
            The threadLimiter is released in the finally block to ensure it is always executed.
        """
        try:
            if not os.path.exists(file_name):
                blob = bucket.blob(img_url.replace(f'gs://{sub_bucket}/', ''))
                os.makedirs(os.path.dirname(file_name), exist_ok=True)
                blob.download_to_filename(file_name)
        finally:
            threadLimiter.release()

    def xy2centerxy(self, x, y, w, h):
        xcenter = x + w / 2.0
        ycenter = y + h / 2.0
        return xcenter, ycenter

    def prepare_data(self, input_datas: str, output_dir: str, token: str):
        global bucket
        sub_bucket = ""
        workers = []
        input_data = input_datas.split(':')[-1]
        labelstudio_name = input_datas.split(':')[0].split('-')[1]
        if labelstudio_name == 'default':
            headers = {"Authorization": f"Token {token}"}
        #     sub_bucket = 'bucket name' # change this to your bucket name
        # Incase you have multiple bucket
        # elif token == 'new':
        #     headers = {"Authorization": f"Token {token}"}
        #     bucket = storage_client.get_bucket("new bucket name") # change this to your new bucket name
        #     sub_bucket = 'new bucket name' # change this to your new bucket name

        for data_id in input_data.split(","):
            url = f"https://labelstudio.skunkworks.ai/api/projects/{data_id}/export?exportType=JSON" # change this to your labelstudio uri
            result = requests.get(url, headers=headers)
            response_json = result.json()
            self.captures[data_id] = response_json
        
        # for Downloading Images stored in cloud storage
        # for data_id, json_data in self.captures.items():
        #     pbar = tqdm(json_data, desc=f"Downloading images in {os.path.join(output_dir, 'downloaded', 'labelstudio', data_id)}")
        #     for annotation in json_data:
        #         img_url = annotation["data"]["image"]
        #         filename = img_url.split("/")[-1]
        #         file_name = os.path.join(output_dir, 'images', data_id, filename)
        #         threadLimiter.acquire()
        #         worker = Thread(target=self.download_images, args=[img_url, file_name, sub_bucket])
        #         pbar.update(1)
        #         worker.daemon = True
        #         worker.start()
        #         workers.append(worker)
        #     for worker in workers:
        #         worker.join()
        #     pbar.close()

    def convert_to_coco(self, input_data, output_dir, partition, input_classes):
        input_classes.sort()
        annotated_classes = []
        coco_format = []
        datasets = {}

        # list all classes in annotation
        for json_data in self.captures.values():
            for annotations in json_data:
                for result in annotations['annotations'][0]['result']:
                    if len(result) == 0:
                        background_image = True
                        continue
                    if 'value' not in result or 'rectanglelabels' not in result['value'] or len(result['value']['rectanglelabels']) <= 0:
                        continue
                    label = result['value']['rectanglelabels'][0]
                    if label not in annotated_classes:
                        annotated_classes.append(label)
        # check is the classes inputted are in annotated classes. Exit if not exist
        for x in input_classes:
            if x not in annotated_classes:
                print(f"{x} not in annotation. Annotated classes: {annotated_classes}")
                exit()

        # Convert to COCO format
        for data_id, json_data in self.captures.items():
            temp_coco = []
            pbar = tqdm(json_data, desc=f"Converting {data_id} to COCO format")
            for annot in json_data:
                image_name = annot['data']['image'].split('/')[-1]
                filename = os.path.join(output_dir, 'images', data_id, image_name)

                format = {
                    "prefix": data_id,
                    "imageSrc": filename,
                    "imageURL": annot['data']['image'],
                    "width": 0,
                    "height": 0,
                    "annotations": []
                }
                for result in annot['annotations'][0]['result']:
                    if 'value' not in result or 'rectanglelabels' not in result['value'] or len(result['value']['rectanglelabels']) <= 0:
                        continue
                    label = result['value']['rectanglelabels'][0]
                    if label not in input_classes:
                        continue
                    width = result['original_width']
                    height = result['original_height']
                    x = result['value']['x'] / 100.0 
                    y = result['value']['y'] / 100.0 
                    w = result['value']['width'] / 100.0 
                    h = result['value']['height'] / 100.0
                    label_index = input_classes.index(label)
                    x_center, y_center = self.xy2centerxy(x, y, w, h)

                    format["width"] = width
                    format["height"] = height
                    format["annotations"].append(dict(
                        label=label,
                        labelIndex=label_index,
                        xCenter=x_center,
                        yCenter=y_center,
                        w=w,
                        h=h
                    ))
                if len(format["annotations"]) > 0:
                    temp_coco.append(format)
                    pbar.update(1)
            pbar.close()
            coco_format += temp_coco

        # Shuffle the COCO formatted data to ensure random distribution
        random.shuffle(coco_format)
        
        start = 0
        end = 0
        
        # Partition the data according to the specified ratios
        for folder, value in partition.items():
            f = np.array(value).dot(len(coco_format)).astype(np.int32)
            end += f
            if folder not in datasets:
                datasets[folder] = []
            datasets[folder] += coco_format[start:end]
            start += f
        
        for folder, datas in datasets.items():
            pbar = tqdm(datas, desc=f"Processing... {folder} data")
            for data in datas:
                threadLimiter.acquire()
                worker = Thread(target=self.save_coco_format, args=[data, folder, output_dir])
                worker.daemon = True
                worker.start()
                pbar.update(1)
                self.workers.append(worker)
            pbar.close

            for worker in self.workers:
                worker.join()
        
        # Save yaml file
        yaml_file = f"path: {os.path.join(output_dir, 'coco')}\ntrain: {os.path.join(output_dir, 'coco', 'images', 'train')}\nval: {os.path.join(output_dir, 'coco', 'images', 'test')}\ntest: {os.path.join(output_dir, 'coco', 'images', 'val')}\n\nnames:"

        for index, label in enumerate(input_classes):
            yaml_file += f'\n  {index}: {label}'

        os.makedirs(os.path.dirname(os.path.join(output_dir, 'coco', "data.yaml")), exist_ok=True)
        with open(os.path.join(output_dir, 'coco', "data.yaml"), 'w') as f:
            f.write(yaml_file)

        print(f'Converted Dataset saved in {output_dir}')
        print('-----------------COCO Dataset Ready!🚀-----------------')

    def save_coco_format(self, data, folder, output_dir):
        """
        Saves the COCO formatted data to the specified directory.

        Args:
            data (dict): The COCO formatted data to be saved.
            folder (str): The folder name where the data will be saved.
            output_dir (str): The base directory where the data will be saved.

        Note:
            This method saves the image and its corresponding annotations in the specified folder.
            It also ensures that the directory structure for the file path exists.
            The threadLimiter is released in the finally block to ensure it is always executed.
        """
        txt = ''
        pref = data["prefix"]
        img_filename = str(uuid.uuid4()) + ".jpg"

        # Create directories for images and labels if they don't exist
        img_path = os.path.join(output_dir, "coco", "images", folder)
        label_path = os.path.join(output_dir, "coco", "labels", folder)
        os.makedirs(img_path, exist_ok=True)
        os.makedirs(label_path, exist_ok=True)

        txt_filename = os.path.splitext(img_filename)[0] + ".txt"
        for annots in data["annotations"]:
            txt += f'{annots["labelIndex"]} {annots["xCenter"]} {annots["yCenter"]} {annots["w"]} {annots["h"]}\n'

        # Check if the source image exists
        if not os.path.exists(data["imageSrc"]):
            print(f'{data["imageSrc"]} does not Exist ❌')
            return
        rand_char = "".join(random.choices(string.ascii_lowercase + string.digits, k=5))
        
        try:
            # Open the image and convert it to RGB
            with Image.open(data["imageSrc"]) as img:
                new_img_filename = os.path.join(img_path, f"{pref}_{rand_char}_{img_filename}")
                rgb_im = img.convert('RGB')
                rgb_im.save(new_img_filename)
        except Exception as e:
            print(f"Error processing the image: {e}")
            return

        # Save label .txt
        with open(os.path.join(label_path, f"{pref}_{rand_char}_{txt_filename}"), 'w') as f:
            f.write(txt)
        threadLimiter.release()
 


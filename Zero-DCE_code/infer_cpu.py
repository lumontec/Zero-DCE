import torch
import torch.nn as nn
import torchvision
import torch.optim
import os
import time
import glob
import numpy as np
from torchvision import transforms
from PIL import Image, ExifTags

import model  # Importing your model

def load_and_preprocess_image(image_path):
    # Open image with PIL and fix orientation
    image = Image.open(image_path)
    for orientation in ExifTags.TAGS.keys():
        if ExifTags.TAGS[orientation] == 'Orientation':
            try:
                exif = dict(image._getexif().items())
                if orientation in exif:
                    if exif[orientation] == 3:
                        image = image.rotate(180, expand=True)
                    elif exif[orientation] == 6:
                        image = image.rotate(270, expand=True)
                    elif exif[orientation] == 8:
                        image = image.rotate(90, expand=True)
            except (AttributeError, KeyError, IndexError):
                pass  # No EXIF data

    # Resize to 1920x1080
    image = image.resize((1920, 1080))

    # Convert to numpy and normalize
    image = np.asarray(image) / 255.0
    image = torch.from_numpy(image).float().permute(2, 0, 1).unsqueeze(0)  # Shape: (1, C, H, W)

    return image

def lowlight(image_path):
    data_lowlight = load_and_preprocess_image(image_path)

    DCE_net = model.enhance_net_nopool()
    DCE_net.load_state_dict(torch.load('snapshots/Epoch99.pth', map_location=torch.device('cpu')))
    DCE_net.eval()

    start = time.time()
    _, enhanced_image, _ = DCE_net(data_lowlight)
    end_time = time.time() - start
    print(f"Processing time: {end_time:.4f} seconds")

    # Save the result
    result_path = image_path.replace('test_data', 'result')
    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    torchvision.utils.save_image(enhanced_image, result_path)

if __name__ == '__main__':
    with torch.no_grad():
        filePath = 'data/test_data/'
        file_list = os.listdir(filePath)

        for file_name in file_list:
            test_list = glob.glob(filePath + file_name + "/*")
            for image in test_list:
                print(f"Processing: {image}")
                lowlight(image)


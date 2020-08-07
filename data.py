import numpy as np 
import matplotlib.pyplot as plt 
import os 
from PIL import Image, ImageOps, ImageChops
import csv 
import copy
import pickle 

directory = os.path.dirname(__file__)

images = []
labels = []

def get_label(name): 
    return {
        'plus': '10',
        'minus': '11',
        'multiply': '12',
        'divide': '13'
    }.get(name, name)

minlength = 8000
for folder in os.listdir(f'{directory}/data'):
    label = get_label(folder)
    folder = f'{directory}/data/{folder}'
    length = len(os.listdir(folder))

    print(10*'_', label, 10*'_')
    print(f"processing {length} images.. ")

    for image in os.listdir(folder)[-minlength:]:
        image = Image.open(f'{folder}/{image}')
        image = image.convert('L') # convert to grayscale
        image = ImageOps.invert(image)
        
        image_copy = copy.copy(image)
        image_copy = np.asarray(image_copy.getdata(), dtype=np.int)
        image_copy.flatten()
        images.append(image_copy)
        labels.append(label)

    if length < minlength:
        print('generating extra data..')
        generated_images = []
        generated_labels = []

        for image in os.listdir(folder): 

            image = Image.open(f'{folder}/{image}')
            image = image.convert('L') # convert to grayscale
            image = ImageOps.invert(image)

            if folder[-1] not in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']:
                mirrored_image = copy.copy(image)
                mirrored_image = ImageOps.mirror(mirrored_image)
                mirrored_image = np.asarray(mirrored_image.getdata(), dtype=np.int)
                mirrored_image = mirrored_image.flatten()
                generated_labels.append(label)
                generated_images.append(mirrored_image)
            
            for i in [-5, -3, 3, 5]: 
                shifted_image = copy.copy(image)
                shifted_image = ImageChops.offset(shifted_image, i, 0)
                shifted_image = np.asarray(shifted_image.getdata(), dtype=np.int)
                shifted_image = shifted_image.flatten()
                generated_labels.append(label)
                generated_images.append(shifted_image)

            if len(generated_images) >= minlength - length: 
                break

        images += generated_images
        labels += generated_labels

    print(f'finished {folder}')

data = zip(images, labels)

import pickle
with open('data.pickle', 'wb') as f: 
    pickle.dump(data, f)

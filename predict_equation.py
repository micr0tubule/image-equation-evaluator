
from tensorflow.keras.models import load_model
from PIL import Image
import tensorflow as tf 
import numpy as np  
import math

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    tf.config.experimental.set_virtual_device_configuration(
        gpus[0],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=(1024*4))])
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
  except RuntimeError as e:
    print(e)

height = 45
threshold = 90
model = load_model('model.h5')
image_file = 'test_equation.jpg'

def split(image): 
    images = []
    pixel_list_list = []
    discovered = []
    for i in range(len(image)): 
        for j in range(len(image[0])): 
            if image[i][j] > 122.5 and (image[i][j], (i, j)) not in discovered: 
                pixels = find_pixels(image, [[image[i][j], (i, j)]], discovered)
                print(len(pixels))
                if len(pixels) > 90: 
                   pixel_list_list.append(pixels)
                   print(len(pixels))
                discovered += pixels
    image = Image.fromarray(image)
    for pixels in pixel_list_list: 
        start, end = get_corners(pixels)
        images.append((start[1], image.crop((start[1], start[0], end[1], end[0]))))
        # cropped = image.crop((start[1], start[0], end[1], end[0]))
        # cropped.show()
    images.sort(key=lambda x: x[0])
    return [image[1] for image in images]


def get_corners(pixels):
    y = []
    x = [] 
    for pixel in pixels: 
        x.append(pixel[1][0])
        y.append(pixel[1][1])
    
    return (min(x), min(y)), (max(x), max(y))
    
# pixel = (value, position)
def find_pixels(image, start, discovered):
    pixels = start
    for pixel in pixels: 
        surroundings = get_surrounding_pixels(image, pixel[1])
        for surrounding in surroundings: 
            if surrounding[0] > threshold and surrounding not in pixels and surrounding not in discovered: 
                pixels.append(surrounding)
    return pixels


def get_surrounding_pixels(image, position):
    allowed_width = image.reshape(-1, height).shape[1] - 1
    allowed_height = 44
    pixels = [] 
    if position[0] < allowed_width and position[1] < allowed_height: 
        pixels.append((image[position[0] + 1, position[1] + 1], (position[0] + 1, position[1] + 1)))
    if position[1] < allowed_width:     
        pixels.append((image[position[0], position[1] + 1], (position[0], position[1] + 1)))
    if position[0] > 0 and position[1] > 0:
        pixels.append((image[position[0] - 1, position[1] - 1], (position[0] - 1, position[1] - 1)))
    if position[0] > 0:
        pixels.append((image[position[0] - 1, position[1]], (position[0] - 1, position[1])))
    if position[0] < allowed_width and position[1] > allowed_height:
        pixels.append((image[position[0] + 1, position[1] - 1], (position[0] + 1, position[1])))
    if position[1] > 0:
        pixels.append((image[position[0], position[1] - 1], (position[0], position[1] - 1)))
    if position[0] < allowed_width:
        pixels.append((image[position[0] + 1, position[1]], (position[0] + 1, position[1])))
    if position[0] > 0 and position[1] < allowed_height:
        pixels.append((image[position[0] - 1, position[1] + 1], (position[0] - 1, position[1] + 1)))
    return pixels


def get_sign(label): 
    return {
        10: '+',
        11: '-',
        12: '*',
        13: '/'
    }.get(label, label)


def construct_equation(l):     
    signs = [10, 11, 12, 13]
    equation = []
    number = []
    for element in l: 
        if element not in signs: 
            number.append(element)
        else:
            if len(number) == 0: 
                return False
            equation.append(''.join([str(digit) for digit in number]))
            equation.append(get_sign(element))
            number = []
    equation.append(''.join([str(digit) for digit in number]))
    return ' '.join([expression for expression in equation])


image = Image.open(image_file)
image = image.convert('L')
image = np.array(image.getdata()).reshape((height, -1))
images = split(image)

predictions = []
for image in images: 
    leftright = (height - image.size[0]) / 2
    left = math.floor(leftright)
    right = math.ceil(leftright) + image.size[0]
    topdown = (height - image.size[1]) / 2
    top = math.floor(topdown)
    bottom = math.ceil(topdown + image.size[1])
    image = image.crop((-left, -top, right, bottom))
    image = np.array(image.getdata()).reshape(1, height, height, 1)
    predictions.append(np.argmax(model.predict(image)))

equation = construct_equation(predictions)
if equation:
    print(equation)
    print(eval(equation))
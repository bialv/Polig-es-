from random import randint
import random
from .charles import Individual
from PIL import Image, ImageOps, ImageDraw, ImagePath
import numpy as np
from skimage import filters
from skimage.metrics import structural_similarity as ssim
from skimage.filters import sobel
from skimage.feature import canny
from skimage.filters import sobel
from skimage.color import rgb2gray
import numpy as np


def blend_xo(self, parent1, parent2, inner_elitism = True):

    child = Individual(self.l, self.w)
    blend_alpha = random.random()
    child_image = Image.blend(parent1.image, parent2.image, blend_alpha)
    child.image = child_image
    child.array = np.array(child_image)
    child.get_fitness(self.target_image)
    
    if inner_elitism :

        if child.fitness <= min(parent1.fitness, parent2.fitness):
            return child
        
        return None
    else:
        return child

def per_channel_crossover(self, parent1, parent2, inner_elitism = True):

    child = Individual(self.l, self.w)
    array1 = np.array(parent1.image)
    array2 = np.array(parent2.image)
    
    channels = []
    for i in range(3): 
        channels.append(array1[:, :, i] if np.random.rand() > 0.5 else array2[:, :, i])
    
    new_array = np.stack(channels, axis=-1)
    
    child.image = Image.fromarray(new_array.astype('uint8'))
    child.array = new_array
    child.get_fitness(self.target_image)
    
    if inner_elitism :

        if child.fitness <= min(parent1.fitness, parent2.fitness):
            return child
        
        return None
    else:
        return child


def per_channel_crossover_2(self, parent1, parent2, inner_elitism = True):
    
    child = Individual(self.l, self.w)
    array1 = normalize_channels(np.array(parent1.image))
    array2 = normalize_channels(np.array(parent2.image))
    
    channels = []
    for i in range(3):  
        if np.random.rand() > 0.5:
            channels.append(array1[:, :, i])
        else:
            channels.append(array2[:, :, i])
    
    new_array = np.stack(channels, axis=-1)
    
    child.image = Image.fromarray(new_array.astype('uint8'))
    child.array = new_array
    child.get_fitness(self.target_image)
    
    if inner_elitism :

        if child.fitness <= min(parent1.fitness, parent2.fitness):
            return child
        
        return None
    else:
        return child

def normalize_channels(array):

    min_val = array.min(axis=(0, 1), keepdims=True)
    max_val = array.max(axis=(0, 1), keepdims=True)
    return (array - min_val) / (max_val - min_val) * 255

def two_point_xo(self, parent1, parent2, inner_elitism = True):

    size = parent1.array.size
    point1 = np.random.randint(0, size)
    point2 = np.random.randint(0, size)

    if point1 > point2:
        point1, point2 = point2, point1

    parent1_flattened = parent1.array.flatten()
    parent2_flattened = parent2.array.flatten()

    child_flattened = np.concatenate((parent1_flattened[:point1], parent2_flattened[point1:point2], parent1_flattened[point2:]))
    child_array = child_flattened.reshape(parent1.array.shape)

    child = Individual(parent1.l, parent1.w)
    child.array = child_array
    child.image = Image.fromarray(np.uint8(child_array))

    child.get_fitness(self.target_image)

    if inner_elitism :

        if child.fitness <= min(parent1.fitness, parent2.fitness):
            return child
        
        return None
    else:
        return child

def patch_xo(self, parent1, parent2, inner_elitism = True):

    l, w, _ = parent1.array.shape
    patch_x = np.random.randint(0, l)
    patch_y = np.random.randint(0, w)
    patch_l = np.random.randint(0, l - patch_x)
    patch_w = np.random.randint(0, w - patch_y)

    child_array = np.copy(parent1.array)
    child_array[patch_x:patch_x+patch_l, patch_y:patch_y+patch_w] = parent2.array[patch_x:patch_x+patch_l, patch_y:patch_y+patch_w]

    child = Individual(parent1.l, parent1.w)
    child.array = child_array
    child.image = Image.fromarray(np.uint8(child_array))

    child.get_fitness(self.target_image) 
   
    if inner_elitism :

        if child.fitness <= min(parent1.fitness, parent2.fitness):
            return child
        
        return None
    else:
        return child





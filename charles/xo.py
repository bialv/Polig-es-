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


def blend_xo(self, parent1, parent2, inner_elitism=True):
    
    child = Individual(self.l, self.w) # initialize a new individual to be the child
    
    blend_alpha = random.random() # get a random blend alpha value between 0 and 1
    
    child_image = Image.blend(parent1.image, parent2.image, blend_alpha) # Blend the images of parent1 and parent2 using the blend alpha value
   
    child.image = child_image  # Set the child's image attribute to the blended image
    
    child.array = np.array(child_image) # Convert the image to an array
    
    child.get_fitness(self.target_image) # Calculate the fitness of the child based on the target image
    
    
    if inner_elitism: # If inner elitism is enabled
        # Check if the child's fitness is less than or equal to the minimum fitness of the parents
        if child.fitness <= min(parent1.fitness, parent2.fitness):
            return child # If so, return the child
        # If not, return None (indicating that the child should not be included in the population)
        return None
    else: # If inner elitism is not enabled, simply return the child
        return child


def per_channel_crossover(self, parent1, parent2, inner_elitism = True):

    child = Individual(self.l, self.w) # initialize a new individual to be the child
    # we need to convert parent images to numpy arrays
    array1 = np.array(parent1.image)
    array2 = np.array(parent2.image)
    
    channels = [] # create a list to store the channels of the new image
    # Loop through each channel (RGB) of the images
    # select a random channel from parent1 or parent2 based on a 'coin flip'
    for i in range(3):  
        if np.random.rand() > 0.5:
            channels.append(array1[:, :, i])
        else:
            channels.append(array2[:, :, i])
    
    new_array = np.stack(channels, axis=-1) # Stack the selected channels to create the new image array
    # with the new array stacked we can convert it back to an image
    child.image = Image.fromarray(new_array.astype('uint8'))
    # update the array attribute of the child
    child.array = new_array
    # get the fitness of the child based on its similarity to the target image
    child.get_fitness(self.target_image)
    
    if inner_elitism: # If inner elitism is enabled
        # Check if the child's fitness is less than or equal to the minimum fitness of the parents
        if child.fitness <= min(parent1.fitness, parent2.fitness):
            return child # If so, return the child
        # If not, return None (indicating that the child should not be included in the population)
        return None
    else: # If inner elitism is not enabled, simply return the child
        return child


def per_channel_crossover_2(self, parent1, parent2, inner_elitism = True):
    
    child = Individual(self.l, self.w)# initialize a new individual to be the child
    # we need to convert parent images to numpy arrays, this time we normalize the channels
    array1 = normalize_channels(np.array(parent1.image))
    array2 = normalize_channels(np.array(parent2.image))
    
    channels = [] # create a list to store the channels of the new image
    # Loop through each channel (RGB) of the images
    # select a random channel from parent1 or parent2 based on a 'coin flip'
    for i in range(3):  
        if np.random.rand() > 0.5:
            channels.append(array1[:, :, i])
        else:
            channels.append(array2[:, :, i])
    
    new_array = np.stack(channels, axis=-1) # Stack the selected channels to create the new image array
    # with the new array stacked we can convert it back to an image
    child.image = Image.fromarray(new_array.astype('uint8'))
    # update the array attribute of the child
    child.array = new_array
    # get the fitness of the child based on its similarity to the target image
    child.get_fitness(self.target_image)
    
    if inner_elitism: # If inner elitism is enabled
        # Check if the child's fitness is less than or equal to the minimum fitness of the parents
        if child.fitness <= min(parent1.fitness, parent2.fitness):
            return child # If so, return the child
        # If not, return None (indicating that the child should not be included in the population)
        return None
    else: # If inner elitism is not enabled, simply return the child
        return child

def normalize_channels(array):
    # function to normalize RBG channels
    min_val = array.min(axis=(0, 1), keepdims=True)
    max_val = array.max(axis=(0, 1), keepdims=True)
    return (array - min_val) / (max_val - min_val) * 255

def two_point_xo(self, parent1, parent2, inner_elitism = True):
    # get the number of elements of the array
    size = parent1.array.size
    # Generate two random points within the size of the array
    point1 = np.random.randint(0, size)
    point2 = np.random.randint(0, size)

    # Ensure point1 is less than point2
    if point1 > point2:
        point1, point2 = point2, point1

    # Flatten the arrays of the parent images
    parent1_flattened = parent1.array.flatten()
    parent2_flattened = parent2.array.flatten()

    # Perform two-point crossover by swapping segments between two points
    child_flattened = np.concatenate((parent1_flattened[:point1], parent2_flattened[point1:point2], parent1_flattened[point2:]))
    # Reshape the flattened array back to the original array shape
    child_array = child_flattened.reshape(parent1.array.shape)

    child = Individual(parent1.l, parent1.w) # initialize a new individual to be the child
    child.array = child_array # update the array attribute of the child
    child.image = Image.fromarray(np.uint8(child_array)) # convert it back to an image
    # get the fitness of the child based on its similarity to the target image
    child.get_fitness(self.target_image)

    if inner_elitism: # If inner elitism is enabled
        # Check if the child's fitness is less than or equal to the minimum fitness of the parents
        if child.fitness <= min(parent1.fitness, parent2.fitness):
            return child # If so, return the child
        # If not, return None (indicating that the child should not be included in the population)
        return None
    else: # If inner elitism is not enabled, simply return the child
        return child

def patch_xo(self, parent1, parent2, inner_elitism = True):
    # Get the dimensions of the image arrays
    l, w, _ = parent1.array.shape
    # Generate random coordinates and dimensions for the patch
    patch_x = np.random.randint(0, l)
    patch_y = np.random.randint(0, w)
    patch_l = np.random.randint(0, l - patch_x)
    patch_w = np.random.randint(0, w - patch_y)

    # Create a child array by copying the parent1 array
    child_array = np.copy(parent1.array)
    # Replace the patch region we created with the corresponding region from parent2 array since we used parent1 to be the base
    child_array[patch_x:patch_x+patch_l, patch_y:patch_y+patch_w] = parent2.array[patch_x:patch_x+patch_l, patch_y:patch_y+patch_w]

    child = Individual(parent1.l, parent1.w) # initialize a new individual to be the child
    child.array = child_array # update the array attribute of the child
    child.image = Image.fromarray(np.uint8(child_array)) # convert it back to an image
    # get the fitness of the child based on its similarity to the target image
    child.get_fitness(self.target_image)
   
    if inner_elitism: # If inner elitism is enabled
        # Check if the child's fitness is less than or equal to the minimum fitness of the parents
        if child.fitness <= min(parent1.fitness, parent2.fitness):
            return child # If so, return the child
        # If not, return None (indicating that the child should not be included in the population)
        return None
    else: # If inner elitism is not enabled, simply return the child
        return child





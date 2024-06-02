from operator import attrgetter
import random
from matplotlib import pyplot as plt
from charles.charles import Population, Individual
from copy import copy
import numpy as np
from charles.selection import fps
from charles.mutation import random_mutate
from charles.xo import blend_xo
import colour
from PIL import Image
from skimage.metrics import structural_similarity as ssim


# Delta E Fitness
def get_fitness(self, target):
        self.fitness = np.mean(colour.difference.delta_e.delta_E_CIE1976(target, self.array))

# Sum of Absolute Differences Fitness 
def get_fitness_2(self, target_image):
        
        target_image_array = np.array(target_image)
        if self.array.shape != target_image_array.shape:
            raise ValueError("Target and individual don't have the same dim")
        abs_diff = np.abs(self.array - target_image_array)
        self.fitness = np.sum(abs_diff)

#Euclidean Color Distance Fitness 
def get_fitness_3(self, target_image):
        target_image_array = np.array(target_image)
        
        if self.array.shape != target_image_array.shape:
            raise ValueError("Target and individual don't have the same dim")
        
        diff = self.array.astype(np.float32) - target_image_array.astype(np.float32)
        squared_diff = np.square(diff)
        sum_squared_diff = np.sum(squared_diff, axis=-1)
        distances = np.sqrt(sum_squared_diff)
        self.fitness = np.sum(distances)
        
        return self.fitness

# Root Mean Square Error Fitness
def get_fitness_4(self, target_image):
        
        if isinstance(target_image, Image.Image):
            target_image_array = np.array(target_image)

        else:
            target_image_array = target_image
        
        if self.array.shape != target_image_array.shape:
            raise ValueError("Target and individual don't have the same dim")
        mse = np.mean((self.array - target_image_array) ** 2)
        self.fitness = np.sqrt(mse)


# Monkey patching
Individual.get_fitness = get_fitness
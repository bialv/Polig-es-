from operator import attrgetter
import random
from matplotlib import pyplot as plt
from charles.charles import Population, Individual
from copy import copy
import numpy as np
import charles.selection
import charles.xo
from charles.selection import fps, roulette_wheel_selection, rank_selection, rank_tournament_selection
from charles.mutation import random_mutate, scramble_patch_mutation, swap_polygons_mutation_color, swap_polygons_mutation, interpolate_color_mutation
from charles.xo import blend_xo, per_channel_crossover, per_channel_crossover_2, two_point_xo, patch_xo
import colour
from charles import fitness
from PIL import Image, ImageOps, ImageDraw, ImagePath


path = r"data\Girl_White_Pearl.jpg"

original_image = Image.open(path).convert('RGB')

target_image = original_image.resize((180,257)).convert('RGB')

# init P with N indvs
P = Population(target_image = target_image, size=100, optim="min", repetition=True)

P.evolve(runs = 1, gens=1500, xo_prob=0.8, mut_prob=0.15, select=roulette_wheel_selection,
         xo=per_channel_crossover, mutate=random_mutate, elitism=True, inner_elitism = True)


 


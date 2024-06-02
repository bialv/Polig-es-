from random import randint
import random
from .charles import Individual
import copy
import numpy as np
from PIL import Image, ImageDraw, ImageStat, ImageFilter

def random_mutate(self, ind):

        iterations = random.randint(1, 3)
        region = random.randint(1,(self.l + self.w)//10)
        img = ind.image

        for i in range(iterations):
            num_points = random.randint(3, 6)
            region_x = random.randint(0, self.l)
            region_y = random.randint(0, self.w)

            xy = []
            for j in range(num_points):
                xy.append((random.randint(region_x - region, region_x + region),
                           random.randint(region_y - region, region_y + region)))

            img1 = ImageDraw.Draw(img)
            img1.polygon(xy, fill=ind.rand_color())

        child = Individual(ind.l, ind.w)
        child.image = img
        child.array = child.to_array(child.image)
        child.get_fitness(self.target_image)

        return child


def scramble_patch_mutation(self, individual):

    l, w, _ = individual.array.shape
    patch_x = np.random.randint(0, l)
    patch_y = np.random.randint(0, w)
    patch_l = np.random.randint((l-patch_x) / 2, l - patch_x)
    patch_w = np.random.randint((w-patch_y) / 2, w - patch_y)

    mutated_array = np.copy(individual.array)

    patch = mutated_array[patch_x:patch_x+patch_l, patch_y:patch_y+patch_w]
    np.random.shuffle(patch.flat)
    mutated_array[patch_x:patch_x+patch_l, patch_y:patch_y+patch_w] = patch

    mutated = Individual(individual.l, individual.w)
    mutated.array = mutated_array
    mutated.image = Image.fromarray(np.uint8(mutated_array))

    mutated.get_fitness(self.target_image)  

    return mutated

def swap_polygons_mutation(self, ind):

    swaps=3
    img = ind.image.copy()  

    for _ in range(swaps):
        polygon_size = (ind.l // 10, ind.w // 10)  

        x1, y1 = random.randint(0, ind.l - polygon_size[0]), random.randint(0, ind.w - polygon_size[1])
        x2, y2 = random.randint(0, ind.l - polygon_size[0]), random.randint(0, ind.w - polygon_size[1])

        region1 = img.crop((x1, y1, x1 + polygon_size[0], y1 + polygon_size[1]))
        region2 = img.crop((x2, y2, x2 + polygon_size[0], y2 + polygon_size[1]))

        img.paste(region2, (x1, y1))
        img.paste(region1, (x2, y2))

    child = Individual(ind.l, ind.w)
    child.image = img
    child.array = child.to_array(child.image)
    child.get_fitness(self.target_image)

    return child

def swap_polygons_mutation_color(self, ind, swaps=3, color_threshold=40):

    img = ind.image.copy()  

    def get_average_color(region):
        stat = ImageStat.Stat(region)
        return tuple(stat.mean)

    def color_difference(color1, color2):
        return sum((a - b) ** 2 for a, b in zip(color1, color2)) ** 0.5

    for _ in range(swaps):
        polygon_size = (ind.l // 10, ind.w // 10) 

        x1, y1 = random.randint(0, ind.l - polygon_size[0]), random.randint(0, ind.w - polygon_size[1])
        x2, y2 = random.randint(0, ind.l - polygon_size[0]), random.randint(0, ind.w - polygon_size[1])

        region1 = img.crop((x1, y1, x1 + polygon_size[0], y1 + polygon_size[1]))
        region2 = img.crop((x2, y2, x2 + polygon_size[0], y2 + polygon_size[1]))

        color1 = get_average_color(region1)
        color2 = get_average_color(region2)

        if color_difference(color1, color2) > color_threshold:
            img.paste(region2, (x1, y1))
            img.paste(region1, (x2, y2))

    child = Individual(ind.l, ind.w)
    child.image = img
    child.array = child.to_array(child.image)
    child.get_fitness(self.target_image)

    return child

def interpolate_color_mutation(self, ind):
    img = ind.image.copy()
    draw = ImageDraw.Draw(img)
    polygon_size = (ind.l // 10, ind.w // 10)
    blend_factor = 0.5 

    for _ in range(random.randint(1, 3)): 
        x, y = random.randint(0, ind.l - polygon_size[0]), random.randint(0, ind.w - polygon_size[1])
        
        num_points = random.randint(3, 6)
        xy = [(random.randint(x, x + polygon_size[0]), random.randint(y, y + polygon_size[1])) for _ in range(num_points)]
        
        region = img.crop((x, y, x + polygon_size[0], y + polygon_size[1]))
        stat = ImageStat.Stat(region)
        current_color = tuple(stat.mean)  

        new_color = tuple(random.randint(0, 255) for _ in range(3))

        interpolated_color = tuple(int(current_color[i] * (1 - blend_factor) + new_color[i] * blend_factor) for i in range(3))

        draw.polygon(xy, fill=interpolated_color)

    child = Individual(ind.l, ind.w)
    child.image = img
    child.array = child.to_array(child.image)
    child.get_fitness(self.target_image)

    return child







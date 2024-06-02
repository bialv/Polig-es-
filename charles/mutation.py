from random import randint
import random
from .charles import Individual
import copy
import numpy as np
from PIL import Image, ImageDraw, ImageStat, ImageFilter

# Adds random polygons with random colors
def random_mutate(self, ind):
    iterations = random.randint(1, 3)  # Number of polygons to add
    region = random.randint(1, (self.l + self.w) // 10)  # Control polygon size
    img = ind.image

    for i in range(iterations):
        num_points = random.randint(3, 6)  # Number of points in the polygon
        region_x = random.randint(0, self.l)
        region_y = random.randint(0, self.w)

        xy = []
        for j in range(num_points):
            # Generate vertices of the polygon
            xy.append((random.randint(region_x - region, region_x + region),
                       random.randint(region_y - region, region_y + region)))

        img1 = ImageDraw.Draw(img)  # Draw on the image
        img1.polygon(xy, fill=ind.rand_color())  # Draw a polygon with a random color

    # Create Child from the mutated image
    child = Individual(ind.l, ind.w)
    child.image = img
    child.array = child.to_array(child.image)
    child.get_fitness(self.target_image)

    return child

# Scramble pixels in a selected patch of the image
def scramble_patch_mutation(self, individual):
    l, w, _ = individual.array.shape
    # Select patch size    
    patch_x = np.random.randint(0, l)
    patch_y = np.random.randint(0, w)
    patch_l = np.random.randint((l - patch_x) / 2, l - patch_x)
    patch_w = np.random.randint((w - patch_y) / 2, w - patch_y)

    mutated_array = np.copy(individual.array)

    # Extract a patch and scramble its pixels
    patch = mutated_array[patch_x:patch_x+patch_l, patch_y:patch_y+patch_w]
    np.random.shuffle(patch.flat)  # Flatten and shuffle the patch
    mutated_array[patch_x:patch_x+patch_l, patch_y:patch_y+patch_w] = patch

    # Create a Child from the scrambled array
    mutated = Individual(individual.l, individual.w)
    mutated.array = mutated_array
    mutated.image = Image.fromarray(np.uint8(mutated_array))
    mutated.get_fitness(self.target_image)

    return mutated

# Swap different areas of the same image
def swap_polygons_mutation(self, ind):
    swaps = 3  # Number of swaps
    img = ind.image.copy()

    for _ in range(swaps):
        polygon_size = (ind.l // 10, ind.w // 10)  # Size of the polygon to swap

        # Randomly select two regions to swap
        x1, y1 = random.randint(0, ind.l - polygon_size[0]), random.randint(0, ind.w - polygon_size[1])
        x2, y2 = random.randint(0, ind.l - polygon_size[0]), random.randint(0, ind.w - polygon_size[1])

        # Crop the selected regions
        region1 = img.crop((x1, y1, x1 + polygon_size[0], y1 + polygon_size[1]))
        region2 = img.crop((x2, y2, x2 + polygon_size[0], y2 + polygon_size[1]))

        # Swap the regions
        img.paste(region2, (x1, y1))
        img.paste(region1, (x2, y2))

    # Create a Child from the mutated image
    child = Individual(ind.l, ind.w)
    child.image = img
    child.array = child.to_array(child.image)
    child.get_fitness(self.target_image)

    return child

# Sswap areas based on color difference
def swap_polygons_mutation_color(self, ind, swaps=3, color_threshold=40):
    img = ind.image.copy()

    def get_average_color(region):
        # Get the average color of a region
        stat = ImageStat.Stat(region)
        return tuple(stat.mean)

    def color_difference(color1, color2):
        # Calculate color difference between two colors
        return sum((a - b) ** 2 for a, b in zip(color1, color2)) ** 0.5

    for _ in range(swaps):
        polygon_size = (ind.l // 10, ind.w // 10)  # Size of the polygon to swap

        # Randomly select two regions to swap
        x1, y1 = random.randint(0, ind.l - polygon_size[0]), random.randint(0, ind.w - polygon_size[1])
        x2, y2 = random.randint(0, ind.l - polygon_size[0]), random.randint(0, ind.w - polygon_size[1])

        # Crop the selected regions
        region1 = img.crop((x1, y1, x1 + polygon_size[0], y1 + polygon_size[1]))
        region2 = img.crop((x2, y2, x2 + polygon_size[0], y2 + polygon_size[1]))

        # Get average colors of the regions
        color1 = get_average_color(region1)
        color2 = get_average_color(region2)

        # Swap regions if their color difference is above the threshold
        if color_difference(color1, color2) > color_threshold:
            img.paste(region2, (x1, y1))
            img.paste(region1, (x2, y2))

    # Create a Child from the mutated image
    child = Individual(ind.l, ind.w)
    child.image = img
    child.array = child.to_array(child.image)
    child.get_fitness(self.target_image)

    return child

# Change color of randomly selected polygons by interpolating the existing colors with a random new one
def interpolate_color_mutation(self, ind):
    img = ind.image.copy()
    draw = ImageDraw.Draw(img) # Draw on the image
    polygon_size = (ind.l // 10, ind.w // 10)  # Size of the polygon
    blend_factor = 0.5  # Blend factor for color interpolation

    for _ in range(random.randint(1, 3)):  # Number of mutation iterations
        x, y = random.randint(0, ind.l - polygon_size[0]), random.randint(0, ind.w - polygon_size[1]) # Select random position

        num_points = random.randint(3, 6)  # Number of points in the polygon
        xy = [(random.randint(x, x + polygon_size[0]), random.randint(y, y + polygon_size[1])) for _ in range(num_points)] # Get polygon vertices

        # Crop the region and get its average color
        region = img.crop((x, y, x + polygon_size[0], y + polygon_size[1]))
        stat = ImageStat.Stat(region)
        current_color = tuple(stat.mean)

        # Generate a new random color
        new_color = tuple(random.randint(0, 255) for _ in range(3))

        # Interpolate between the current color and the new color
        interpolated_color = tuple(int(current_color[i] * (1 - blend_factor) + new_color[i] * blend_factor) for i in range(3))

        # Draw a polygon with the interpolated color
        draw.polygon(xy, fill=interpolated_color)

    # Create a Child from the mutated image
    child = Individual(ind.l, ind.w)
    child.image = img
    child.array = child.to_array(child.image)
    child.get_fitness(self.target_image)

    return child

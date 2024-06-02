from random import shuffle, choice, sample
import random
from PIL import Image, ImageOps, ImageDraw, ImagePath
import numpy as np
from operator import attrgetter
from copy import copy
import pandas as pd
import random


class Individual:
#first we initialize 
    def __init__(self, l, w):
        self.l = l
        self.w = w
        self.fitness = float('inf')
        self.array = None
        self.image = None
        # our representation is done with create random image function
        self.representation = self.create_random_image_array()
        # fitness will be assigned to each individual
        self.fitness = float('inf')


    def get_fitness(self, target_image):
        raise Exception("You need to monkey patch the fitness function.")

    def get_neighbours(self):
        raise Exception("You need to monkey patch the neighbourhood function.")
    
    def rand_color(self):
        return f'#{random.randint(0, 0xFFFFFF):06X}'
    
    def create_random_image_array(self):

        iterations = random.randint(3, 6)  # Random number of polygons
        region = (self.l + self.w) // 8    # Region size 
        img = Image.new("RGB", (self.l, self.w), self.rand_color())  

        for _ in range(iterations):
            num_points = random.randint(3, 6)  # Random number of points per polygon
            region_x = random.randint(0, self.l)
            region_y = random.randint(0, self.w)
            # we do a list comprehension to generate polygon vertices
            xy = [(random.randint(region_x - region, region_x + region),
                random.randint(region_y - region, region_y + region)) for _ in range(num_points)]
            # with polygons done we draw them in the image we created above
            ImageDraw.Draw(img).polygon(xy, fill=self.rand_color())

        self.image = img
        self.array = np.array(img)  

        return self.image
    
    def __len__(self):
        return len(self.representation)

    def __getitem__(self, position):
        return self.representation[position]

    def __setitem__(self, position, value):
        self.representation[position] = value

    def __repr__(self):
        return f" Fitness: {self.fitness}"
    
    def to_array(self, image):
        return np.array(image)


class Population:
    #first we initialize
    def __init__(self, size, optim, target_image,**kwargs):

        self.target_image = target_image

        self.l, self.w = self.target_image.size
        
        self.target_image_array = np.array(self.target_image)

        self.size = size

        self.optim = optim

        self.individuals = []
        # to populate we append Indivituals until we reach the size defined
        for i in range(size):
            new_indiv = Individual(self.l, self.w)

            new_indiv.get_fitness(self.target_image)

            self.individuals.append(new_indiv)

    def to_array(self, image):
        return np.array(image)
    
    def evolve(self, runs, gens, xo_prob, mut_prob, select, xo, mutate, elitism, inner_elitism):
        # First step is to do a dictionary that will store all fitness values for each run and generation,
        # we will safe this after transformations to keep track of changes different runs
        all_fitness_values = {'run': [], 'generation': [], 'fitness': [], 'mutation_method': [], 'xo_method': [], 'selection_method': []}
        # we loop through each run 
        for run in range(runs):
            # since we do runs we need to create individuals for every population of that run
            self.individuals = [Individual(self.l, self.w) for _ in range(self.size)]
            for individual in self.individuals:
                individual.get_fitness(self.target_image)
            fitness_values = []
            
            # also loop through generations
            for i in range(gens):
                new_pop = []
                # we want to keep the best individual to carry over to the next generation, replacement elitism
                if elitism:
                    if self.optim == "max":
                        elite = copy(max(self.individuals, key=attrgetter('fitness')))
                    elif self.optim == "min":
                        elite = copy(min(self.individuals, key=attrgetter('fitness')))

                # perform xo and mut while we don't have the desirable number on the generation we currently are on
                while len(new_pop) < self.size:
                    # select the parents using the selection method we pass when calling this function
                    parent1, parent2 = select(self), select(self)
                    # randomly decide whether or not to perform crossover and xo
                    rand = random.uniform(0, 1)
                    # check probabilite against the random value
                    if rand < xo_prob:
                        # do the crossover
                        offspring1 = xo(self, parent1, parent2)
                        if (inner_elitism):
                            while offspring1 == None:
                                parent1 = select(self)
                                parent2 = select(self)
                                offspring1 = xo(self, parent1, parent2)
                        else:   
                            parent1 = select(self)
                            parent2 = select(self)
                            offspring1 = xo(self, parent1, parent2, inner_elitism = inner_elitism)
                        
                    else:
                        # we choose the parent with best fitness as offspring if no crossover is performed
                        offspring1 = parent1 if parent1.fitness <= parent2.fitness else parent2

                    # check probabilite against the random value
                    if rand < mut_prob:
                        # do the mutation
                        offspring1 = mutate(self, offspring1)
                    new_pop.append(offspring1)

                # in the end we replace the worst individual with the elite individual, replacement elitism
                if elitism:
                    if self.optim == "max":
                        worst = min(new_pop, key=attrgetter('fitness'))
                        if elite.fitness > worst.fitness:
                            new_pop.pop(new_pop.index(worst))
                            new_pop.append(elite)
                    if self.optim == "min":
                        worst = max(new_pop, key=attrgetter('fitness'))
                        if elite.fitness < worst.fitness:
                            new_pop.pop(new_pop.index(worst))
                            new_pop.append(elite)

                # upadate the population with the one we just created
                self.individuals = new_pop
                # sort the fitness values
                self.individuals.sort(key=lambda ind: ind.fitness)
                
                # the first of the list is our best individual
                fittest = self.individuals[0]
                best_fitness = fittest.fitness

                # we store fitness values for analysis
                all_fitness_values['run'].append(run)
                all_fitness_values['generation'].append(i)
                all_fitness_values['fitness'].append(best_fitness)
                all_fitness_values['mutation_method'].append(mutate.__name__)
                all_fitness_values['xo_method'].append(xo.__name__)
                all_fitness_values['selection_method'].append(select.__name__)

                # we print the current generation's best fitness and population size to keep track of what's happening when we run the code
                print(f"Best individual of gen #{i + 1}: {best_fitness}")
                print(f"Population of run {run} has: {len(new_pop)}")

                # we save the image of the fittest individual every 100th generations
                if i % 100 == 0 or i == gens - 1:
                    # this always needs to change to the path where we want to save your images
                    fittest.image.save(r"C:\Users\afspf\Documents\FAC\2nd semester\CIFO\Project" + str(i)+".png")

                    print("Most fit individual in epoch " + str(i) +
                        " has fitness: " + str(best_fitness))

        # this saves the fitness values in csv to futher analysis
        df = pd.DataFrame(all_fitness_values)
        filename = f"xo_prob_{xo_prob}_mut_prob_{mut_prob}_fitness_values.csv"
        df.to_csv(filename, index=False)

                   

    def __len__(self):
        return len(self.individuals)

    def __getitem__(self, position):
        return self.individuals[position]
    
    def to_array(self, image):
        return np.array(image)
    
       

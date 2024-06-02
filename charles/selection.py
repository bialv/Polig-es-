from random import uniform
import numpy as np
import random

def fps(population):

    #print("Entrei na selection")
    if population.optim == "min":
        #print("Entrei na selection max")
        tournament_size=6
        # randomly sample participants
        indices = np.random.choice(len(population), tournament_size)
        random_subset = [population[i] for i in indices]

        winner = None

        # find individual with best fitness 
        for i in random_subset:
            if (winner == None):
                winner = i
            elif i.fitness < winner.fitness:
                winner = i

        return winner
    
    # tirar esta merda
    elif population.optim == "max":
        raise NotImplementedError
    else:
        raise Exception(f"Optimization not specified (max/min)")


def roulette_wheel_selection(population):
    # Inverting fitness for minimization problem: lower fitness, higher selection probability
    total_fitness = sum(1.0 / individual.fitness for individual in population)
    pick = random.uniform(0, total_fitness)
    current = 0
    for individual in population:
        current += 1.0 / individual.fitness
        if current > pick:
            return individual
        
def rank_selection(population):
    sorted_population = sorted(population, key=lambda x: x.fitness, reverse=True)

    # Number of individuals
    num_individuals = len(sorted_population)
    
    # Calculate the sum of ranks (using the formula for the sum of the first N natural numbers)
    sum_of_ranks = num_individuals * (num_individuals + 1) / 2

    # Assign selection probabilities based on ranks
    probabilities = [(i + 1) / sum_of_ranks for i in range(num_individuals)]

    # Select one individual based on the probabilities
    chosen_index = np.random.choice(num_individuals, p=probabilities)
    return sorted_population[chosen_index]
        

        
def rank_tournament_selection(population):
    # Sort population based on fitness
    tournament_size=3
    
    sorted_population = sorted(population, key=lambda x: x.fitness)
    
    # Conduct a tournament with a specified size
    tournament_contestants = random.sample(sorted_population, tournament_size)
    
    # Sort contestants by rank in the overall population
    tournament_sorted = sorted(tournament_contestants, key=lambda x: sorted_population.index(x))
    
    # The contestant with the highest rank (lowest index after sorting) wins
    return tournament_sorted[0]


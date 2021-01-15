from tsp_utils import data_reader, data_generator
import numpy as np
from GeneticSolver.utils import *
from matplotlib import pyplot as plt
from collections import Counter

import os

num_city, instance_name = 30, 1
population_size = 2000
generation_size = 2000
mutation_rate = 0.01
low_prob, high_prob = 0.1, 0.9
num_threads = 4
coordinates, dist_mat = data_reader(num_city, instance_name)


def ga_solve(num_city, dist_mat, population_size, generation_size, mutation_rate, low_prob, high_prob):
    """
    main function
    :param num_city:
    :param dist_mat:
    :param population_size:
    :param generation_size:
    :param mutation_rate:
    :param low_prob:
    :param high_prob:
    :return: best route_length in each generation, best individual in each generation, final solution score, final route
    """
    population = generate_initial_solution(num_city, population_size)  # generate initial solution
    evaluation = evaluate_solution(population, dist_mat)  # evaluate initial solution
    best_individual = np.zeros(generation_size)  # best individual score in each generation
    best_solution_pool = np.zeros((generation_size, num_city), dtype=int)  # best solution in each generation
    global_best_score = np.inf  # the global best score
    global_best_route = np.zeros(num_city, dtype=int)  # the global best route
    for i in range(generation_size):
        # fitness = 1 / evaluation
        # probability = roulette_selection(fitness)
        probability = rank_selection(evaluation, low_prob, high_prob)  # rank based selection
        father_list, mother_list = mating(probability)
        population, evaluation = reproduce(father_list, mother_list, population, evaluation, dist_mat, mutation_rate)
        population, evaluation = mutation(population, evaluation, dist_mat, mutation_rate)
        # next_generation = reproduce(father_list, mother_list, population, mutation_rate)
        # next_generation_eval = evaluate_solution(next_generation, dist_mat)
        # mutate parents
        population, evaluation = natural_selection(population, evaluation, population_size)
        best_index = np.argmin(evaluation)
        current_best_score = evaluation[best_index]
        current_best_route = population[best_index]
        best_individual[i] = current_best_score
        best_solution_pool[i] = current_best_route
        if current_best_score < global_best_score:
            global_best_score = current_best_score
            global_best_route = current_best_route
    return best_individual, best_solution_pool, global_best_score, global_best_route


best_individual, best_solution_pool, global_best_score, global_best_route = ga_solve(num_city, dist_mat, population_size, generation_size, mutation_rate, low_prob, high_prob)
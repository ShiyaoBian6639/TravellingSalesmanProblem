from tsp_utils import data_reader, data_generator
import numpy as np
from GeneticSolver.utils import ga_solve, multi_start_ga
from matplotlib import pyplot as plt

num_city, instance_name = 30, 1
population_size = 500
generation_size = 1500
mutation_rate = 0.02
low_prob, high_prob = 0.1, 0.9
num_threads = 12
coordinates, dist_mat = data_reader(num_city, instance_name)

score, route, best_sol, best_score = ga_solve(num_city, dist_mat, population_size, generation_size, mutation_rate,
                                              low_prob, high_prob)

plt.plot(score)
plt.show()
score, route = multi_start_ga(num_city, dist_mat, population_size, generation_size, mutation_rate, low_prob, high_prob,
                              num_threads)

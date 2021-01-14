from tsp_utils import data_reader, data_generator
import numpy as np
from GeneticSolver.utils import ga_solve, multi_start_ga
from matplotlib import pyplot as plt
from collections import Counter

num_city, instance_name = 30, 1
population_size = 200
generation_size = 50000
mutation_rate = 1
low_prob, high_prob = 0.1, 0.9
num_threads = 2
coordinates, dist_mat = data_reader(num_city, instance_name)
aug_dist_mat = dist_mat.copy()
np.fill_diagonal(aug_dist_mat, 0)
score, route, best_sol, best_score = ga_solve(num_city, dist_mat, population_size, generation_size, mutation_rate,
                                              low_prob, high_prob)

plt.plot(score)
plt.show()
score, route = multi_start_ga(num_city, dist_mat, population_size, generation_size, mutation_rate, low_prob, high_prob,
                              num_threads)

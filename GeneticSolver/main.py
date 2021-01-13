from tsp_utils import data_reader
import numpy as np
from GeneticSolver.utils import ga_solve
from matplotlib import pyplot as plt

num_city, instance_name = 30, 1
population_size = 10000
generation_size = 1000
mutation_rate = 0.01
low_prob, high_prob = 0.1, 0.9
coordinates, dist_mat = data_reader(num_city, instance_name)

score, route, best_sol,  best_score = ga_solve(num_city, dist_mat, population_size, generation_size, mutation_rate, low_prob, high_prob)

plt.plot(score)
plt.show()

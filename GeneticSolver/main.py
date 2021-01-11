from tsp_utils import data_reader
import numpy as np
from GeneticSolver.utils import generate_initial_solution, evaluate_solution, route_length, rank_selection
from GeneticSolver.utils import crossover, check_individual_feasibility, mutate
from matplotlib import pyplot as plt

num_city, instance_name = 30, 1
population_size = 100
coordinates, dist_mat = data_reader(num_city, instance_name)
initial_solution = generate_initial_solution(num_city, population_size)
route_length(dist_mat, initial_solution[0])
evaluation = evaluate_solution(initial_solution, dist_mat)
probability = rank_selection(evaluation, 0.9, 0.1)

# do crossover
child = crossover(initial_solution[0], initial_solution[1], 4, 8)
print(initial_solution[0])
print(initial_solution[1])
print(child)
check_individual_feasibility(child)

mutate(child, 1)
print(child)

# check result
plt.plot(evaluation, probability, '.')
plt.show()
print(min(probability), max(probability))

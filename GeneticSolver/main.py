from tsp_utils import data_reader
import numpy as np
from GeneticSolver.utils import ga_solve
from matplotlib import pyplot as plt

num_city, instance_name = 30, 1
population_size = 10000
generation_size = 10000
coordinates, dist_mat = data_reader(num_city, instance_name)

sol = ga_solve(num_city, dist_mat, population_size, generation_size)

plt.plot(sol)
plt.show()

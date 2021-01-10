import numpy as np
from numba import njit, int32


class GeneticAlgorithm:
    def __init__(self, dist_mat, population_size):
        self.dist_mat = dist_mat
        self.num_city = dist_mat.shape[0]
        self.population_size = population_size
        self.initial_solution = self.generate_initial_solution()

    def generate_initial_solution(self):
        return generate_initial_solution(self.num_city, self.population_size)


@njit()
def route_length(dist_mat, route):
    total = 0
    for i in range(len(route) - 1):
        total += dist_mat[route[i], route[i + 1]]
    total += dist_mat[route[-1], route[0]]
    return total


@njit()
def generate_initial_solution(n, population_size):
    """
    :param n: number of cities
    :param population_size: number of individuals in the initial solution
    :return: initial_solution (n * population_size)
    """
    initial_solution = np.zeros(shape=(population_size, n), dtype=int32)
    for i in range(population_size):
        individual = np.random.permutation(n)  # randomly generate an individual
        for j in range(n):
            initial_solution[i, j] = individual[j]
    return initial_solution


@njit()
def evaluate_solution(solution, dist_mat):
    population_size = solution.shape[0]
    evaluation = np.zeros(population_size)
    for i in range(population_size):
        evaluation[i] = route_length(dist_mat, solution[i])
    return evaluation


@njit()
def rank_selection(evaluation, best_prob, worst_prob):
    population_size = evaluation.shape[0]
    probability = np.zeros(evaluation.shape)
    rank = np.argsort(-evaluation)
    for i in range(population_size):
        probability[rank[i]] = worst_prob + (best_prob - worst_prob) * i / (population_size - 1)
    return probability


@njit()
def roulette_selection():
    pass


@njit()
def tournament_selection():
    pass


@njit()
def check_individual_feasibility(individual):
    return len(np.unique(individual)) == len(individual)


@njit()
def check_population_feasibility():
    pass

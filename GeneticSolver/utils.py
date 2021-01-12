import numpy as np
from numba import njit, int32, prange


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
    population_size, num_city = solution.shape
    evaluation = np.zeros(population_size)
    for i in prange(population_size):
        total = 0
        for j in range(num_city - 1):
            total += dist_mat[solution[i, j], solution[i, j + 1]]
        total += dist_mat[solution[i, -1], solution[i, 0]]
        evaluation[i] = total
    return evaluation


@njit()
def rank_selection(evaluation, best_prob, worst_prob):
    population_size = evaluation.shape[0]
    score = np.zeros(evaluation.shape)
    rank = np.argsort(-evaluation)
    for i in range(population_size):
        score[rank[i]] = worst_prob + (best_prob - worst_prob) * i / (population_size - 1)
    return score, rank


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
def crossover(father, mother, begin, end):
    num_city = father.shape[0]
    temp = np.ones(num_city, dtype=int32)
    child = np.zeros(num_city, dtype=int32)

    for i in range(begin):
        city = father[i]
        child[i] = city
        temp[city] = 0

    pos = begin
    for i in range(begin, end):
        city = mother[i]
        if temp[city]:
            child[pos] = city
            temp[city] = 0
            pos += 1

    for i in range(begin, num_city):
        city = father[i]
        if temp[city]:
            child[pos] = city
            temp[city] = 0
            pos += 1

    return child


@njit()
def generate_crossover_position(num_city):
    pos = np.random.randint(0, num_city, 2)
    if pos[0] > pos[1]:
        swap(pos, 0, 1)


@njit()
def swap(arr, pos1, pos2):
    temp = arr[pos2]
    arr[pos2] = arr[pos1]
    arr[pos1] = temp


@njit()
def mutate(child, mutation_rate):
    if np.random.rand() < mutation_rate:
        num_city = child.shape[0]
        pos1 = np.random.randint(num_city)
        pos2 = np.random.randint(num_city)
        swap(child, pos1, pos2)


@njit()
def parent_selection(score):
    probability = score / score.sum()  # get probability distribution
    cdf = np.cumsum(probability)  # get cumulative distribution function
    return np.searchsorted(cdf, np.random.rand()), np.searchsorted(cdf, np.random.rand())


@njit()
def ga_solve(num_city, dist_mat, population_size, generation_size):
    initial_solution = generate_initial_solution(num_city, population_size)  # generate initial solution
    evaluation = evaluate_solution(initial_solution, dist_mat)  # evaluate initial solution
    score, rank = rank_selection(evaluation, 0.9, 0.1)  # rank based selection
    father, mother = parent_selection(score)  # get parents
    crossover_pos = generate_crossover_position(num_city)  # get crossover position
    child = crossover(father, mother, crossover_pos[0], crossover_pos[1])  # reproduce

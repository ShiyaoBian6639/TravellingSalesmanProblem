import numpy as np
from numba import njit, int32, prange


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
    """
    :param solution: row population size, col num_city
    :param dist_mat:
    :return: route length of each row
    """
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
def rank_selection(evaluation, worst_prob, best_prob):
    """
    perform rank based selection
    :param evaluation: individual score represented as route length
    :param best_prob: highest probability of being selected as parents
    :param worst_prob: lowest probability of being selected as parents
    :return: probability of each individual being selected as parents, rank of each individual
    """
    population_size = evaluation.shape[0]
    probability = np.zeros(evaluation.shape)
    rank = np.argsort(-evaluation)
    for i in range(population_size):
        probability[rank[i]] = worst_prob + (best_prob - worst_prob) * i / (population_size - 1)
    return probability


@njit()
def roulette_selection(fitness):
    return fitness / fitness.sum()


@njit()
def tournament_selection():
    pass


@njit()
def check_individual_feasibility(individual):
    """
    validates the crossover and mutation procedure
    :param individual: a permutation of num_city
    :return: if the individual is feasible
    """
    return len(np.unique(individual)) == len(individual)


@njit()
def crossover(father, mother, begin, end):
    """
    perform crossover operation
    :param father: feasible individual
    :param mother: feasible individual
    :param begin: beginning position of gene sequence
    :param end: ending position of gene sequence
    :return: feasible individual (child)
    """
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
def generate_crossover_position(num_city, num_child):
    """
    randomly select crossover position
    :param num_city:
    :param num_child:
    :return: sorted crossover position
    """
    pos = np.random.randint(0, num_city, (num_child, 2))
    for i in range(num_child):
        if pos[i, 0] > pos[i, 1]:
            temp = pos[i, 1]
            pos[i, 1] = pos[i, 0]
            pos[i, 0] = temp
    return pos


@njit()
def swap(arr, pos1, pos2):
    temp = arr[pos2]
    arr[pos2] = arr[pos1]
    arr[pos1] = temp


@njit()
def mutate(child, mutation_rate):
    """
    mutation operation is performed by swapping two random positions in chromosome
    :param child: feasible individual inheriting parents gene
    :param mutation_rate: probability of mutation
    :return: mutated child (mutation may not happen)
    """
    if np.random.rand() < mutation_rate:
        num_city = child.shape[0]
        pos1 = np.random.randint(num_city)
        pos2 = np.random.randint(num_city)
        swap(child, pos1, pos2)


@njit()
def mating(probability):
    """
    :param probability: the probability of each individual to be chosen as parents
    :return: father_list, mother_list
    """
    qualified = probability > np.random.random(probability.shape)
    num_parents = int(qualified.sum() / 2)
    father_list = np.zeros(num_parents, dtype=int32)
    mother_list = np.zeros(num_parents, dtype=int32)
    count = 0
    count_father = 0
    count_mother = 0
    for index, boolean in enumerate(qualified):
        if boolean:
            if count % 2:
                father_list[count_father] = index
                count_father += 1
            else:
                mother_list[count_mother] = index
                count_mother += 1
            count += 1
            if count == num_parents * 2:
                break
    return father_list, mother_list


@njit()
def reproduce(father_list, mother_list, population, evaluation, dist_mat, mutation_rate):
    """
    generating children inheriting parents gene
    :param father_list: index of father
    :param mother_list: index of mother
    :param population: current population
    :param evaluation: fitness of population
    :param dist_mat: distance matrix
    :param mutation_rate: mutation rate
    :return: population + new born children
    """
    population_size, num_city = population.shape
    num_child = len(father_list)
    pos = generate_crossover_position(num_city, num_child)
    next_generation = np.zeros((num_child, num_city), dtype=int32)
    for i in range(num_child):
        child = crossover(population[father_list[i]], population[mother_list[i]], pos[i, 0], pos[i, 1])
        mutate(child, mutation_rate)
        next_generation[i] = child
    next_generation_eval = evaluate_solution(next_generation, dist_mat)
    population = np.vstack((population, next_generation))
    evaluation = np.hstack((evaluation, next_generation_eval))
    return population, evaluation


@njit(parallel=True)
def mutation(population, evaluation, dist_mat, mutation_rate):
    population_size, num_city = population.shape
    pos = generate_crossover_position(num_city, population_size)
    mutation_prob = np.random.random(population_size)
    for i in prange(population_size):
        if mutation_prob[i] < mutation_rate:
            if pos[i, 1] - pos[i, 0] == 1:
                pos1 = population[i, pos[i, 0] - 1]
                pos2 = population[i, pos[i, 0]]
                pos3 = population[i, pos[i, 1]]
                pos4 = population[i, (pos[i, 1] + 1) % num_city]
                destroy = dist_mat[pos1, pos2] + dist_mat[pos3, pos4]
                repair = dist_mat[pos1, pos3] + dist_mat[pos2, pos4]
            elif pos[i, 1] - pos[i, 0] == num_city - 1:
                pos1 = population[i, pos[i, 1] - 1]
                pos2 = population[i, pos[i, 1]]
                pos3 = population[i, pos[i, 0]]
                pos4 = population[i, (pos[i, 0] + 1) % num_city]
                destroy = dist_mat[pos1, pos2] + dist_mat[pos3, pos4]
                repair = dist_mat[pos1, pos3] + dist_mat[pos2, pos4]
            else:
                pos1 = population[i, pos[i, 0] - 1]
                pos2 = population[i, pos[i, 0]]
                pos3 = population[i, (pos[i, 0] + 1) % num_city]
                pos4 = population[i, pos[i, 1] - 1]
                pos5 = population[i, pos[i, 1]]
                pos6 = population[i, (pos[i, 1] + 1) % num_city]
                destroy = dist_mat[pos1, pos2] + dist_mat[pos2, pos3] + dist_mat[pos4, pos5] + dist_mat[pos5, pos6]
                repair = dist_mat[pos1, pos5] + dist_mat[pos5, pos3] + dist_mat[pos4, pos2] + dist_mat[pos2, pos6]
            evaluation[i] += repair - destroy
            temp = population[i, pos[i, 1]]
            population[i, pos[i, 1]] = population[i, pos[i, 0]]
            population[i, pos[i, 0]] = temp
    return population, evaluation


@njit()
def natural_selection(population, evaluation, population_size):
    """
    determines who can survive
    :param population: current population
    :param evaluation: route length of each individual in the population
    :param population_size: number of individuals allowed in the environment
    :return: new population and their route length after natural selection
    """
    rank = np.argsort(-evaluation)
    survive = rank < population_size
    new_population = population[survive, :]
    new_evaluation = evaluation[survive]
    return new_population, new_evaluation


# merge a sorted list and another unsorted list
@njit()
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
    best_solution_pool = np.zeros((generation_size, num_city), dtype=int32)  # best solution in each generation
    global_best_score = np.inf  # the global best score
    global_best_route = np.zeros(num_city, dtype=int32)  # the global best route
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


@njit(parallel=True)
def multi_start_ga(num_city, dist_mat, population_size, generation_size, mutation_rate, low_prob, high_prob,
                   num_threads):
    global_score = np.zeros(num_threads)
    global_route = np.zeros((num_threads, num_city), dtype=int32)
    for i in prange(num_threads):
        _, _, score, route = ga_solve(num_city, dist_mat, population_size, generation_size, mutation_rate, low_prob,
                                      high_prob)
        global_score[i] = score
        global_route[i] = route
    index = np.argmin(global_score)
    return global_score[index], global_route[index]

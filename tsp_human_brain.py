from tsp_utils import Tsp, data_generator, solve_instance, evaluate_user_solution
import time
import numpy as np
import pandas as pd
import math

# generate instances
cities = list(range(2, 5))
table = np.zeros((len(cities), 3))
n_instance = 1
for index, n_city in enumerate(cities):
    data_generator(n_city, n_instance)
    instance = Tsp(n_city, n_instance)
    start = time.perf_counter()
    best_route, shortest_len = instance.exhaustive_search()
    time_consumed = time.perf_counter() - start
    table[index, 0] = n_city
    table[index, 1] = math.factorial(n_city)
    table[index, 2] = time_consumed
    print(f"{n_city} cities takes {time_consumed} seconds")

col_names = ['城市数量', '有效路径数量', '穷举时间']
table_df = pd.DataFrame(data=table, columns=col_names)
table_df.to_csv("./data/table.csv", encoding='gbk')
# 20 cities
fact20 = math.factorial(20)
print("{:e}".format(fact20))
fact20 / math.factorial(10) * 12.36 / 3600 / 24 / 356

# generate multiple instances and corresponding plot


# input user route
# plot user route
instances = list(range(3))

# generate multiple instances and corresponding plot
optimal_sol = dict()  # key: (n_city, n_instance), value: optimal route length
solve_instance(n_city=20, instances=instances, optimal_sol=optimal_sol)
solve_instance(n_city=30, instances=instances, optimal_sol=optimal_sol)

# evaluate user route
evaluate_user_solution(20, 1,
                       [10, 9, 12, 14, 5, 13, 0, 7, 15, 6, 8, 3, 11, 17, 2, 4, 18, 1, 16, 19], optimal_sol)
len([10, 9, 12, 14, 5, 13, 0, 7, 15, 6, 8, 3, 11, 17, 2, 4, 18, 1, 16, 19])

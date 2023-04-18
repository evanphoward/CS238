import networkx as nx
from scipy.optimize import linprog
import numpy as np
import matplotlib.pyplot as plt

# Two questions: 
# - Want to find EF solutions for general cases
# - Want to know under what values of w_i is an EF solution also a welfare-maximizing assn

def welfare_maximizing_assn(agents_values, total_rent):
    num_agents = len(agents_values)
    num_rooms = len(agents_values[0])

    # Create the bipartite graph
    G = nx.Graph()
    G.add_nodes_from(range(num_agents), bipartite=0)
    G.add_nodes_from(range(num_agents, num_agents+num_rooms), bipartite=1)

    # Add edges with weights (values)
    for i, values in enumerate(agents_values):
        for j, value in enumerate(values):
            G.add_edge(i, num_agents+j, weight=value)

    # Find max weight matching
    matching_set = nx.max_weight_matching(G, maxcardinality=True)
    return {agent: room - num_agents for agent, room in ((a, b) if a < b else (b, a) for a, b in matching_set)}
    
def rent_division(agents_values, agents_weights, total_rent):
    num_agents = len(agents_values)
    num_rooms = len(agents_values[0])

    matching = welfare_maximizing_assn(agents_values, agents_weights)

    # Prepare data for linear programming
    c = [-1] * num_rooms
    A_eq = np.zeros((1, num_rooms))
    b_eq = [total_rent]

    A_ub = []
    b_ub = []

    for i in range(num_agents):
        room_idx = matching[i]
        agent_room_value = agents_values[i][room_idx]

        for j in range(num_agents):
            if i != j:
                other_room_idx = matching[j]
                agent_value_for_other_room = agents_values[i][other_room_idx]

                row = [0] * num_rooms
                row[room_idx] = agents_weights[i]
                row[other_room_idx] = -agents_weights[i]

                A_ub.append(row)
                b_ub.append(agent_room_value - agent_value_for_other_room)

        A_eq[0][room_idx] = 1

    # Linear programming to find room prices
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, method="highs")
    room_prices = res.x

    if room_prices is None:
        return False

    # Assign rooms and prices
    assignments = {agent: (matching[agent], round(room_prices[matching[agent]], 2)) for agent in range(num_agents)}

    return assignments

def is_envy_free(assignments, agents_values, agents_weights, total_rent):
    num_agents = len(agents_values)

    for i in range(num_agents):
        i_room, i_payment = assignments[i]
        i_value = agents_values[i][i_room]
        i_utility = i_value - (i_payment * agents_weights[i])

        for j in range(num_agents):
            if i != j:
                j_room, j_payment = assignments[j]
                i_utility_for_j_room = agents_values[i][j_room] - (j_payment * agents_weights[i])

                if i_utility < i_utility_for_j_room - 0.01:
                    print(i_utility, i_utility_for_j_room, i, j)
                    return False

    return True

def run_instance(agents_values, agents_weights, total_rent, print_ans=True):
    assert(all(sum(values) == total_rent for values in agents_values))
    assignments = rent_division(agents_values, agents_weights, total_rent)
    if not assignments:
        if print_ans:
            print("No EF soln under welfare-maximizing assignment")
        return False
    # If Linear Programming is correct, then it must be envy-free
    try:
        assert(is_envy_free(assignments, agents_values, agents_weights, total_rent))
    except AssertionError:
        print("Envy Freeness Check Failed", agents_weights)

    if print_ans:
        print("EF Solution found:", assignments)
    return True

agents_values = [
    [6, 2, 2],
    [8, 1, 1],
    [3, 3, 4]
]
agents_weights = [0.4, 0.7, 1]
total_rent = 10

run_instance(agents_values, agents_weights, total_rent)

# precision = 100
# upper_bound = 1
# bool_array = np.zeros(((upper_bound * precision) + 1, (upper_bound * precision) + 1), dtype=bool)
# for i in range(0, (upper_bound * precision) + 1):
#     for j in range(0, (upper_bound * precision) + 1):
#         bool_array[i, j] = run_instance(agents_values, [i / precision, j / precision], total_rent, False)

# # create a color map that maps True to green and False to red
# cmap = plt.get_cmap('RdYlGn')
# cmap.set_bad(color='red')
# cmap.set_over(color='green')
# cmap.set_under(color='red')

# # plot the boolean array as an image
# fig, ax = plt.subplots()
# ax.imshow(bool_array, cmap=cmap, interpolation='nearest', vmin=0, vmax=1, extent=[0, upper_bound, 0, upper_bound])
# ax.set_xlabel("w_0")
# ax.set_ylabel("w_1")

# # show the plot
# plt.show()

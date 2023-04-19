import networkx as nx
from scipy.optimize import linprog
import numpy as np

def rent_division(agents_values, total_rent):
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
    matching = {agent: room - num_agents for agent, room in ((a, b) if a < b else (b, a) for a, b in matching_set)}

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
                row[room_idx] = 1
                row[other_room_idx] = -1

                A_ub.append(row)
                b_ub.append(agent_room_value - agent_value_for_other_room)

        A_eq[0][room_idx] = 1

    # Linear programming to find room prices
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, method="highs")
    room_prices = res.x

    # Assign rooms and prices
    assignments = {agent: (matching[agent], round(room_prices[matching[agent]], 2)) for agent in range(num_agents)}

    return assignments

def is_envy_free(assignments, agents_values, total_rent):
    num_agents = len(agents_values)

    for i in range(num_agents):
        i_room, i_payment = assignments[i]
        i_value = agents_values[i][i_room]
        i_utility = i_value - i_payment

        for j in range(num_agents):
            if i != j:
                j_room, j_payment = assignments[j]
                i_utility_for_j_room = agents_values[i][j_room] - j_payment

                if i_utility < i_utility_for_j_room - 0.01:
                    print(i, j)
                    return False

    return True

agents_values = [
    [300, 200, 100],
    [100, 300, 200],
    [200, 100, 300]
]

total_rent = 600

assignments = rent_division(agents_values, total_rent)
print(assignments)
print(is_envy_free(assignments, agents_values, total_rent))

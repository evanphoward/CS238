import networkx as nx
from scipy.optimize import linprog
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import random

def matching_assn(agents_values, agents_weights, total_rent):
    num_agents = len(agents_values)
    num_rooms = len(agents_values[0])

    # Create the bipartite graph
    G = nx.Graph()
    G.add_nodes_from(range(num_agents), bipartite=0)
    G.add_nodes_from(range(num_agents, num_agents+num_rooms), bipartite=1)

    # Add edges with weights (values)
    for i, values in enumerate(agents_values):
        for j, value in enumerate(values):
            G.add_edge(i, num_agents+j, weight=(value / agents_weights[i]))

    # Find max weight matching
    matching_set = nx.max_weight_matching(G, maxcardinality=True)
    return {agent: room - num_agents for agent, room in ((a, b) if a < b else (b, a) for a, b in matching_set)}
    
def rent_division(agents_values, agents_weights, total_rent):
    num_agents = len(agents_values)
    num_rooms = len(agents_values[0])

    matching = matching_assn(agents_values, agents_weights, agents_weights)

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
    assignments = {agent: (matching[agent], room_prices[matching[agent]]) for agent in range(num_agents)}

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

                if i_utility < i_utility_for_j_room - 0.02:
                    print(i_utility, i_utility_for_j_room, i, j)
                    return False

    return True

def run_instance(agents_values, agents_weights, total_rent, print_ans=True):
    assert(all(sum(values) == total_rent for values in agents_values))
    assignments = rent_division(agents_values, agents_weights, total_rent)
    if not assignments:
        if print_ans:
            print("No EF soln under matching assignment")
        return False
    # If Linear Programming is correct, then it must be envy-free
    try:
        assert(is_envy_free(assignments, agents_values, agents_weights, total_rent))
    except AssertionError:
        print("Envy Freeness Check Failed", agents_weights)

    if print_ans:
        print("EF Solution found:", assignments)
    return True

def check_sufficient_cond(agents_values, agents_weights, total_rent):
    failed = []
    for i, values in enumerate(agents_values):
        if total_rent * (1 - agents_weights[i]) > min(values) * len(agents_values):
            failed.append(i)
    return failed

def get_random_values(total_rent, num_rooms):
    random_numbers = np.random.rand(num_rooms - 1) * total_rent
    random_numbers.sort()
    random_numbers = np.concatenate(([0], random_numbers, [total_rent]))
    random_numbers = np.diff(random_numbers)
    return list(random_numbers)

def plot_2_agents(agents_values, total_rent, precision):
    bool_array = np.zeros((precision, precision), dtype=bool)
    for i in range(1, precision + 1):
        for j in range(1, precision + 1):
                bool_array[i - 1, j - 1] = run_instance(agents_values, [i / precision, j / precision], total_rent, False)

    # create a color map that maps True to green and False to red
    cmap = plt.get_cmap('RdYlGn')
    cmap.set_bad(color='red')
    cmap.set_over(color='green')
    cmap.set_under(color='red')

    # plot the boolean array as an image
    fig, ax = plt.subplots()
    ax.imshow(bool_array, cmap=cmap, interpolation='nearest', vmin=0, vmax=1, extent=[1 / precision, 1, 1 / precision, 1])
    ax.set_xlabel("w_0")
    ax.set_ylabel("w_1")

    # show the plot
    plt.show()

def plot_3_agents(agents_values, total_rent, precision):
    bool_array = np.zeros((precision, precision, precision), dtype=bool)
    for i in range(1, precision + 1):
        for j in range(1, precision + 1):
            for k in range(1, precision + 1):
                bool_array[i - 1, j - 1, k - 1] = run_instance(agents_values, [i / precision, j / precision, k / precision], total_rent, False)

    # Convert boolean data to integers (1 for True, 0 for False)
    int_data = bool_array.astype(int)

    # Create a meshgrid for the x, y, and z coordinates
    x, y, z = np.meshgrid(np.arange(bool_array.shape[0]), np.arange(bool_array.shape[1]), np.arange(bool_array.shape[2]), indexing='ij')

    # Create the 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    kw = {
        'vmin': 0,
        'vmax': 1,
        'cmap': plt.cm.get_cmap('RdYlGn', 2),  # Set colormap to RdYlGn and discretize it into 2 levels
    }

    _ = ax.contourf(
        x[:, :, -1], y[:, :, -1], int_data[:, :, -1],
        zdir='z', offset=x.max(), **kw
    )
    _ = ax.contourf(
        x[:, 0, :], int_data[:, 0, :], z[:, 0, :],
        zdir='y', offset=0, **kw
    )
    _ = ax.contourf(
        int_data[-1, :, :], y[-1, :, :], z[-1, :, :],
        zdir='x', offset=x.max(), **kw
    )

    ax.set_xlabel('W_0')
    ax.set_ylabel('W_1')
    ax.set_zlabel('W_2')

    xmin, xmax = x.min(), x.max()
    ymin, ymax = y.min(), y.max()
    zmin, zmax = z.min(), z.max()
    ax.set(xlim=[xmin, xmax], ylim=[ymin, ymax], zlim=[zmin, zmax])
    edges_kw = dict(color='0.8', linewidth=0.5, zorder=1e3)
    ax.plot([xmax, xmax], [ymin, ymax], [zmax, zmax], **edges_kw)
    ax.plot([xmin, xmax], [ymin, ymin], [zmax, zmax], **edges_kw)
    ax.plot([xmax, xmax], [ymin, ymin], [zmin, zmax], **edges_kw)

    ax.set_xticklabels([round(float(item) / precision, 2) for item in ax.get_xticks()])
    ax.set_yticklabels([round(float(item) / precision, 2) for item in ax.get_yticks()])
    ax.set_zticklabels([round(float(item) / precision, 2) for item in ax.get_zticks()])

    ax.view_init(30, -60)  # Adjust the viewing angle
    plt.show()


# EXAMPLE
total_rent = 100
agents_values_2 = [
    [80, 20],
    [50, 50]
]
agents_values_3 = [
    [10, 80, 10],
    [20, 50, 30],
    [20, 70, 10]
]
agents_values_8 = [
    [15, 10, 5, 0, 30, 10, 20, 10],
    [15, 30, 5, 0, 10, 10, 20, 10],
    [20, 10, 5, 0, 30, 10, 15, 10],
    [15, 10, 5, 10, 30, 10, 20, 0],
    [15, 5, 10, 0, 20, 10, 30, 10],
    [10, 15, 5, 30, 0, 10, 10, 20],
    [0, 10, 10, 15, 30, 5, 10, 20],
    [15, 10, 5, 0, 30, 10, 20, 10]
]

plot_2_agents(agents_values_2, total_rent, 10)
plot_3_agents(agents_values_3, total_rent, 10)
run_instance(agents_values_8, [0.5, 0.7, 1, 0.8, 1, 1.2, 1, 0.2], total_rent)


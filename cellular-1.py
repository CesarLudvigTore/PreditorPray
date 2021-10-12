""" Imports """
import pylab
from scipy.ndimage import measurements
from scipy.ndimage import generate_binary_structure
import numpy as np
import random as rnd
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib import colors


plt.rcParams.update({'font.size': 20})

""" Functions """


def controller(empty_pop, food_pop, human_pop, undead_pop):
    global Food_growth, Human_growth, Undead_growth, Undead_growth,\
        Human_strength, Human_expansion, Human_survive_rate, Human_growth_in_woods, it,\
        humans_str, humans_s_rate, std_val, prop_undead_move

    empty_ratio_c = empty_pop/(grid_width*grid_height)
    food_ratio_c = food_pop/(grid_width*grid_height)

    """ Human survive rate """
    Human_survive_rate = Human_survive_rate
    """
    # Test 1
    if 400 <= it < 500:
        Human_survive_rate = 0.8
    else:
        Human_survive_rate = 0.99
    """

    """
    # Test2
    human_survive_rates = [0.9, 0.99, 1]
    
    if it % 250 == 0:
        Human_survive_rate = human_survive_rates[np.random.randint(0, len(human_survive_rates))]
    """
    humans_s_rate.append(Human_survive_rate)

    """ Human strength """
    undead_human_ratio = undead_pop/(human_pop + undead_pop)
    k_hs = 0.5
    kp_hs = 0.7
    a_hs = (k_hs**2 - 2*k_hs*kp_hs + kp_hs) / (((k_hs - 1)**2)*k_hs**2)
    b_hs = (-2*k_hs**3 + 3*(k_hs**2)*kp_hs - kp_hs) / (((k_hs - 1)**2)*k_hs**2)
    c_hs = (k_hs**3 - 3*k_hs*kp_hs + 2*kp_hs) / (((k_hs - 1)**2)*k_hs)
    Human_strength = a_hs*undead_human_ratio**3 + b_hs*undead_human_ratio**2 + c_hs*undead_human_ratio
    humans_str.append(Human_strength)


    """ Human expansion """
    e = 1/5
    Human_expansion = (food_ratio_c**2 + food_ratio_c**3 + empty_ratio_c)*e
    # print(Human_expansion)

    """ Food growth """

    """
    # test3
    stds = [0.01, 0.001, 0.0001, 0.00001]
    if it < 500:
        std = stds[0]
    elif 500 <= it < 700:
        std = stds[1]
    elif 700 <= it < 900:
        std = stds[2]
    elif 900 <= it < 1100:
        std = stds[3]
    elif it >= 1100:
        std = stds[0]
    std_val.append(std)
    """
    std = 0.01
    Food_growth = fitted_gauss(Food_growth, std)

    """ Undead movement """
    # test 4
    """
    movements = [0.5, 0.6, 0.8, 0.9, 1]
    move = 0
    if it < 500:
        move = movements[0]
    elif 500 <= it < 750:
        move = movements[1]
    elif 750 <= it < 1000:
        move = movements[2]
    elif 1000 <= it < 1250:
        move = movements[3]
    elif it >= 1250:
        move = movements[4]
    prop_undead_move = move
    """
    move = prop_undead_move
    move_val.append(move)



def human_overuse_food(prob, n):
    p = prob*n**(1/3)
    # p += rnd.uniform(-Food_growth, Food_growth)

    return p


def fitted_gauss(x, std):
    mu = 0.5
    fitted_val = _gauss(x, std, mu) - _gauss(0, std, mu)
    return fitted_val


def _gauss(x, std, mu):
    const = (2 * np.pi * std ** 2) ** (-1)
    exponent = (-(x-mu)**2)/(2*std**2)
    exp = np.exp(exponent)
    val = const*exp
    return val


def generate_start_grid():

    grid_positions = grid_height*grid_width

    plants_amount_start_tiles = grid_positions*food_start_ratio
    prey_amount_start_tiles = grid_positions*human_start_ratio
    predator_amount_start_tiles = grid_positions*undead_start_ratio

    empty_tile = 0
    food_tile = 1
    human_tile = 2
    undead_tile = 3

    start_list = [empty_tile]*grid_positions
    for i in range(grid_positions):
        if i < plants_amount_start_tiles:
            start_list[i] = food_tile
        elif i < plants_amount_start_tiles + prey_amount_start_tiles:
            start_list[i] = human_tile
        elif i < plants_amount_start_tiles + prey_amount_start_tiles + predator_amount_start_tiles:
            start_list[i] = undead_tile

    rnd.shuffle(start_list)

    f = 0
    h = 0
    z = 0
    e = 0
    for elem in start_list:
        if elem == 0:
            e += 1
        if elem == 1:
            f += 1
        if elem == 2:
            h += 1
        if elem == 3:
            z += 1
    print(e, f, h, z)

    start_grid = np.zeros((grid_height, grid_width), dtype=int)

    pos = 0
    for x in range(grid_width):
        for y in range(grid_height):
            start_grid[y, x] = start_list[pos]
            pos += 1

    return start_grid


def is_humans_close_to_food(cur_grid, y_g, x_g):
    global human_is_close_to_food_check_grid

    close_humans = []

    nearest_x = [-1, 0, 1]
    nearest_y = [-1, 0, 1]

    if x_g == 0:
        nearest_x = [0, 1]
    elif x_g == grid_width - 1:
        nearest_x = [-1, 0]

    if y_g == 0:
        nearest_y = [0, 1]
    elif y_g == grid_height - 1:
        nearest_y = [-1, 0]

    for xx in nearest_x:
        for yy in nearest_y:
            x_c = x_g + xx
            y_c = y_g + yy
            if xx == 0 and yy == 0:
                human_is_close_to_food_check_grid[y_g, x_g] = 11
            else:
                if cur_grid[y_c, x_c] == 1:
                    return True
                elif cur_grid[y_c, x_c] == 2 and human_is_close_to_food_check_grid[y_c, x_c] != 11:
                    close_humans.append((y_c, x_c))
                    human_is_close_to_food_check_grid[y_g, x_g] = 11
                elif cur_grid[y_c, x_c] in [0, 3] and human_is_close_to_food_check_grid[y_c, x_c] != 11:
                    human_is_close_to_food_check_grid[y_g, x_g] = 11

    for humans in close_humans:
        y_n, x_n = humans
        return is_humans_close_to_food(cur_grid, y_n, x_n)

    return False


def which_humans_are_close_to_food(given_grid, matrix_with_humans):
    global humans_close_to_food

    for x in range(grid_width):
        for y in range(grid_height):
            human_cluster_nr = matrix_with_humans[y, x]
            if human_cluster_nr != 0 and human_cluster_nr not in humans_close_to_food:
                if is_humans_close_to_food(given_grid, y, x):
                    humans_close_to_food.append(human_cluster_nr)


def step(itera):
    global grid, human_is_close_to_food_check_grid, humans_close_to_food

    grid_new = np.zeros((grid_height, grid_width))
    humans_close_to_food = []  # resets humans close to food
    s = generate_binary_structure(2, 2)
    human_grid = grid == 2  # Grid with all humans with ones, everything else 0

    # Array with human clusters numbers
    labeled_array_with_humans, num_features_humans = measurements.label(human_grid, structure=s)
    # List with human cluster areas human_areas[i] is the i'th human cluster's area
    human_areas = measurements.sum(human_grid, labeled_array_with_humans,
                                   index=pylab.arange(labeled_array_with_humans.max() + 1))

    # Checks which humans that are close to food
    which_humans_are_close_to_food(grid, labeled_array_with_humans)

    for actor in ['undead', 'human', 'food', 'empty']:
        for x in range(grid_width):
            for y in range(grid_height):

                # Movements to reach close tiles
                nearest_x = [-1, 0, 1]
                nearest_y = [-1, 0, 1]

                if x == 0:
                    nearest_x = [0, 1]
                elif x == grid_width-1:
                    nearest_x = [-1, 0]
                if y == 0:
                    nearest_y = [0, 1]
                elif y == grid_height-1:
                    nearest_y = [-1, 0]

                if actor == 'undead' and grid[y, x] == 3 and grid_new[y, x] == 0:
                    places_to_attack = []
                    places_to_move = [(y, x)]
                    for x_a in nearest_x:
                        for y_a in nearest_y:
                            if x_a == 0 and y_a == 0:
                                pass
                            else:
                                x_ar = x + x_a
                                y_ar = y + y_a

                                if grid[y_ar, x_ar] == 2 and grid_new[y_ar, x_ar] == 0:
                                    # Undead attacks humans
                                    places_to_attack.append((y_ar, x_ar))
                                elif grid[y_ar, x_ar] == 0 or grid[y_ar, x_ar] == 1:
                                    if grid_new[y_ar, x_ar] == 0:
                                        places_to_move.append((y_ar, x_ar))

                    if places_to_attack == []:
                        # Undead move
                        if rnd.random() < prop_undead_move:
                            rnd.shuffle(places_to_move)
                        # print('move', places_to_move)
                        y_new, x_new = places_to_move[0]
                        # print('move', y_new, x_new)
                        grid_new[y, x] = 0
                        grid_new[y_new, x_new] = 3

                    elif places_to_attack != []:
                        # Undead attack
                        rnd.shuffle(places_to_attack)

                        y_att, x_att = places_to_attack[0]
                        if rnd.random() > Human_strength:
                            # Humans looses and becomes undead
                            grid_new[y_att, x_att] = 3
                            grid_new[y, x] = 3
                            # print('loose')
                        else:
                            # Human wins --> undead die and the tile turn empty

                            grid_new[y, x] = 0
                            # print('win')

                elif actor == "human" and grid[y, x] == 2 and grid_new[y, x] == 0:
                    human_is_close_to_food_check_grid = np.zeros((grid_height, grid_width))
                    empty_space = []
                    food_space = []
                    close_to_food = False
                    human = labeled_array_with_humans[y, x]

                    if human in humans_close_to_food:
                        close_to_food = True

                    for x_a in nearest_x:  # Loop through all close tiles
                        for y_a in nearest_y:
                            if x_a == 0 and y_a == 0:
                                pass
                            else:
                                x_ar = x + x_a
                                y_ar = y + y_a
                                if grid[y_ar, x_ar] == 0:
                                    empty_space.append((y_ar, x_ar))
                                elif grid[y_ar, x_ar] == 1:
                                    food_space.append((y_ar, x_ar))
                    places_to_move = empty_space*5 + food_space
                    if close_to_food and rnd.random() < Human_survive_rate:
                        # Humans survives to next phase
                        grid_new[y, x] = 2
                        if places_to_move != [] and rnd.random() < Human_expansion:
                            # Humans expands their territory
                            rnd.shuffle(places_to_move)
                            y_n, x_n = places_to_move[0]
                            grid_new[y_n, x_n] = 2
                    else:
                        # Humans starve or has an accident, they die and become undead
                        grid_new[y, x] = 3

                elif actor == 'food' and grid[y, x] == 1 and grid_new[y, x] == 0:

                    close_human_pos = []
                    human_cluster_size = 0
                    checked_human_area_indices = []

                    for x_a in nearest_x:   # Loop through all close tiles
                        for y_a in nearest_y:
                            if x_a == 0 and y_a == 0:
                                pass
                            else:
                                x_ar = x + x_a
                                y_ar = y + y_a
                                if grid[y_ar, x_ar] == 2:
                                    close_human_pos.append((y_ar, x_ar))

                    for human_pos in close_human_pos:
                        y_c, x_c = human_pos
                        cur_area_index = labeled_array_with_humans[y_c, x_c]
                        if cur_area_index not in checked_human_area_indices:
                            human_cluster_size += human_areas[cur_area_index]
                            checked_human_area_indices.append(cur_area_index)

                    if rnd.random() < human_overuse_food(Human_overuse_food_var, human_cluster_size):
                        grid_new[y, x] = 0
                    elif rnd.random() < Human_growth_in_woods:
                        grid_new[y, x] = 2
                    else:
                        grid_new[y, x] = 1

                elif actor == 'empty' and grid[y, x] == 0 and grid_new[y, x] == 0:

                    extra_prob_to_food_growth = 0
                    extra_prob_to_undead_growth = 0

                    is_human_close = False
                    is_food_close = False

                    for x_a in nearest_x:
                        for y_a in nearest_y:
                            if x_a == 0 and y_a == 0:
                                pass
                            else:
                                x_ar = x + x_a
                                y_ar = y + y_a
                                if grid[y_ar, x_ar] == 2:
                                    is_human_close = True
                                elif grid[y_ar, x_ar] == 1:
                                    is_food_close = True
                                if is_human_close and is_food_close:
                                    break

                    if is_human_close:
                        extra_prob_to_food_growth += 0.1
                        extra_prob_to_undead_growth += 0.1

                    if is_food_close:
                        extra_prob_to_food_growth += 0.2

                    if rnd.random() < Food_growth + extra_prob_to_food_growth:
                        grid_new[y, x] = 1
                    elif rnd.random() < Human_growth:
                        grid_new[y, x] = 2
                    elif rnd.random() < Undead_growth:
                        grid_new[y, x] = 3

    grid = grid_new
    iteration.append(itera)

    empty_pop = 0
    food_pop = 0
    human_pop = 0
    undead_pop = 0

    for x in range(grid_width):
            for y in range(grid_height):
                tile_type = grid[y, x]
                if tile_type == 0:
                    empty_pop += 1
                elif tile_type == 1:
                    food_pop += 1
                elif tile_type == 2:
                    human_pop += 1
                elif tile_type == 3:
                    undead_pop += 1

    controller(empty_pop, food_pop, human_pop, undead_pop)  # Update variables
    food_ratio.append(food_pop/(grid_width*grid_height))
    human_ratio.append(human_pop/(grid_width*grid_height))
    undead_ratio.append(undead_pop/(grid_width*grid_height))


def animate(i):
    global it
    if i > 0:
        it = i

        # pylab.savefig(str(i) + '.png')
        step(i)
        im.set_data(grid)
        food_graph.set_data(iteration, food_ratio)
        human_graph.set_data(iteration, human_ratio)
        undead_graph.set_data(iteration, undead_ratio)

    return im, food_graph, human_graph, undead_graph,


""" Global variables """

""" Constants """
it = 0

# Time steps
N = 2000

# grid side lengths
grid_height = 100
grid_width = 100

# Start ratios
human_start_ratio = 0.4
undead_start_ratio = 0
food_start_ratio = 0.4

""" Constant parameters """
Human_survive_rate = 0.99     # Human's probability to not die of accident
prop_undead_move = 0.5  # Undead's probability to move
Undead_growth = 0.00001     # Undead growth from empty tile
Human_growth = 0.0001     # Human growth from empty tile
Human_growth_in_woods = 0.0001  # Human growth from food tile

""" Parameters that varies """
Human_strength = 0.7     # Human's probability to win a fight against undead.
Food_growth = 0.5     # Food growth from empty tile
Human_overuse_food_var = 0.1     # Human's probability to over use a food tile with only on human tile adjunct
Human_expansion = 0.01      # probability that an empty or food tile close to a human tile will turn human

""" Init """
human_is_close_to_food_check_grid = np.zeros((grid_height, grid_width))
humans_close_to_food = []
humans_str = []

""" First iteration """
grid = generate_start_grid()  # Initialized start grid.
iteration = [0]
food_ratio = [food_start_ratio]
human_ratio = [human_start_ratio]
undead_ratio = [undead_start_ratio]
humans_str = [0]
humans_s_rate = [0.99]
std_val = [0.01]
move_val = [0.5]

""" Plot """

#  -------------

cmap = colors.ListedColormap(['black', 'green', 'yellow', 'red'])  # [0,1,2,3]
bounds = [0, 1, 2, 3, 4]
norm = colors.BoundaryNorm(bounds, cmap.N)

fig, (continent_overview, continent_diagram) = plt.subplots(1, 2)
im = continent_overview.imshow(grid, cmap=cmap, norm=norm)
continent_diagram.set_xlim([0, N])
continent_diagram.set_xlabel('Steg')
continent_diagram.set_ylim([0, 1.1])
continent_diagram.set_ylabel('Beståndsförhållande')
continent_diagram.set_aspect(N)
food_graph, = continent_diagram.plot([], [], 'g', label='Föda')
human_graph, = continent_diagram.plot([], [], 'y', label='Människor')
undead_graph, = continent_diagram.plot([], [], 'r', label='Odöda')
continent_overview.axis('off')
continent_diagram.yaxis.set_label_position("right")
continent_diagram.yaxis.tick_right()
continent_diagram.legend()

a = animation.FuncAnimation(fig, animate, frames=N+1, interval=1, blit=True, repeat=False)
plt.show()

""" Plottar """

"""
# test 4
fig1, ax1 = plt.subplots()
color = 'tab:blue'
ax1.set_xlabel('Steg')
ax1.set_ylabel('$F$', color=color)
ax1.plot(iteration, move_val, color=color, linestyle='--')

ax1.tick_params(axis='y')

ax2 = ax1.twinx()
ax2.set_ylabel('Beståndsförhållande')
ax2.plot(iteration, food_ratio, 'g', label='Föda')
ax2.plot(iteration, human_ratio, 'y', label='Människor')
ax2.plot(iteration, undead_ratio, 'r', label='Odöda')
ax2.tick_params(axis='y')
fig1.tight_layout()

plt.show()
"""

"""
# test 3
fig1, ax1 = plt.subplots()
color = 'tab:blue'
ax1.set_xlabel('Steg')
ax1.set_yscale('log')
ax1.set_ylim([0.000008, 0.011])
ax1.set_ylabel('$\sigma$', color=color)
ax1.plot(iteration, std_val, color=color, linestyle='--')
ax1.tick_params(axis='y')

ax2 = ax1.twinx()
ax2.set_ylabel('Beståndsförhållande')
ax2.plot(iteration, food_ratio, 'g', label='Föda')
ax2.plot(iteration, human_ratio, 'y', label='Människor')
ax2.plot(iteration, undead_ratio, 'r', label='Odöda')
ax2.tick_params(axis='y')
fig1.tight_layout()

plt.show()
"""

"""
# test 2
fig1, ax1 = plt.subplots()
color = 'tab:blue'
ax1.set_xlabel('Steg')
ax1.set_ylabel('$S$ och $O$', color=color)
ax1.plot(iteration, humans_str, color=color, linestyle='--')
ax1.plot(iteration, humans_s_rate, color=color, linestyle=':')
ax1.tick_params(axis='y')

ax2 = ax1.twinx()
ax2.set_ylabel('Beståndsförhållande')
ax2.plot(iteration, food_ratio, 'g', label='Föda')
ax2.plot(iteration, human_ratio, 'y', label='Människor')
ax2.plot(iteration, undead_ratio, 'r', label='Odöda')
ax2.tick_params(axis='y')
fig1.tight_layout()

plt.show()
"""

"""
# test 1
fig1, ax1 = plt.subplots()
color = 'tab:blue'
ax1.set_xlabel('Steg')
ax1.set_yscale('log')
ax1.set_ylim([0.89, 1.01])
ax1.set_ylabel('$O$', color=color)
ax1.plot(iteration, humans_s_rate, color=color, linestyle=':')
ax1.tick_params(axis='y')

ax2 = ax1.twinx()
ax2.set_ylabel('Beståndsförhållande')
ax2.plot(iteration, food_ratio, 'g', label='Föda')
ax2.plot(iteration, human_ratio, 'y', label='Människor')
ax2.plot(iteration, undead_ratio, 'r', label='Odöda')
ax2.tick_params(axis='y')
fig1.tight_layout()

plt.show()
"""

"""
with open('200.txt', 'w') as f:

    f.write('Food mean - ' + str(np.mean(food_ratio[100:])) + '\n'
            + 'Human mean - ' + str(np.mean(human_ratio[100:])) + '\n'
            + 'Undead mean - ' + str(np.mean(undead_ratio[100:])))
            
#  -------------
"""

"""
#  -------------
food_diff =[]
human_diff = []
undead_diff = []

food_ref = 0.3547148166851833
human_ref = 0.3068628724371276
undead_ref = 0.08842092717907282

iters = [200, 500, 750, 1000, 1500, 2000]
food_means = [0.34078316831683175, 0.35658628428927686, 0.3520400921658986, 0.3523480577136515,
              0.3508518915060671, 0.3522599158337717]
human_means = [0.3147435643564357, 0.3056403990024938, 0.30667173579109064, 0.3080297447280799,
               0.3072219842969307, 0.3072329826407154]
undead_means = [0.09092574257425745, 0.08720822942643391, 0.08814086021505377, 0.08923429522752498,
                0.08793254817987152, 0.08842635455023672]

for i in range(len(iters)):
    food_diff.append(np.abs(np.mean(food_means[i]) - food_ref))
    human_diff.append(np.abs(np.mean(human_means[i]) - human_ref))
    undead_diff.append(np.abs(np.mean(undead_means[i]) - undead_ref))

plt.plot(iters, food_diff, 'g', label='Föda', marker='o')
plt.plot(iters, human_diff, 'y', label='Människor', marker='o')
plt.plot(iters, undead_diff, 'r', label='Odöda', marker='o')
plt.xlim([200, 2000])
plt.xlabel('Steg')

plt.ylabel('Avvikelse från referensvärdet')
plt.legend()
plt.show()
"""










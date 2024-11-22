# Partnership Round
def phaseIpreferences(player, community, global_random):
    # Too tired to be a good teammate
    if player.energy < 0:
        return []

    # Returns a list of [task_index, partner_id]
    preferences = []
    sorted_tasks = sorted(enumerate(community.tasks), key=lambda x: sum(x[1]), reverse=True)

    # Iterate over all tasks
    for task_index, task in sorted_tasks:
        # Get best partner for the task
        preferences = get_best_partner(preferences, task, task_index, player, community)

    return preferences

# Individual Round
def phaseIIpreferences(player, community, global_random):

    bids = []
    sorted_tasks = sorted(enumerate(community.tasks), key=lambda x: sum(x[1]), reverse=True)
    bids = get_all_possible_tasks(bids, sorted_tasks, player)

    return bids

# todo workshop this name
def get_team_size_strategy(player, community):
    """
    This function calculates a person's cost matrix for performing every task alone or partnered and then returns a preference to work alone or in partnerships.
    :param player:
    :param community:
    :return:
    """

    partnered_abilities = get_possible_partnerships(player, community)
    penalty_matrix = calculate_penalty_matrix(partnered_abilities, tasks)

    # [ 1 0 4 8 0 2 ]
    # [ 0 0 2 4 0 1 ]
    # [ 1 0 4 8 0 2 ]

    # Fre

    return penalty_matrix

def get_possible_partnerships(player, community):
    partnered_abilities = [player.abilities] + [[max(p, q) for p, q in zip(player.abilities, partner.abilities)] for partner in community.members]
    return partnered_abilities


def calculate_penalty_matrix(partnered_abilities, tasks):
    """
    Calculate the penalty matrix for given players and tasks.

    Args:
    - team_abilities: A 2D list or array where each row represents a player, or partnerships, abilities. Index 0 is the individual player. Index 1 to N represents the partnerships between the player and
    - tasks: A 2D list or array where each row represents a task's requirements.

    Returns:
    - A 2D numpy array containing the penalties for each player-task pair.
    """
    partnered_abilities = np.array(partnered_abilities)
    tasks = np.array(tasks)
    penalty_matrix = []

    for i, team in partnered_abilities:
        penalties = []
        for task in tasks:
            # Calculate energy expended as the sum of positive differences split between two partners
            penalty = np.sum(np.maximum(0, task - team)) / 2
            # First value is individual completing the task
            if i == 0:
                penalty = penalty * 2
            penalties.append(penalty)
        penalty_matrix.append(penalties)

    return np.array(penalty_matrix)


def get_all_possible_tasks(bids, sorted_tasks, player):
    for task_index, task in sorted_tasks:
        # Volunteer for all tasks that will keep your energy above 0.
        energy_cost = sum(max(task[i] - player.abilities[i], 0) for i in range(len(task)))
        if energy_cost <= player.energy:
            bids.append(task_index)

    return bids


def get_best_partner(preferences, task, task_index, player, community):
    # Find a partner with complementary skills and sufficient energy
    best_partner = None
    min_energy_cost = float('inf')

    for partner in community.members:
        # Skip invalid partners
        if partner.id == player.id or partner.energy < 0:
            continue

        # Compute combined abilities
        combined_abilities = [
            max(player.abilities[i], partner.abilities[i]) for i in range(len(task))
        ]

        # Compute energy cost per partner for the task
        energy_cost = sum(
            max(task[i] - combined_abilities[i], 0) for i in range(len(task))
        ) / 2

        # Finding best partner for the task.
        if (
                energy_cost < min_energy_cost
                and energy_cost <= player.energy
                and energy_cost <= partner.energy
        ):
            min_energy_cost = energy_cost
            best_partner = partner.id

    if best_partner is not None:
        preferences.append([task_index, best_partner])

    return preferences

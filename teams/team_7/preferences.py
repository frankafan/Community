import numpy as np
import csv

tracking_data = []


# Partnership Round
def phaseIpreferences(player, community, global_random):
    if player.energy <= 0:
        return []

    partnered_abilities = get_possible_partnerships(player, community)
    penalty_matrix = calculate_penalty_matrix(partnered_abilities, community.tasks)

    # This returns a dictionary with all stats about task difficulties / player abilities
    stats = get_stats(penalty_matrix, player, community)

    partner_bids = get_best_partner(player, community)
    return partner_bids


# Individual Round
def phaseIIpreferences(player, community, global_random):
    solo_bids = get_all_possible_tasks(community, player)
    return solo_bids


def get_stats(penalty_matrix, player, community):
    med_player_task_penalty, avg_player_task_penalty = np.median(penalty_matrix[0]), np.mean(penalty_matrix[0])
    med_player_ability, avg_player_ability = np.median(player.abilities), np.mean(player.abilities)

    avg_task_difficulty = np.mean([t for task in community.tasks for t in task])
    med_task_difficulty = np.median([t for task in community.tasks for t in task])

    avg_member_ability = np.mean([np.mean(member.abilities) for member in community.members])
    med_member_ability = np.median([np.median(member.abilities) for member in community.members])

    stats = {
        'community': {
            'avg_task_difficulty': avg_task_difficulty,
            'med_task_difficulty': med_task_difficulty,
            'avg_member_ability': avg_member_ability,
            'med_member_ability': med_member_ability
        },
        'player': {'avg_player_task_penalty': avg_player_task_penalty,
                   'med_player_task_penalty': med_player_task_penalty,
                   'avg_player_ability': avg_player_ability,
                   'med_player_ability': med_player_ability}
    }
    return stats


def get_possible_partnerships(player, community):
    """
    :return: [2D array] containing a partnership's combined ability level.
        Row 0 is for the player's solo abilities.
        Row 1:N is for the combo player/partner abilities.
    """
    partnered_abilities = np.array(
        [player.abilities] + [[int(max(p, q)) for p, q in zip(player.abilities, partner.abilities)] for
                              partner in community.members])
    return partnered_abilities


def calculate_penalty_matrix(partnered_abilities, tasks):
    """
    Calculate the penalty matrix for given players and tasks.

    Args:
    - team_abilities: A 2D list or array where each row represents a player, or partnerships, abilities. Index 0 is the individual player. Index 1 to N represents the partnerships between the player and
    - tasks: A 2D list or array where each row represents a task's requirements.

    Returns:
    - A 2D numpy array containing the penalties for each partner-task pair.
    """

    penalty_matrix = []
    for i, abilities in enumerate(partnered_abilities):
        row_penalties = []
        for task in tasks:
            # Calculate energy expended as the sum of positive differences split between two partners
            penalty = np.sum(np.maximum(0, task - abilities)) / 2
            # First value is individual completing the task so it is not shared
            if i == 0:
                penalty = penalty * 2
            row_penalties.append(penalty)
        penalty_matrix.append(row_penalties)

    return np.array(penalty_matrix)


def get_all_possible_tasks(community, player):
    """
    Volunteer for all tasks that keep your energy level above 0.
    """
    solo_bids = []
    sorted_tasks = sorted(enumerate(community.tasks), key=lambda x: sum(x[1]), reverse=True)

    for task_index, task in sorted_tasks:
        # Volunteer for all tasks that will keep your energy above 0.
        energy_cost = sum(max(task[i] - player.abilities[i], 0) for i in range(len(task)))
        if energy_cost <= player.energy:
            solo_bids.append(task_index)

    return solo_bids


def get_best_partner(player, community):
    """
    Volunteer for every task with the minimum penalty partner.
    """

    # Returns a list of [task_index, partner_id]
    preferences = []
    sorted_tasks = sorted(enumerate(community.tasks), key=lambda x: sum(x[1]), reverse=True)
    # Iterate over all tasks
    for task_index, task in sorted_tasks:

        solo_energy_cost = sum(
            max(task[i] - player.abilities[i], 0) for i in range(len(task))
        )

        # Find a partner with complementary skills and sufficient energy
        best_partner = None
        min_energy_cost = float('inf')

        for partner in community.members:
            # if player.abilities[i] == partner.abilities[i]:

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

        if best_partner is not None and (solo_energy_cost >= 1.5 * min_energy_cost):
            preferences.append([task_index, best_partner])
    return preferences


def log_turn_data(turn, community, tasks_completed):
    """
    Logs the state of the community for a single turn.
    """
    global tracking_data

    energy_levels = [player.energy for player in community.members]
    median_energy = np.median(energy_levels)
    exhausted_count = sum(1 for energy in energy_levels if energy < 0)

    tracking_data.append({
        "Turn": turn,
        "Tasks Completed": tasks_completed,
        "Median Energy": median_energy,
        "Exhausted Players": exhausted_count,
    })


def export_csv(filename="simulation_data.csv"):
    """
    Exports the logged data to a CSV file.
    """
    global tracking_data
    with open(filename, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=["Turn", "Tasks Completed", "Median Energy", "Exhausted Players"])
        writer.writeheader()
        writer.writerows(tracking_data)

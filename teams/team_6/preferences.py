from scipy.optimize import linear_sum_assignment
import numpy as np


def phaseIpreferences(player, community, global_random):
    """Return a list of task index and the partner id for the particular player. The output format should be a list of lists such that each element
    in the list has the first index task [index in the community.tasks list] and the second index as the partner id
    """
    list_choices = []
    if player.energy < 0:
        return list_choices
    num_members = len(community.members)
    partner_id = num_members - player.id - 1
    list_choices.append([0, partner_id])
    if len(community.tasks) > 1:
        list_choices.append([1, partner_id])
    list_choices.append([0, min(partner_id + 1, num_members - 1)])
    return list_choices

def phaseIIpreferences(player, community, global_random):
    """
    Return a list of tasks for the particular player to do individually.
    Includes sacrifice logic if tasks cannot be completed without dropping
    some players below the -10 energy threshold.
    """
    bids = []
    if player.energy < 0:
        return bids

    try:
        wait_energy_threshold = 0
        player_index = community.members.index(player)
        assignments, total_cost, sacrifices = assign_with_sacrifices(community, community.tasks)

        if player in sacrifices:
            # If the player is sacrificed, they cannot perform tasks
            return []

        best_task = assignments.get(player_index)
        if best_task is None:
            return []

        best_task_cost = loss_phase2(
            community.tasks[best_task], player.abilities, player.energy
        )
        if player.energy - best_task_cost < wait_energy_threshold:
            return []

        return [best_task]
    except Exception as e:
        print(e)
        return bids

def assign_phase1(tasks, members):
    num_tasks = len(tasks)
    num_members = len(members)

    # Generate all possible partnerships
    partnerships = []
    for i in range(num_members):
        for j in range(i + 1, num_members):
            partnerships.append((i, j))
    num_partnerships = len(partnerships)

    # Create cost matrix with columns for both partnerships and individual assignments
    cost_matrix = np.zeros((num_tasks, num_partnerships + num_members))

    # Fill partnership costs (first num_partnerships columns)
    for i, task in enumerate(tasks):
        for j, (member1_idx, member2_idx) in enumerate(partnerships):
            member1 = members[member1_idx]
            member2 = members[member2_idx]
            cost_matrix[i][j] = loss_phase1(task, member1, member2)

    # Fill individual costs (remaining columns)
    for i, task in enumerate(tasks):
        for j, member in enumerate(members):
            cost_matrix[i][num_partnerships + j] = loss_phase2(
                task, member.abilities, member.energy
            )

    # Solve assignment problem
    row_indices, col_indices = linear_sum_assignment(cost_matrix)

    # Convert results to assignments
    assignments = []
    for task_idx, col_idx in zip(row_indices, col_indices):
        task = tasks[task_idx]
        # If column index is less than num_partnerships, it's a partnership
        if col_idx < num_partnerships:
            member1_idx, member2_idx = partnerships[col_idx]
            member1 = members[member1_idx]
            member2 = members[member2_idx]
            loss = cost_matrix[task_idx, col_idx]
            assignments.append(([member1.id, member2.id], task, loss))
        else:
            # Individual assignment
            member_idx = col_idx - num_partnerships
            member = members[member_idx]
            loss = cost_matrix[task_idx, col_idx]
            assignments.append(([member.id], task, loss))

    total_cost = sum(
        cost_matrix[row_indices[i], col_indices[i]] for i in range(len(row_indices))
    )

    return assignments, total_cost

def assign_with_sacrifices(community, tasks):
    """
    Assign tasks, but allow for sacrifices if all tasks cannot be completed
    without dropping some players below -10 energy.
    Sacrifices are iteratively tested, starting with the weakest player.
    """
    try:
        # Step 1: Attempt initial assignment
        assignments, total_cost = assign_phase2(tasks, community.members)

        # Check if all assignments are feasible (no energy drops below -10)
        if is_assignment_feasible(assignments, community, tasks):
            return assignments, total_cost, []

        # Step 2: Iterative Sacrifice
        sacrifices = []
        remaining_members = community.members[:]
        
        while remaining_members:
            # Determine the weakest player to sacrifice
            weakest_player = determine_weakest_player(remaining_members)
            sacrifices.append(weakest_player)
            remaining_members.remove(weakest_player)

            # Reassign tasks with the remaining players
            assignments, total_cost = assign_phase2(tasks, remaining_members)

            # Check if the new assignment is feasible
            if is_assignment_feasible(assignments, community, tasks, sacrifices):
                return assignments, total_cost, sacrifices

        # If no feasible assignment is found, return with all players sacrificed
        return {}, float("inf"), sacrifices

    except Exception as e:
        print(f"Error during assignment with sacrifices: {e}")
        return {}, float("inf"), []

def is_assignment_feasible(assignments, community, tasks, sacrifices=None):
    """
    Check if the current assignment is feasible (no player drops below -10 energy).
    """
    sacrifices = sacrifices or []
    for player_index, task_index in assignments.items():
        player = community.members[player_index]
        if player in sacrifices:
            continue  # Ignore sacrificed players
        task = tasks[task_index]
        task_cost = loss_phase2(task, player.abilities, player.energy)
        if player.energy - task_cost < -10:
            return False
    return True


def determine_weakest_player(members):
    """
    Determine the weakest player in the given list of members.
    Weakness is based on abilities and proximity to the -10 energy threshold.
    """
    weakest_player = None
    max_weakness_score = float("-inf")
    for player in members:
        # Weakness score calculation
        weakness_score = (
            sum(player.abilities) * -1  # Higher abilities reduce weakness
            + (10 + player.energy) * 2  # Closer to -10 increases weakness
        )
        if weakness_score > max_weakness_score:
            max_weakness_score = weakness_score
            weakest_player = player
    return weakest_player


def assign_phase2(tasks, members):
    """
    Assign tasks to members using the Hungarian Algorithm.
    """
    num_tasks = len(tasks)
    num_members = len(members)

    cost_matrix = np.zeros((num_tasks, num_members))

    for i, task in enumerate(tasks):
        for j, member in enumerate(members):
            cost_matrix[i][j] = loss_phase2(task, member.abilities, member.energy)

    row_indices, col_indices = linear_sum_assignment(cost_matrix)

    assignments = {col_indices[i]: row_indices[i] for i in range(len(row_indices))}
    total_cost = sum(
        cost_matrix[row_indices[i], col_indices[i]] for i in range(len(row_indices))
    )

    return assignments, total_cost

def loss_phase1(task, player1, player2):
    cost = sum(
        max(task[k] - max(player1.abilities[k], player2.abilities[k]), 0)
        for k in range(len(task))
    )
    cost += max(0, cost - player1.energy - player2.energy) / 2
    cost += sum(
        max(max(player1.abilities[k], player2.abilities[k]) - task[k], 0)
        for k in range(len(task))
    )
    cost += sum(
        abs(player1.abilities[k] - player2.abilities[k]) for k in range(len(task))
    )
    return cost


def loss_phase2(task, abilities, current_energy):
    """
    Calculate the cost of a player performing a task, factoring in abilities and energy.
    """
    cost = sum([max(task[k] - abilities[k], 0) for k in range(len(abilities))])
    cost += max(0, cost - current_energy)
    return cost
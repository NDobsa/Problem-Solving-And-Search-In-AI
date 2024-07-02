# Nina Dobsa, Vittorio Vicevic
import random
import math
import time

# ------------------------------ Agents ------------------------------
class Agent:
    def __init__(self, start, goal):
        self.start = start
        self.goal = goal
        self.path = [start] # end goal will be added in the algorithm

# ------------------------------ Start ------------------------------
def starting_solution(agents):
    for agent in agents:
        current=agent.start
        visited=set(agent.path)  # 1st optimization tehnique - keeping track of visited positions
        while current!=agent.goal:
            neighbors=[pos for pos in get_neighbors(current) if pos not in visited]  # Avoid revisiting squares
            if not neighbors:
                neighbors=get_neighbors(current)  # If no unvisited neighbors, allow revisiting
            if agent.goal in neighbors:
                next_step=agent.goal
            else:
                next_step=direction_towards_goal(current, neighbors, agent.goal)
            agent.path.append(next_step)
            visited.add(next_step)
            current=next_step
    return agents


# ------------------------------ Movement ------------------------------
# 2st optimization tehnique - choosing direction towards goal
def direction_towards_goal(current, neighbors, goal):
    if goal[0]<current[0]:  # Goal is above
        neighbors=[pos for pos in neighbors if pos!=(current[0]+1, current[1])]  # Exclude down
    elif goal[0]>current[0]:  # Goal is below
        neighbors=[pos for pos in neighbors if pos!=(current[0]-1, current[1])]  # Exclude up

    elif goal[1]<current[1]:  # Goal is to the left
        neighbors=[pos for pos in neighbors if pos!=(current[0], current[1]+1)]  # Exclude right
    elif goal[1]>current[1]:  # Goal is to the right
        neighbors=[pos for pos in neighbors if pos!=(current[0], current[1]-1)]  # Exclude left

    if not neighbors:
        neighbors=get_neighbors(current)  # If all neighbors are excluded, allow all moves

    return random.choice(neighbors) # Choose only one of the solutions


def get_neighbors(position):
    neighbors = []
    directions= [(-1, 0), (1, 0), (0, -1), (0, 1)]  #up left down right
    for move in directions:
        new_pos =(position[0]+move[0], position[1]+move[1])
        if 0<=new_pos[0]<grid_size and 0<=new_pos[1]<grid_size:
            neighbors.append(new_pos)
    return neighbors


# ------------------------------ Cost and Conflicts ------------------------------
def check_conflict(agents):
    pos_time={}  # track positions and times
    conflicts=[]

    for agent_index, agent in enumerate(agents):
        for time, pos in enumerate(agent.path):
            if (pos, time) not in pos_time:
                pos_time[(pos, time)]=[]
            pos_time[(pos, time)].append(agent_index)

    # Detect conflicts
    for (position, time), agent_indices in pos_time.items():
        if len(agent_indices)>1:  # Conflict detected
            conflicts.append((position, time, agent_indices))

    return pos_time, conflicts


def cost(agents):
    total_cost=0
    pos_time=check_conflict(agents)[0] # Array with keys position and time
    collision_penalty=10000  # Large penalty for collisions

    for agent in agents:
        total_cost+=len(agent.path)

    for (pos, time), agent_indices in pos_time.items():
        if len(agent_indices)>1:  # Conflict detected
            total_cost+=collision_penalty*(len(agent_indices)-1)

    return total_cost




# ------------------------------ Neighborhood ------------------------------
def neighborhood(agents):
    new_agents=[Agent(agent.start, agent.goal) for agent in agents]  # Initialize new agents
    for agent, new_agent in zip(agents, new_agents):
        new_agent.path=agent.path[:]  # Copy existing paths

    conflicts=check_conflict(agents)[1]

    if conflicts:
        # Get the first conflict
        conflict_position, conflict_time, agent_indices =conflicts[0]
        
        # Only modify the first agent in conflict
        agent_index=agent_indices[0]
        agent=agents[agent_index]
        new_agent=new_agents[agent_index]

        if conflict_time>0:  
            index=conflict_time-1  # Point just before the conflict
            current=new_agent.path[index]
            new_path_segment=[current]  # Generate new steps from current point to the goal

            while new_path_segment[-1] != new_agent.goal:
                neighbors=[pos for pos in get_neighbors(new_path_segment[-1]) if pos!=conflict_position]
                if not neighbors:
                    neighbors= get_neighbors(new_path_segment[-1])  
                    next_step=direction_towards_goal(new_path_segment[-1], neighbors, new_agent.goal)
                    new_path_segment.append(next_step)

                # Update the path of the agent
                new_agent.path=new_agent.path[:index]+new_path_segment


    else:
        for _, new_agent in enumerate(new_agents):
            if len(new_agent.path) > 2:
                # Randomly choose an index in the path (excluding start and goal positions)
                index=random.randint(1, len(new_agent.path) - 2)
                current=new_agent.path[index]
                new_path_segment=[current]  # Generate new steps from current point to the goal
                visited=set(new_agent.path[:index + 1])  # Include previously visited positions up to current index

                while new_path_segment[-1] != new_agent.goal:
                    neighbors=[pos for pos in get_neighbors(new_path_segment[-1]) if pos not in visited]
                    if not neighbors:
                        neighbors= get_neighbors(new_path_segment[-1])  # If no unvisited neighbors, allow revisiting
                    next_step=direction_towards_goal(new_path_segment[-1], neighbors, new_agent.goal)
                    new_path_segment.append(next_step)
                    visited.add(next_step)

                # Update the path of the agent
                new_agent.path=new_agent.path[:index]+new_path_segment
    return new_agents


# ------------------------------ Simulated Annealing ------------------------------
# Simulated annealing algorithm
def simulated_annealing(agents, max_iter, initial_temp, cooling_rate):
    start_time = time.time()
    
    current_solution = starting_solution(agents)
    current_cost = cost(current_solution)
    
    print("\nInitial solution:")
    for i, agent in enumerate(current_solution):
            print(f"Agent {i+1} path: {agent.path}")

    print(f"Initial cost: {current_cost}")
    
    best_solution = current_solution
    best_cost = current_cost
    temperature = initial_temp
    best_iteration = 0

    for t in range(max_iter):
        new_solution = neighborhood(current_solution)
        new_cost = cost(new_solution)

        if new_cost < current_cost or math.exp((current_cost - new_cost) / temperature) > random.random():
            current_solution = new_solution
            current_cost = new_cost

            if new_cost < best_cost:
                best_solution = new_solution
                best_cost = new_cost
                best_iteration = t

        temperature *= cooling_rate

    end_time = time.time()
    duration = end_time - start_time
    return best_solution, best_cost, best_iteration, duration



# ------------------------------ Printing ------------------------------
def print_matrix(matrix):
    length = max(len(j) for row in matrix for j in row)
    
    def padd(i, length):
        return i.ljust(length)
    
    updated_matrix = [[padd(i, length) for i in j] for j in matrix]
    
    for row in updated_matrix:
        print(' '.join(row))


def random_pos(grid_size, num_agents):
    positions = set() # can't have same postions (start,goal) as start/goal so we use set

    while len(positions)<num_agents * 2:  # need start+goal for all agents
        pos = (random.randint(0, grid_size - 1), random.randint(0, grid_size - 1))
        positions.add(pos)
    positions = list(positions)

    return [(positions[i],positions[i+num_agents]) for i in range(num_agents)]



# ------------------------------ Global variables ------------------------------

# Test parameters
max_iter = 200000
initial_temp = 1000
cooling_rate = 0.99

# Random parameters for matrix
grid_size = random.randint(10,10)  # Random grid size between 10 and 20
#num_agents = random.randint(grid_size//3, int(grid_size/1.5))  # Number of agents based on grid size
num_agents=5

# ------------------------------ Main ------------------------------

def main():
    # Update agents and matrix
    starts_goals = random_pos(grid_size, num_agents)
    agents = [Agent(start, goal) for start, goal in starts_goals]

    starting_matrix = [[". " for _ in range(grid_size)] for _ in range(grid_size)]


    for i, agent in enumerate(agents):
        start = agent.start
        goal = agent.goal
        starting_matrix[start[0]][start[1]] = str(i+1)
        starting_matrix[goal[0]][goal[1]] = str(i+1)+"*"


    # Printing
    print(f"Grid size: {grid_size}x{grid_size}, Number of agents: {num_agents}")
    print("Initial Matrix:")
    print_matrix(starting_matrix)


    # Run simulated annealing
    best_solution, best_cost, best_iteration, duration = simulated_annealing(agents, max_iter, initial_temp, cooling_rate)


    # Printing the best solution found
    print(f"\nBest solution cost: {best_cost} found at iteration {best_iteration}")
    print(f"Time taken: {duration:.2f} seconds")
    for i, agent in enumerate(best_solution):
        print(f"Agent {i+1} path: {agent.path}")


if __name__ == "__main__":
    main()

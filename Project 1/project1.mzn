int: N = 5;
int: numAgents = 3;
int: T = 1000;
set of int: Nodes = 1..N*N;

% Decision variables
array[1..T, 1..numAgents] of var Nodes: position;
array[1..T, 1..numAgents] of var 0..4: move_index;  % Index for the move
array[1..5] of int: moves = [-N, -1, 0, 1, N];  % Corresponding move values

% Randomly generate starting and ending positions
array[1..numAgents] of var Nodes: starts = [1, 12, 20];
array[1..numAgents] of var Nodes: targets = [25, 3, 10]; 

% Constraints

% Initial and final positions
constraint forall(i in 1..numAgents) (
    position[1, i] = starts[i] /\ position[T, i] = targets[i]
);

% Movement constraints - agents can move one step in all four directions
constraint forall(t in 1..T-1, a in 1..numAgents) (
    move_index[t, a] in 0..4
);

% Update positions based on movement
constraint forall(t in 1..T-1, a in 1..numAgents) (
    position[t+1, a] = position[t, a] + moves[move_index[t, a]]
);

% Collision avoidance
constraint forall(t in 1..T-1, a1, a2 in 1..numAgents where a1 < a2) (
    position[t, a1] + moves[move_index[t, a1]] != position[t, a2] + moves[move_index[t, a2]]
);

% Time taken for each agent to reach its target
array[1..numAgents] of var 1..T: timeTaken;
constraint forall(a in 1..numAgents) (
    timeTaken[a] = min([t | t in 1..T where position[t, a] = targets[a]])
);

% Objective: Minimize makespan
var int: makespan = max(timeTaken);

solve minimize makespan;

% Output start positions, target positions, and positions over time
output ["Start positions:"] ++ [show(starts[a]) ++ " " | a in 1..numAgents] ++
       ["\nTarget positions:"] ++ [show(targets[a]) ++ " " | a in 1..numAgents] ++
       ["\nPositions:\n"] ++ [ "time = " ++ show(t) ++ " " ++ show(position[t, a]) ++ "\n" | t in 1..T, a in 1..numAgents];

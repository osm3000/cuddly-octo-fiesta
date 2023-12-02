Quick writeup of the experiment results. An experiment will be marked by a git tag.


# v0.1
- Date: 02-12-2023
- Exp: tank vs tank
- Objective: train two tanks/agents to fight each other
- Result: 
  - The two tanks collaborated in order to win the game!
- Comments:
  - The game design was incorrect. In the stress to get a good setup up and running, the reward function didn't lead to a zero-sum game. 
    - The reward is set to give a win to neural network if a game is won.
    - Both tanks used the same neural network.
  - One tank would go to the corner, the other would follow and shoot the first one.
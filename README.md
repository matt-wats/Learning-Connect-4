# Learning-Connect-4

Uses an Agent (Convolutional Neural Network) to evaluate Connect 4 Positions, to guide a Monte-Carlo Tree Search (MCTS) in playing Connect 4.
It plays a game against itself, then trains the Agent's parameters to better predict the game's final scoring.

Loosely based off of, and very much simplified version of, AlphaZero.


### Future

We could implement weightings in the loss functions, for the Evaluator to focus more on becoming adept on learning end states, as learning the beginning moves
are going to be more difficult to accurately learn as their associated value will change drastically throughout training.

Each game takes a long time to play, so speeding up this process is very important. We could change the process of finding new moves from a search to having possible 
moves as a property of a board state, which should drastically improve search times.

The MCTS will almost every time repeat searching the same board state in different branches, so we could use a graph search instead to prevent this.

Much more difficult idea: We could have the agent play "probabilistic" games, in which each move considered has an assigned probability and thus an assigned outcome,
which may improve learning speed.

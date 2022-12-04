# Learning-Connect-4

Uses an Agent (Convolutional Neural Network) to evaluate Connect 4 Positions, to guide a Monte-Carlo Tree Search (MCTS) in playing Connect 4.
It plays a game against itself, then trains the Agent's parameters to better predict the game's final scoring.

Loosely based off of, and very much simplified version of, AlphaZero.

### Previous Project

My goal for this project was to improve upon a previous project of mine: https://openprocessing.org/sketch/911295, which was made in javascript, which used a custom 
evaluation function in a minimax search. The methods used are very different, but I really was just interested in creating and learning games.

## Future Improvements

The most imminent improvement to be made is a way to avoid the failure mode: all evaluations collapsing to 0.5 (a tie), rather than trying to actually predict the 
outcome of the game. We could add more loss weight when the outcome of the game isn't a draw, to make actual predictions more important. It would be interesting 
to investiate this further and find better solutions.

We could implement weightings in the loss functions, for the Evaluator to focus more on becoming adept on learning end states, as learning the beginning moves
are going to be more difficult to accurately learn as their associated value will change drastically throughout training.

Each game takes a long time to play, so speeding up this process is very important. We could change the process of finding new moves from a search to having possible 
moves as a property of a board state, which should drastically improve search times.

The MCTS will almost every time repeat searching the same board state in different branches, so we could use a graph search instead to prevent this.

We could have the agent assign probabilities to each possible action, then have the next move be chosen according to those probabilities, and then train the probability
predictions to be which is actually chosen (or something similar).

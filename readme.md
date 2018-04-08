# Simple RL-based selfplay experiments framework

## Description
Framework design to perform simple selfplay experiments using only reinforcement learning (no planning - i.e. no MCTS, etc.).
At the moment both discrete action space and continuous action space are implemented, however discrete action space is tested a bit more extensively.
The framework is quite raw, and hyperparameters are largely untuned, however it does converge to some sane results for simple tests like Cartpole, Acrobot, and Pendulum (that one is contunuous action space).
The custom environment - TicTacToe selfplay - learns to play (although quite naive) within 100k iterations.
TODO:
* tons of debugging
* multithreaded rollout generation
* more agents (only clipped-surrogate PPO at the moment)


## License
[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International Public License](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode)

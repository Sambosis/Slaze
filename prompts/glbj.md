Create a Python application using Pygame and reinforcement learning that simulates and learns optimal bet sizing in blackjack. Here are the specific requirements:







1. Core Components:



- Implement a blackjack game engine using Pygame for visualization



- Create a reinforcement learning agent that focuses on bet sizing optimization



- Enforce basic strategy for all playing decisions (hit, stand, double, split)



- Track bankroll and maintain betting statistics







2. Game Logic Requirements:



- The agent must strictly follow basic strategy for all playing decisions



- Implement standard blackjack rules (dealer stands on soft 17, blackjack pays 3:2)



- Track the running count and true count for bet sizing decisions



- Maintain a betting spread based on the count and bankroll







3. Reinforcement Learning Components:



- State space should include: current bankroll, true count, and bet sizing history



- Action space should be a discrete set of possible bet sizes



- Reward function should be based on bankroll changes



- Use Q-learning or SARSA for the learning algorithm



- Implement epsilon-greedy exploration strategy







4. Display and Visualization:



- Show a full game playthrough every 50 games, :



- Update and display training statistics every 10 games, including:



  * Average return per hand



  * Bankroll graph



  * Win/loss ratio



  * Average bet size



  * Maximum drawdown



  * Current learning rate and epsilon value







5. Technical Requirements:



- Use Pygame for the graphical interface



- Implement proper separation of concerns (game logic, RL agent, visualization)



- Include save/load functionality for the trained model



- Add configuration options for:



  * Initial bankroll



  * Betting limits



  * Number of decks



  * Training parameters







Please provide the complete implementation with appropriate comments and documentation.
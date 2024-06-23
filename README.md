# Deep Learning on Ramsey Graphs

## Naive Policy Agent

Create a new neural network with create_agent.py
Create game experience to learn from with create_experience.py
Train the agent with train_agent.py
Evaluate the agent with eval_agent.py

Upon playing 10,000 games, the trained policy agent showed the results:
No red 3-cliques: 1885/10000
No red or blue 3-cliques: 254/10000

These results do not suggest an improvement over random selection of edges
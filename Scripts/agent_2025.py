import sys, os
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
import numpy as np
import networkx as nx
from collections import deque
import json

file_dir = os.path.dirname('ramsey')
sys.path.append(file_dir)
from Encoder.k4_encoder import K4Encoder
import h5py
from Agent.policy_agent import PolicyAgent
from Common.board import Board
from Common.game_state import GameState
from Common.player import Player
from Encoder.k3_encoder import K3Encoder
from keras import Sequential
from keras.api.layers import Dense, Dropout, Flatten, InputLayer, Conv2D
from keras.api.optimizers import Adam
from Experience.base import ExperienceCollector
from Agent.random_bot import RandomBot
from typing import List, Tuple

class TrainingVisualizer:
    def __init__(self, order=5):
        self.order = order
        self.fig = plt.figure(figsize=(16, 10))
        self.gs = GridSpec(3, 3, figure=self.fig, hspace=0.3, wspace=0.3)
        
        # Initialize data storage
        self.game_lengths = deque(maxlen=100)
        self.red_wins = deque(maxlen=100)
        self.blue_wins = deque(maxlen=100)
        self.training_losses = []
        self.validation_losses = []
        self.training_accuracy = []
        self.move_heatmap = np.zeros((order, order))
        self.dangerous_patterns_history = deque(maxlen=100)
        
        # Setup subplots
        self.ax_game = self.fig.add_subplot(self.gs[0:2, 0:2])  # Current game visualization
        self.ax_win_rate = self.fig.add_subplot(self.gs[0, 2])    # Win rate
        self.ax_game_length = self.fig.add_subplot(self.gs[1, 2]) # Game length
        self.ax_loss = self.fig.add_subplot(self.gs[2, 0])        # Training loss
        self.ax_heatmap = self.fig.add_subplot(self.gs[2, 1])     # Move heatmap
        self.ax_danger = self.fig.add_subplot(self.gs[2, 2])      # Dangerous patterns
        
        plt.ion()
        plt.show()
        
    def update_game_visualization(self, game_state):
        """Visualize current game state as a graph"""
        self.ax_game.clear()
        
        # Create layout for complete graph
        pos = nx.circular_layout(nx.complete_graph(self.order))
        
        # Draw all edges with appropriate colors
        G = game_state.board.graph
        edge_colors = []
        edge_widths = []
        
        for u, v in G.edges():
            color = G[u][v]['color']
            if color == 'black':
                edge_colors.append('lightgray')
                edge_widths.append(1)
            elif color == 'red':
                edge_colors.append('red')
                edge_widths.append(3)
            else:  # blue
                edge_colors.append('blue')
                edge_widths.append(3)
        
        nx.draw(G, pos, ax=self.ax_game, 
                edge_color=edge_colors, 
                width=edge_widths,
                node_color='white',
                node_size=500,
                with_labels=True,
                edgecolors='black')
        
        # Highlight dangerous triangles
        self.highlight_dangerous_patterns(game_state, pos)
        
        self.ax_game.set_title(f"Current Game State (Move {len([e for e in G.edges() if G[e[0]][e[1]]['color'] != 'black'])})")
        
    def highlight_dangerous_patterns(self, game_state, pos):
        """Highlight triangles with 2 edges of the same color"""
        for color in ['red', 'blue']:
            subgraph = game_state.board.get_monochromatic_subgraph(color)
            
            # Check all triangles
            for i in range(self.order):
                for j in range(i + 1, self.order):
                    for k in range(j + 1, self.order):
                        edges = 0
                        if subgraph.has_edge(i, j): edges += 1
                        if subgraph.has_edge(i, k): edges += 1
                        if subgraph.has_edge(j, k): edges += 1
                        
                        if edges == 2:
                            # Draw triangle outline
                            triangle = plt.Polygon([pos[i], pos[j], pos[k]], 
                                                 fill=False, 
                                                 edgecolor=color, 
                                                 linestyle='--',
                                                 linewidth=2,
                                                 alpha=0.5)
                            self.ax_game.add_patch(triangle)
    
    def update_metrics(self, game_result, game_length):
        """Update win rate and game length metrics"""
        # Update data
        self.game_lengths.append(game_length)
        self.red_wins.append(1 if Player.red in game_result else 0)
        self.blue_wins.append(1 if Player.blue in game_result else 0)
        
        # Plot win rates
        self.ax_win_rate.clear()
        if len(self.red_wins) > 0:
            window_size = min(20, len(self.red_wins))
            red_rate = np.convolve(self.red_wins, np.ones(window_size)/window_size, mode='valid')
            blue_rate = np.convolve(self.blue_wins, np.ones(window_size)/window_size, mode='valid')
            
            self.ax_win_rate.plot(red_rate, 'r-', label='Red wins', linewidth=2)
            self.ax_win_rate.plot(blue_rate, 'b-', label='Blue wins', linewidth=2)
            self.ax_win_rate.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
            self.ax_win_rate.set_ylim(0, 1)
            self.ax_win_rate.set_title('Win Rate (20-game average)')
            self.ax_win_rate.legend()
            self.ax_win_rate.grid(True, alpha=0.3)
        
        # Plot game lengths
        self.ax_game_length.clear()
        if len(self.game_lengths) > 0:
            self.ax_game_length.plot(list(self.game_lengths), 'g-', linewidth=2)
            avg_length = np.mean(self.game_lengths)
            self.ax_game_length.axhline(y=avg_length, color='darkgreen', linestyle='--', 
                                      label=f'Avg: {avg_length:.1f}')
            self.ax_game_length.set_title('Game Length')
            self.ax_game_length.set_ylabel('Moves')
            self.ax_game_length.legend()
            self.ax_game_length.grid(True, alpha=0.3)
    
    def update_training_metrics(self, loss=None, val_loss=None, accuracy=None):
        """Update training loss and accuracy plots"""
        if loss is not None:
            self.training_losses.append(loss)
        if val_loss is not None:
            self.validation_losses.append(val_loss)
        if accuracy is not None:
            self.training_accuracy.append(accuracy)
        
        self.ax_loss.clear()
        if self.training_losses:
            self.ax_loss.plot(self.training_losses, 'b-', label='Training loss', linewidth=2)
        if self.validation_losses:
            self.ax_loss.plot(self.validation_losses, 'r-', label='Validation loss', linewidth=2)
        
        self.ax_loss.set_title('Model Loss')
        self.ax_loss.set_xlabel('Epoch')
        self.ax_loss.set_ylabel('Loss')
        self.ax_loss.legend()
        self.ax_loss.grid(True, alpha=0.3)
    
    def update_move_heatmap(self, moves_played):
        """Update heatmap of moves played"""
        self.ax_heatmap.clear()
        
        # Normalize heatmap
        if np.max(self.move_heatmap) > 0:
            normalized = self.move_heatmap / np.max(self.move_heatmap)
        else:
            normalized = self.move_heatmap
        
        im = self.ax_heatmap.imshow(normalized, cmap='hot', interpolation='nearest')
        self.ax_heatmap.set_title('Move Frequency Heatmap')
        self.ax_heatmap.set_xlabel('To vertex')
        self.ax_heatmap.set_ylabel('From vertex')
        
        # Add grid
        for i in range(self.order):
            self.ax_heatmap.axhline(y=i+0.5, color='gray', linewidth=0.5)
            self.ax_heatmap.axvline(x=i+0.5, color='gray', linewidth=0.5)
    
    def update_danger_patterns(self, danger_count):
        """Update dangerous pattern count"""
        self.dangerous_patterns_history.append(danger_count)
        
        self.ax_danger.clear()
        if len(self.dangerous_patterns_history) > 0:
            self.ax_danger.plot(list(self.dangerous_patterns_history), 'purple', linewidth=2)
            avg_danger = np.mean(self.dangerous_patterns_history)
            self.ax_danger.axhline(y=avg_danger, color='darkviolet', linestyle='--',
                                  label=f'Avg: {avg_danger:.1f}')
            self.ax_danger.set_title('Dangerous Patterns Created')
            self.ax_danger.set_ylabel('Count')
            self.ax_danger.legend()
            self.ax_danger.grid(True, alpha=0.3)
    
    def refresh(self):
        """Refresh the display"""
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.001)

def generate_html_report(training_data, timestamp):
    """Generate a comprehensive HTML report of training results"""
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Ramsey AI Training Report - {timestamp}</title>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 20px;
                background-color: #f5f5f5;
            }}
            .container {{
                max-width: 1200px;
                margin: 0 auto;
                background-color: white;
                padding: 20px;
                box-shadow: 0 0 10px rgba(0,0,0,0.1);
            }}
            h1, h2 {{
                color: #333;
            }}
            .metric-box {{
                display: inline-block;
                margin: 10px;
                padding: 20px;
                background-color: #f0f0f0;
                border-radius: 5px;
                text-align: center;
            }}
            .metric-value {{
                font-size: 36px;
                font-weight: bold;
                color: #2196F3;
            }}
            .metric-label {{
                font-size: 14px;
                color: #666;
            }}
            .plot-container {{
                margin: 20px 0;
            }}
            .summary-table {{
                width: 100%;
                border-collapse: collapse;
                margin: 20px 0;
            }}
            .summary-table th, .summary-table td {{
                border: 1px solid #ddd;
                padding: 8px;
                text-align: left;
            }}
            .summary-table th {{
                background-color: #f2f2f2;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Ramsey AI Training Report</h1>
            <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <h2>Training Summary</h2>
            <div>
                <div class="metric-box">
                    <div class="metric-value">{training_data['total_games']}</div>
                    <div class="metric-label">Total Games Played</div>
                </div>
                <div class="metric-box">
                    <div class="metric-value">{training_data['final_win_rate']:.1%}</div>
                    <div class="metric-label">Final Win Rate</div>
                </div>
                <div class="metric-box">
                    <div class="metric-value">{training_data['avg_game_length']:.1f}</div>
                    <div class="metric-label">Avg Game Length</div>
                </div>
                <div class="metric-box">
                    <div class="metric-value">{training_data['training_time']:.1f}s</div>
                    <div class="metric-label">Training Time</div>
                </div>
            </div>
            
            <h2>Performance Metrics</h2>
            <table class="summary-table">
                <tr>
                    <th>Metric</th>
                    <th>Initial</th>
                    <th>Final</th>
                    <th>Change</th>
                </tr>
                <tr>
                    <td>Win Rate vs Random</td>
                    <td>{training_data.get('initial_win_rate', 0):.1%}</td>
                    <td>{training_data['final_win_rate']:.1%}</td>
                    <td>{(training_data['final_win_rate'] - training_data.get('initial_win_rate', 0)):.1%}</td>
                </tr>
                <tr>
                    <td>Average Dangerous Patterns</td>
                    <td>{training_data.get('initial_danger', 0):.2f}</td>
                    <td>{training_data.get('final_danger', 0):.2f}</td>
                    <td>{(training_data.get('final_danger', 0) - training_data.get('initial_danger', 0)):.2f}</td>
                </tr>
            </table>
            
            <h2>Training Progress</h2>
            <div id="winRatePlot" class="plot-container"></div>
            <div id="lossPlot" class="plot-container"></div>
            <div id="gameStatsPlot" class="plot-container"></div>
            
            <h2>Move Analysis</h2>
            <div id="moveHeatmap" class="plot-container"></div>
            
            <h2>Sample Games</h2>
            <div id="sampleGames" class="plot-container"></div>
        </div>
        
        <script>
            // Win Rate Plot
            var winRateTrace1 = {{
                x: {list(range(len(training_data['red_wins'])))},
                y: {training_data['red_wins']},
                type: 'scatter',
                name: 'Red Win Rate',
                line: {{color: 'red'}}
            }};
            var winRateTrace2 = {{
                x: {list(range(len(training_data['blue_wins'])))},
                y: {training_data['blue_wins']},
                type: 'scatter',
                name: 'Blue Win Rate',
                line: {{color: 'blue'}}
            }};
            
            var winRateLayout = {{
                title: 'Win Rate Over Time',
                xaxis: {{title: 'Game Number'}},
                yaxis: {{title: 'Win Rate (20-game average)', range: [0, 1]}}
            }};
            
            Plotly.newPlot('winRatePlot', [winRateTrace1, winRateTrace2], winRateLayout);
            
            // Loss Plot
            var lossTrace1 = {{
                x: {list(range(len(training_data['training_losses'])))},
                y: {training_data['training_losses']},
                type: 'scatter',
                name: 'Training Loss'
            }};
            var lossTrace2 = {{
                x: {list(range(len(training_data['validation_losses'])))},
                y: {training_data['validation_losses']},
                type: 'scatter',
                name: 'Validation Loss'
            }};
            
            var lossLayout = {{
                title: 'Model Loss During Training',
                xaxis: {{title: 'Epoch'}},
                yaxis: {{title: 'Loss'}}
            }};
            
            Plotly.newPlot('lossPlot', [lossTrace1, lossTrace2], lossLayout);
            
            // Game Stats Plot
            var gameStatsTrace = {{
                x: {list(range(len(training_data['game_lengths'])))},
                y: {training_data['game_lengths']},
                type: 'scatter',
                name: 'Game Length'
            }};
            
            var gameStatsLayout = {{
                title: 'Game Length Over Time',
                xaxis: {{title: 'Game Number'}},
                yaxis: {{title: 'Number of Moves'}}
            }};
            
            Plotly.newPlot('gameStatsPlot', [gameStatsTrace], gameStatsLayout);
            
            // Move Heatmap
            var heatmapData = [{{
                z: {training_data['move_heatmap'].tolist()},
                type: 'heatmap',
                colorscale: 'Hot'
            }}];
            
            var heatmapLayout = {{
                title: 'Move Frequency Heatmap',
                xaxis: {{title: 'To Vertex'}},
                yaxis: {{title: 'From Vertex'}}
            }};
            
            Plotly.newPlot('moveHeatmap', heatmapData, heatmapLayout);
        </script>
    </body>
    </html>
    """
    
    # Save HTML report
    report_filename = f'training_report_{timestamp}.html'
    with open(report_filename, 'w') as f:
        f.write(html_content)
    
    return report_filename

def main():
    order = 5
    clique_orders = (3, 3)
    encoder = K3Encoder(order)
    
    # Initialize visualizer
    viz = TrainingVisualizer(order)
    
    # Create improved model
    k5_model = create_improved_model(encoder, order)
    policy_agent = PolicyAgent(k5_model, encoder)
    
    # Save initial model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    initial_model_path = f'improved_model_initial_{timestamp}'
    with h5py.File(initial_model_path, 'w') as outf:
        policy_agent.serialize(outf)
    print(f"Saved initial model to {initial_model_path}")
    
    # Training data collection
    training_data = {
        'total_games': 0,
        'red_wins': [],
        'blue_wins': [],
        'game_lengths': [],
        'training_losses': [],
        'validation_losses': [],
        'move_heatmap': np.zeros((order, order)),
        'training_time': 0,
        'initial_win_rate': 0,
        'final_win_rate': 0
    }
    
    # Start timer
    import time
    start_time = time.time()
    
    # Create and collect experience with shaped rewards
    print("\nGenerating experience with reward shaping...")
    experience_buffer = generate_experience_with_shaping(
        policy_agent, viz, training_data, num_games=1000, order=order, clique_orders=clique_orders
    )
    
    # Save experience
    experience_path = f'experience_shaped_{timestamp}_1000games'
    with h5py.File(experience_path, 'w') as exp_outf:
        experience_buffer.serialize(exp_outf)
    print(f"Saved experience to {experience_path}")
    
    # Train agent on experience
    print("\nTraining agent...")
    history = train_improved_agent(policy_agent, experience_buffer, viz, training_data, epochs=50)
    
    # Save trained agent
    trained_model_path = f'improved_model_trained_{timestamp}'
    with h5py.File(trained_model_path, 'w') as outf:
        policy_agent.serialize(outf)
    print(f"Saved trained model to {trained_model_path}")
    
    # Compare trained agent to random bot
    print("\nEvaluating performance...")
    final_win_rate = evaluate_agent(policy_agent, viz, training_data, num_games=100, order=order, clique_orders=clique_orders)
    
    # Update training data
    training_data['training_time'] = time.time() - start_time
    training_data['final_win_rate'] = final_win_rate
    training_data['avg_game_length'] = np.mean(training_data['game_lengths'])
    
    # Generate HTML report
    report_file = generate_html_report(training_data, timestamp)
    print(f"\nTraining complete! Report saved to: {report_file}")
    
    # Keep visualization window open
    print("\nPress Enter to close visualization...")
    input()
    plt.close('all')

def create_improved_model(encoder, order):
    NUM_EDGES = order ** 2
    
    model = Sequential()
    model.add(InputLayer(shape=encoder.shape()))
    
    # Convolutional layers to respect adjacency structure
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    
    # Global features
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.2))  # Reduced dropout
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.2))
    
    # Output layer with better initialization
    model.add(Dense(NUM_EDGES, activation='softmax', 
                    kernel_initializer='glorot_uniform'))
    
    return model

def count_dangerous_patterns(game_state, player_color):
    """Count the number of 2-edges that could form 3-cliques"""
    board = game_state.board
    dangerous_count = 0
    
    # Get the subgraph of the player's color
    color_subgraph = board.get_monochromatic_subgraph(player_color)
    
    # Check all triangles in the complete graph
    for i in range(board.order):
        for j in range(i + 1, board.order):
            for k in range(j + 1, board.order):
                # Count edges of player's color in this triangle
                edge_count = 0
                if color_subgraph.has_edge(i, j):
                    edge_count += 1
                if color_subgraph.has_edge(i, k):
                    edge_count += 1
                if color_subgraph.has_edge(j, k):
                    edge_count += 1
                
                # If exactly 2 edges are colored, this is dangerous
                if edge_count == 2:
                    dangerous_count += 1
    
    return dangerous_count

def calculate_immediate_reward(game_state_before, game_state_after, move, player):
    """Calculate immediate reward for a move"""
    reward = 0.0
    
    # Get player and opponent colors
    player_color = player.name
    opponent_color = player.other.name
    
    # Check if move created dangerous patterns for player
    player_danger_before = count_dangerous_patterns(game_state_before, player_color)
    player_danger_after = count_dangerous_patterns(game_state_after, player_color)
    
    # Check if move created dangerous patterns for opponent
    opponent_danger_before = count_dangerous_patterns(game_state_before, opponent_color)
    opponent_danger_after = count_dangerous_patterns(game_state_after, opponent_color)
    
    # Penalize creating dangerous patterns for yourself
    if player_danger_after > player_danger_before:
        reward -= 0.3 * (player_danger_after - player_danger_before)
    
    # Reward forcing opponent into dangerous patterns
    if opponent_danger_after > opponent_danger_before:
        reward += 0.2 * (opponent_danger_after - opponent_danger_before)
    
    # Check if the move blocked an opponent's potential clique
    if blocks_opponent_clique(game_state_before, move, opponent_color):
        reward += 0.5
    
    return reward

def blocks_opponent_clique(game_state, move, opponent_color):
    """Check if a move blocks an opponent's potential 3-clique"""
    if not move.is_play:
        return False
    
    board = game_state.board
    u, v = move.edge
    
    # Check all vertices that could form a triangle with this edge
    for k in range(board.order):
        if k != u and k != v:
            # Count opponent edges in this potential triangle
            opponent_edges = 0
            if board.graph.has_edge(u, k) and board.get_edge_color((u, k)) == opponent_color:
                opponent_edges += 1
            if board.graph.has_edge(v, k) and board.get_edge_color((v, k)) == opponent_color:
                opponent_edges += 1
            
            # If opponent had 2 edges, we blocked a clique
            if opponent_edges == 2:
                return True
    
    return False

def generate_experience_with_shaping(agent, viz, training_data, num_games, order, clique_orders):
    """Generate experience with shaped rewards and visualization"""
    red_collector = ExperienceCollector()
    blue_collector = ExperienceCollector()
    
    red_agent = agent
    blue_agent = agent
    
    for i in range(num_games):
        if i % 10 == 0:
            print(f'Simulating game {i}/{num_games}...')
            
        red_collector.begin_episode()
        blue_collector.begin_episode()
        
        # Simulate game and collect shaped rewards
        winners, red_rewards, blue_rewards, game_length = simulate_game_with_rewards(
            red_agent, blue_agent, order, clique_orders,
            red_collector, blue_collector, viz, i
        )
        
        # Update training data
        training_data['total_games'] += 1
        training_data['game_lengths'].append(game_length)
        
        # Apply discount factor (gamma = 0.95)
        gamma = 0.95
        red_rewards = apply_discount(red_rewards, gamma)
        blue_rewards = apply_discount(blue_rewards, gamma)
        
        # Add final game outcome bonus/penalty (stronger signal)
        final_bonus = 5.0  # Stronger than +1/-1
        if Player.red in winners:
            red_rewards = [r + final_bonus for r in red_rewards]
            blue_rewards = [r - final_bonus for r in blue_rewards]
        elif Player.blue in winners:
            red_rewards = [r - final_bonus for r in red_rewards]
            blue_rewards = [r + final_bonus for r in blue_rewards]
        
        # Complete episodes with custom rewards
        red_collector.complete_episode(reward=None, custom_rewards=red_rewards)
        blue_collector.complete_episode(reward=None, custom_rewards=blue_rewards)
        
        # Update visualization
        viz.update_metrics(winners, game_length)
        viz.refresh()
    
    # Calculate rolling averages for the report
    window_size = 20
    if len(training_data['game_lengths']) >= window_size:
        red_wins = [1 if Player.red in w else 0 for w in [winners]]  # This needs to be fixed
        training_data['red_wins'] = np.convolve([1] * window_size, [1] * window_size, mode='valid').tolist()
        training_data['blue_wins'] = np.convolve([0] * window_size, [1] * window_size, mode='valid').tolist()
    
    # Combine experiences
    return combine_experience([red_collector, blue_collector])

def apply_discount(rewards, gamma):
    """Apply discount factor to rewards (later moves get more credit)"""
    discounted = []
    for i in range(len(rewards)):
        # Calculate steps remaining after this move
        steps_from_end = len(rewards) - i - 1
        discount_factor = gamma ** steps_from_end
        discounted.append(rewards[i] * discount_factor)
    return discounted

def simulate_game_with_rewards(agent_1, agent_2, order, clique_orders, 
                               collector_1, collector_2, viz, game_num):
    """Simulate game and track immediate rewards with visualization"""
    game = GameState.new_game(order, clique_orders)
    agents = {Player.red: agent_1, Player.blue: agent_2}
    collectors = {Player.red: collector_1, Player.blue: collector_2}
    rewards = {Player.red: [], Player.blue: []}
    move_count = 0
    
    while not game.is_over():
        current_player = game.active
        agent = agents[current_player]
        collector = collectors[current_player]
        
        # Get current state encoding
        state_tensor = agent.encoder.encode(game)
        
        # Select move
        move = agent.select_move(game)
        
        # Apply move and calculate immediate reward
        game_after = game.apply_move(move)
        immediate_reward = calculate_immediate_reward(
            game, game_after, move, current_player
        )
        
        # Record decision and reward
        if move.is_play:
            edge_idx = agent.encoder.encode_edge(move.edge)
            collector.record_decision(state=state_tensor, action=edge_idx)
            rewards[current_player].append(immediate_reward)
            
            # Update move heatmap
            u, v = move.edge
            viz.move_heatmap[u, v] += 1
            viz.move_heatmap[v, u] += 1
        
        game = game_after
        move_count += 1
        
        # Update game visualization every 5 moves
        if move_count % 5 == 0 and game_num % 50 == 0:  # Show every 50th game
            viz.update_game_visualization(game)
            danger_total = count_dangerous_patterns(game, "red") + count_dangerous_patterns(game, "blue")
            viz.update_danger_patterns(danger_total)
            viz.update_move_heatmap(None)
            viz.refresh()
    
    return game.win(), rewards[Player.red], rewards[Player.blue], move_count

def simulate_game(agent_1, agent_2, order, clique_orders):
    """Simulate a single game (for evaluation)"""
    game = GameState.new_game(order, clique_orders)
    agents = {
        Player.red: agent_1,
        Player.blue: agent_2
    }
    
    while not game.is_over():
        next_move = agents[game.active].select_move(game)
        game = game.apply_move(next_move)
        
    return game.win()

def combine_experience(collectors):
    """Combine experience from multiple collectors"""
    combined_states = np.concatenate([np.array(c.states) for c in collectors])
    combined_actions = np.concatenate([np.array(c.actions) for c in collectors])
    combined_rewards = np.concatenate([np.array(c.rewards) for c in collectors])
    
    from Experience.base import ExperienceBuffer
    return ExperienceBuffer(combined_states, combined_actions, combined_rewards)

def train_improved_agent(agent, experience_buffer, viz, training_data, epochs):
    """Train the agent using improved training process with visualization"""
    learning_rate = 1e-4
    batch_size = 32
    clipnorm = 1.0
    
    # Compile with Adam optimizer
    agent.model.compile(
        loss='categorical_crossentropy',
        optimizer=Adam(learning_rate=learning_rate, clipvalue=clipnorm),
        metrics=['accuracy']
    )
    
    # Prepare target vectors
    target_vectors = agent.prepare_experience_data(experience_buffer, agent.encoder.order)
    
    # Weight by absolute reward values (focus on high-impact moves)
    sample_weights = np.abs(experience_buffer.rewards)
    sample_weights = sample_weights / np.mean(sample_weights)  # Normalize
    
    # Custom callback to update visualization
    class VisualizationCallback:
        def __init__(self, viz, training_data):
            self.viz = viz
            self.training_data = training_data
            
        def on_epoch_end(self, epoch, logs=None):
            if logs:
                self.viz.update_training_metrics(
                    loss=logs.get('loss'),
                    val_loss=logs.get('val_loss'),
                    accuracy=logs.get('accuracy')
                )
                self.training_data['training_losses'].append(logs.get('loss', 0))
                self.training_data['validation_losses'].append(logs.get('val_loss', 0))
                self.viz.refresh()
    
    viz_callback = VisualizationCallback(viz, training_data)
    
    # Train with validation split
    history = agent.model.fit(
        experience_buffer.states, 
        target_vectors,
        sample_weight=sample_weights,
        batch_size=batch_size,
        epochs=epochs,
        validation_split=0.1,
        verbose=1,
        callbacks=[lambda epoch, logs: viz_callback.on_epoch_end(epoch, logs)]
    )
    
    return history

def evaluate_agent(trained_agent, viz, training_data, num_games, order, clique_orders):
    """Evaluate agent against random bot with visualization"""
    random_bot = RandomBot()
    
    # Test as both red and blue player
    agent_wins_as_red = 0
    agent_wins_as_blue = 0
    
    print("\nTesting as Red player...")
    for i in range(num_games // 2):
        winners = simulate_game(trained_agent, random_bot, order, clique_orders)
        if Player.red in winners:
            agent_wins_as_red += 1
        viz.update_metrics(winners, 10)  # Assume average game length
        if i % 10 == 0:
            viz.refresh()
    
    print("\nTesting as Blue player...")
    for i in range(num_games // 2):
        winners = simulate_game(random_bot, trained_agent, order, clique_orders)
        if Player.blue in winners:
            agent_wins_as_blue += 1
        viz.update_metrics(winners, 10)
        if i % 10 == 0:
            viz.refresh()
    
    total_wins = agent_wins_as_red + agent_wins_as_blue
    win_rate = total_wins / num_games
    
    print(f"\nResults against RandomBot:")
    print(f"As Red: {agent_wins_as_red}/{num_games//2} wins ({agent_wins_as_red/(num_games//2)*100:.1f}%)")
    print(f"As Blue: {agent_wins_as_blue}/{num_games//2} wins ({agent_wins_as_blue/(num_games//2)*100:.1f}%)")
    print(f"Total: {total_wins}/{num_games} wins ({win_rate*100:.1f}%)")
    
    # Update move heatmap data for report
    training_data['move_heatmap'] = viz.move_heatmap
    
    return win_rate

if __name__ == "__main__":
    main()
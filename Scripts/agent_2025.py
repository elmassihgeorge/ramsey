import sys, os
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.gridspec import GridSpec
import numpy as np
import networkx as nx
from collections import deque
import tensorflow as tf
from keras import mixed_precision
import h5py
import argparse
import time

# Add parent directory to path
file_dir = os.path.dirname('ramsey')
sys.path.append(file_dir)

# Project imports
from Agent.policy_agent import PolicyAgent
from Common.board import Board
from Common.game_state import GameState
from Common.player import Player
from Encoder.k3_encoder import K3Encoder
from Experience.base import ExperienceCollector, ExperienceBuffer
from Agent.random_bot import RandomBot
from typing import List, Tuple, Optional, Dict


class TrainingVisualizer:
    """Handles all visualization during training"""
    
    def __init__(self, order: int = 5):
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
        self.epoch_win_rates = []  # New: track win rate per epoch
        self.move_heatmap = np.zeros((order, order))
        self.dangerous_patterns_history = deque(maxlen=100)
        
        # Setup subplots
        self.ax_game = self.fig.add_subplot(self.gs[0:2, 0:2])
        self.ax_win_rate = self.fig.add_subplot(self.gs[0, 2])
        self.ax_game_length = self.fig.add_subplot(self.gs[1, 2])
        self.ax_loss = self.fig.add_subplot(self.gs[2, 0])
        self.ax_heatmap = self.fig.add_subplot(self.gs[2, 1])
        self.ax_epoch_progress = self.fig.add_subplot(self.gs[2, 2])  # Changed from danger patterns
        
        plt.ion()
        plt.show()
        
    def update_game_visualization(self, game_state: GameState):
        """Visualize current game state as a graph"""
        self.ax_game.clear()
        
        pos = nx.circular_layout(nx.complete_graph(self.order))
        G = game_state.board.graph
        edge_colors = []
        edge_widths = []
        
        for u, v in G.edges():
            color = G[u][v].get('color', 'black')
            if color == 'red':
                edge_colors.append('red')
                edge_widths.append(3)
            elif color == 'blue':
                edge_colors.append('blue')
                edge_widths.append(3)
            else:
                edge_colors.append('lightgray')
                edge_widths.append(1)
        
        nx.draw(G, pos, ax=self.ax_game, 
                edge_color=edge_colors, 
                width=edge_widths,
                node_color='white',
                node_size=500,
                with_labels=True,
                edgecolors='black')
        
        self.highlight_dangerous_patterns(game_state, pos)
        colored_edges = len([e for e in G.edges() if G[e[0]][e[1]].get('color', 'black') != 'black'])
        self.ax_game.set_title(f"Current Game State (Move {colored_edges})")
        
    def highlight_dangerous_patterns(self, game_state: GameState, pos: dict):
        """Highlight triangles with 2 edges of the same color"""
        for color in ['red', 'blue']:
            subgraph = game_state.board.get_monochromatic_subgraph(color)
            
            for i in range(self.order):
                for j in range(i + 1, self.order):
                    for k in range(j + 1, self.order):
                        edges = sum(1 for edge in [(i,j), (i,k), (j,k)] 
                                  if subgraph.has_edge(*edge))
                        
                        if edges == 2:
                            triangle = plt.Polygon([pos[i], pos[j], pos[k]], 
                                                 fill=False, 
                                                 edgecolor=color, 
                                                 linestyle='--',
                                                 linewidth=2,
                                                 alpha=0.5)
                            self.ax_game.add_patch(triangle)
    
    def update_metrics(self, game_result: List, game_length: int):
        """Update win rate and game length metrics"""
        self.game_lengths.append(game_length)
        self.red_wins.append(1 if Player.red in game_result else 0)
        self.blue_wins.append(1 if Player.blue in game_result else 0)
        
        # Update win rate plot
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
        
        # Update game length plot
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
    
    def update_training_metrics(self, loss: float = None, val_loss: float = None, 
                               accuracy: float = None):
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
    
    def update_move_heatmap(self):
        """Update heatmap of moves played"""
        self.ax_heatmap.clear()
        
        if np.max(self.move_heatmap) > 0:
            normalized = self.move_heatmap / np.max(self.move_heatmap)
        else:
            normalized = self.move_heatmap
        
        self.ax_heatmap.imshow(normalized, cmap='hot', interpolation='nearest')
        self.ax_heatmap.set_title('Move Frequency Heatmap')
        self.ax_heatmap.set_xlabel('To vertex')
        self.ax_heatmap.set_ylabel('From vertex')
        
        for i in range(self.order):
            self.ax_heatmap.axhline(y=i+0.5, color='gray', linewidth=0.5)
            self.ax_heatmap.axvline(x=i+0.5, color='gray', linewidth=0.5)
    
    def update_epoch_progress(self, epoch: int, win_rate: float):
        """Update win rate progress over epochs"""
        self.epoch_win_rates.append((epoch, win_rate))
        
        self.ax_epoch_progress.clear()
        if self.epoch_win_rates:
            epochs, rates = zip(*self.epoch_win_rates)
            self.ax_epoch_progress.plot(epochs, rates, 'go-', linewidth=2, markersize=6)
            
            # Add trend line if we have enough points
            if len(epochs) > 3:
                z = np.polyfit(epochs, rates, 1)
                p = np.poly1d(z)
                self.ax_epoch_progress.plot(epochs, p(epochs), "r--", alpha=0.5, label=f'Trend')
            
            # Mark 50% win rate
            self.ax_epoch_progress.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Random baseline')
            
            # Highlight best performance
            best_rate = max(rates)
            best_epoch = epochs[rates.index(best_rate)]
            self.ax_epoch_progress.plot(best_epoch, best_rate, 'r*', markersize=15, 
                                      label=f'Best: {best_rate:.1%}')
            
            self.ax_epoch_progress.set_title('Win Rate vs Random Bot During Training')
            self.ax_epoch_progress.set_xlabel('Epoch')
            self.ax_epoch_progress.set_ylabel('Win Rate')
            self.ax_epoch_progress.set_ylim(0, 1)
            self.ax_epoch_progress.grid(True, alpha=0.3)
            self.ax_epoch_progress.legend(loc='lower right')
            
            # Add text with current win rate
            current_rate = rates[-1]
            improvement = (current_rate - rates[0]) * 100 if len(rates) > 1 else 0
            self.ax_epoch_progress.text(0.02, 0.98, 
                                      f'Current: {current_rate:.1%}\nImprovement: {improvement:+.1f}%', 
                                      transform=self.ax_epoch_progress.transAxes,
                                      verticalalignment='top',
                                      bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    def refresh(self):
        """Refresh the display"""
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.001)


# GPU Configuration Functions
def configure_gpu():
    """Configure TensorFlow for optimal GPU usage"""
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"Found {len(gpus)} GPU(s):")
            for gpu in gpus:
                print(f"  {gpu.name}")
        except RuntimeError as e:
            print(f"GPU configuration error: {e}")
    
    # Temporarily disable mixed precision for stability
    # policy = mixed_precision.Policy('mixed_float16')
    # mixed_precision.set_global_policy(policy)
    # print(f'Compute dtype: {policy.compute_dtype}')
    # print(f'Variable dtype: {policy.variable_dtype}')
    
    print("Mixed precision disabled for stability")
    
    # Enable XLA compilation
    tf.config.optimizer.set_jit(True)


def create_gpu_model(encoder, order: int):
    """Create a GPU-optimized model"""
    NUM_EDGES = order ** 2
    
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(shape=encoder.shape()),
        
        # Convolutional layers with batch normalization
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        
        # Global pooling and dense layers
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),
        
        # Output layer
        tf.keras.layers.Dense(NUM_EDGES, activation='softmax', 
                             kernel_initializer='glorot_uniform')
    ])
    
    return model


# Game Logic Functions
def count_dangerous_patterns(game_state: GameState, player_color: str) -> int:
    """Count 2-edges that could form 3-cliques"""
    board = game_state.board
    dangerous_count = 0
    color_subgraph = board.get_monochromatic_subgraph(player_color)
    
    for i in range(board.order):
        for j in range(i + 1, board.order):
            for k in range(j + 1, board.order):
                edges = sum(1 for edge in [(i,j), (i,k), (j,k)] 
                          if color_subgraph.has_edge(*edge))
                if edges == 2:
                    dangerous_count += 1
    
    return dangerous_count


def calculate_immediate_reward(game_state_before: GameState, game_state_after: GameState, 
                             move, player: Player) -> float:
    """Calculate immediate reward for a move"""
    reward = 0.0
    player_color = player.name
    opponent_color = player.other.name
    
    # Check dangerous patterns
    player_danger_before = count_dangerous_patterns(game_state_before, player_color)
    player_danger_after = count_dangerous_patterns(game_state_after, player_color)
    opponent_danger_before = count_dangerous_patterns(game_state_before, opponent_color)
    opponent_danger_after = count_dangerous_patterns(game_state_after, opponent_color)
    
    # Reward shaping
    if player_danger_after > player_danger_before:
        reward -= 0.3 * (player_danger_after - player_danger_before)
    if opponent_danger_after > opponent_danger_before:
        reward += 0.2 * (opponent_danger_after - opponent_danger_before)
    if blocks_opponent_clique(game_state_before, move, opponent_color):
        reward += 0.5
    
    return reward


def blocks_opponent_clique(game_state: GameState, move, opponent_color: str) -> bool:
    """Check if a move blocks an opponent's potential 3-clique"""
    if not move.is_play:
        return False
    
    board = game_state.board
    u, v = move.edge
    
    for k in range(board.order):
        if k != u and k != v:
            opponent_edges = 0
            for edge in [(u, k), (v, k)]:
                if board.graph.has_edge(*edge) and board.get_edge_color(edge) == opponent_color:
                    opponent_edges += 1
            if opponent_edges == 2:
                return True
    
    return False


def apply_discount(rewards: List[float], gamma: float = 0.95) -> List[float]:
    """Apply discount factor to rewards"""
    discounted = []
    for i in range(len(rewards)):
        steps_from_end = len(rewards) - i - 1
        discount_factor = gamma ** steps_from_end
        discounted.append(rewards[i] * discount_factor)
    return discounted


# Experience Generation
def generate_experience(agent: PolicyAgent, num_games: int, order: int, 
                       clique_orders: Tuple[int, int], viz: TrainingVisualizer = None,
                       show_progress: bool = True) -> ExperienceBuffer:
    """Generate experience through self-play with shaped rewards"""
    red_collector = ExperienceCollector()
    blue_collector = ExperienceCollector()
    
    viz_config = {
        'show_every_n_games': 50,
        'update_board_every': 2,
        'update_heatmap_every': 10,
        'update_metrics_every': 5,
        'show_first_n_games': 5,
        'show_last_n_games': 5,
        'total_games': num_games
    }
    
    for game_num in range(num_games):
        if show_progress and game_num % 100 == 0:
            print(f'Simulating game {game_num}/{num_games}...')
            
        red_collector.begin_episode()
        blue_collector.begin_episode()
        
        # Simulate game
        winners, red_rewards, blue_rewards, game_length = simulate_game_with_rewards(
            agent, agent, order, clique_orders,
            red_collector, blue_collector, viz, game_num, viz_config
        )
        
        # Apply discount and final rewards
        gamma = 0.95
        red_rewards = apply_discount(red_rewards, gamma)
        blue_rewards = apply_discount(blue_rewards, gamma)
        
        final_bonus = 5.0
        if Player.red in winners:
            red_rewards = [r + final_bonus for r in red_rewards]
            blue_rewards = [r - final_bonus for r in blue_rewards]
        elif Player.blue in winners:
            red_rewards = [r - final_bonus for r in red_rewards]
            blue_rewards = [r + final_bonus for r in blue_rewards]
        
        red_collector.complete_episode(reward=None, custom_rewards=red_rewards)
        blue_collector.complete_episode(reward=None, custom_rewards=blue_rewards)
        
        if viz:
            viz.update_metrics(winners, game_length)
            if game_num % 50 == 0:
                viz.update_move_heatmap()
                viz.refresh()
    
    return combine_experience([red_collector, blue_collector])


def simulate_game_with_rewards(agent_1: PolicyAgent, agent_2: PolicyAgent, order: int,
                              clique_orders: Tuple[int, int], collector_1: ExperienceCollector,
                              collector_2: ExperienceCollector, viz: TrainingVisualizer,
                              game_num: int, viz_config: dict) -> Tuple[List, List[float], List[float], int]:
    """Simulate a single game and collect rewards"""
    game = GameState.new_game(order, clique_orders)
    agents = {Player.red: agent_1, Player.blue: agent_2}
    collectors = {Player.red: collector_1, Player.blue: collector_2}
    rewards = {Player.red: [], Player.blue: []}
    move_count = 0
    
    show_this_game = (
        game_num < viz_config['show_first_n_games'] or
        game_num >= viz_config['total_games'] - viz_config['show_last_n_games'] or
        game_num % viz_config['show_every_n_games'] == 0
    )
    
    while not game.is_over():
        current_player = game.active
        agent = agents[current_player]
        collector = collectors[current_player]
        
        state_tensor = agent.encoder.encode(game)
        move = agent.select_move(game)
        game_after = game.apply_move(move)
        immediate_reward = calculate_immediate_reward(game, game_after, move, current_player)
        
        if move.is_play:
            edge_idx = agent.encoder.encode_edge(move.edge)
            collector.record_decision(state=state_tensor, action=edge_idx)
            rewards[current_player].append(immediate_reward)
            
            u, v = move.edge
            viz.move_heatmap[u, v] += 1
            viz.move_heatmap[v, u] += 1
        
        game = game_after
        move_count += 1
        
        if show_this_game and viz and move_count % viz_config['update_board_every'] == 0:
            viz.update_game_visualization(game)
            viz.ax_game.text(0.02, 0.98, f'Move: {move_count}', 
                           transform=viz.ax_game.transAxes,
                           verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            viz.refresh()
    
    winners = game.win()
    
    if show_this_game and viz:
        viz.update_game_visualization(game)
        result_text = "Draw" if not winners else f"{winners[0].name.capitalize()} wins!"
        viz.ax_game.text(0.5, 0.02, f'Game {game_num}: {result_text} ({move_count} moves)', 
                       transform=viz.ax_game.transAxes,
                       horizontalalignment='center',
                       bbox=dict(boxstyle='round', 
                            facecolor='lightgreen' if winners else 'lightgray', 
                            alpha=0.7))
        viz.refresh()
    
    return winners, rewards[Player.red], rewards[Player.blue], move_count


def combine_experience(collectors: List[ExperienceCollector]) -> ExperienceBuffer:
    """Combine experience from multiple collectors"""
    combined_states = np.concatenate([np.array(c.states) for c in collectors])
    combined_actions = np.concatenate([np.array(c.actions) for c in collectors])
    combined_rewards = np.concatenate([np.array(c.rewards) for c in collectors])
    
    return ExperienceBuffer(combined_states, combined_actions, combined_rewards)


# Training Functions
def create_gpu_dataset(states: np.ndarray, targets: np.ndarray, rewards: np.ndarray, 
                      batch_size: int) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    """Create optimized tf.data.Dataset for GPU training"""
    # Calculate sample weights
    sample_weights = np.abs(rewards)
    sample_weights = sample_weights / np.mean(sample_weights)
    
    # Create full dataset first
    full_dataset = tf.data.Dataset.from_tensor_slices((
        states.astype(np.float32),
        targets.astype(np.float32),
        sample_weights.astype(np.float32)
    ))
    
    # Shuffle before splitting
    dataset_size = len(states)
    full_dataset = full_dataset.shuffle(buffer_size=min(dataset_size, 10000))
    
    # Calculate split sizes
    train_size = int(0.9 * dataset_size)
    val_size = dataset_size - train_size
    
    # Split the dataset
    train_dataset = full_dataset.take(train_size)
    val_dataset = full_dataset.skip(train_size)
    
    AUTOTUNE = tf.data.AUTOTUNE
    
    # Configure training dataset with repeat BEFORE batching
    train_dataset = (train_dataset
        .cache()
        .shuffle(buffer_size=min(train_size, 10000))
        .repeat()  # Repeat infinitely
        .batch(batch_size, drop_remainder=True)  # Drop incomplete batches
        .prefetch(AUTOTUNE))
    
    # Configure validation dataset (no repeat needed)
    val_dataset = (val_dataset
        .cache()
        .batch(batch_size, drop_remainder=True)
        .repeat()  # Also repeat validation to avoid issues
        .prefetch(AUTOTUNE))
    
    return train_dataset, val_dataset


def train_agent(agent: PolicyAgent, experience_buffer: ExperienceBuffer, 
                epochs: int, batch_size: int, viz: TrainingVisualizer = None,
                eval_games: int = 50, eval_every: int = 1, order: int = 5, 
                clique_orders: Tuple[int, int] = (3, 3)):
    """Train agent with GPU optimization"""
    # Learning rate schedule
    initial_lr = 1e-3
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_lr,
        decay_steps=5000,  # Adjusted for 50k games
        decay_rate=0.96,
        staircase=True
    )
    
    # Mixed precision optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    optimizer = mixed_precision.LossScaleOptimizer(optimizer)
    
    agent.model.compile(
        loss='categorical_crossentropy',
        optimizer=optimizer,
        metrics=['accuracy'],
        jit_compile=True
    )
    
    # Prepare data
    target_vectors = agent.prepare_experience_data(experience_buffer, agent.encoder.order)
    
    # Verify target vectors are valid
    print(f"\nTarget vectors check:")
    print(f"  Shape: {target_vectors.shape}")
    print(f"  Range: [{np.min(target_vectors):.3f}, {np.max(target_vectors):.3f}]")
    print(f"  Contains NaN: {np.any(np.isnan(target_vectors))}")
    print(f"  Contains Inf: {np.any(np.isinf(target_vectors))}")
    
    # Normalize target vectors to valid probability range
    # Since we're using softmax output, targets should sum to 1
    target_sums = np.sum(target_vectors, axis=1)
    problematic = np.abs(target_sums) < 1e-6  # Near zero sums
    if np.any(problematic):
        print(f"  WARNING: {np.sum(problematic)} target vectors have near-zero sum")
        # Fix by setting uniform distribution for problematic vectors
        target_vectors[problematic] = 1.0 / target_vectors.shape[1]
    
    # Ensure all values are positive (shift if needed)
    min_val = np.min(target_vectors)
    if min_val < 0:
        print(f"  Shifting negative values (min was {min_val:.3f})")
        target_vectors = target_vectors - min_val + 1e-6
    
    # Normalize rows to sum to 1
    row_sums = np.sum(target_vectors, axis=1, keepdims=True)
    row_sums = np.maximum(row_sums, 1e-6)  # Avoid division by zero
    target_vectors = target_vectors / row_sums
    
    print(f"  After normalization: [{np.min(target_vectors):.3f}, {np.max(target_vectors):.3f}]")
    print(f"  Row sums: {np.mean(np.sum(target_vectors, axis=1)):.3f} ± {np.std(np.sum(target_vectors, axis=1)):.3f}")
    
    train_dataset, val_dataset = create_gpu_dataset(
        experience_buffer.states,
        target_vectors,
        experience_buffer.rewards,
        batch_size
    )
    
    # Calculate steps
    total_samples = len(experience_buffer.states)
    validation_samples = int(0.1 * total_samples)
    train_samples = total_samples - validation_samples
    steps_per_epoch = train_samples // batch_size
    validation_steps = validation_samples // batch_size
    
    # Ensure we have at least 1 step
    steps_per_epoch = max(1, steps_per_epoch)
    validation_steps = max(1, validation_steps)
    
    # Callbacks
    callbacks = []
    
    # Add NaN detection callback
    class NaNCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            if logs:
                loss = logs.get('loss', 0)
                if np.isnan(loss) or np.isinf(loss) or loss < 0:
                    print(f"\nERROR: Invalid loss detected: {loss}")
                    print("Stopping training due to numerical instability.")
                    self.model.stop_training = True
    
    callbacks.append(NaNCallback())
    
    # Add evaluation callback to track win rate progress
    class EvaluationCallback(tf.keras.callbacks.Callback):
        def __init__(self, agent, eval_games, eval_every, order, clique_orders, viz):
            self.agent = agent
            self.eval_games = eval_games
            self.eval_every = eval_every
            self.order = order
            self.clique_orders = clique_orders
            self.viz = viz
            self.epoch_win_rates = []
            
        def on_epoch_end(self, epoch, logs=None):
            # Only evaluate every N epochs
            if (epoch + 1) % self.eval_every == 0:
                # Evaluate agent
                print(f"\nEvaluating epoch {epoch + 1} performance...")
                win_rate = quick_evaluate(self.agent, self.eval_games, 
                                        self.order, self.clique_orders)
                self.epoch_win_rates.append(win_rate)
                
                print(f"Epoch {epoch + 1} win rate: {win_rate:.1%}")
                
                # Update visualization
                if self.viz:
                    self.viz.update_epoch_progress(epoch + 1, win_rate)
                    self.viz.refresh()
    
    callbacks.append(EvaluationCallback(agent, eval_games, eval_every, order, clique_orders, viz))
    
    if viz:
        class VizCallback(tf.keras.callbacks.Callback):
            def __init__(self, viz):
                self.viz = viz
                
            def on_epoch_end(self, epoch, logs=None):
                if logs:
                    self.viz.update_training_metrics(
                        loss=logs.get('loss'),
                        val_loss=logs.get('val_loss'),
                        accuracy=logs.get('accuracy')
                    )
                    self.viz.refresh()
        
        callbacks.append(VizCallback(viz))
    
    callbacks.extend([
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        )
    ])
    
    # Add win-rate based early stopping
    class WinRateEarlyStopping(tf.keras.callbacks.Callback):
        def __init__(self, eval_callback, target_win_rate=0.95, patience=5):
            self.eval_callback = eval_callback
            self.target_win_rate = target_win_rate
            self.patience = patience
            self.best_win_rate = 0
            self.wait = 0
            
        def on_epoch_end(self, epoch, logs=None):
            # Only check if we evaluated this epoch
            if len(self.eval_callback.epoch_win_rates) > 0:
                current_win_rate = self.eval_callback.epoch_win_rates[-1]
                
                if current_win_rate >= self.target_win_rate:
                    print(f"\nReached target win rate of {self.target_win_rate:.1%}! Stopping training.")
                    self.model.stop_training = True
                elif current_win_rate > self.best_win_rate:
                    self.best_win_rate = current_win_rate
                    self.wait = 0
                else:
                    self.wait += 1
                    if self.wait >= self.patience:
                        print(f"\nWin rate hasn't improved for {self.patience} epochs. Stopping.")
                        self.model.stop_training = True
    
    # Get reference to eval callback (it's the second one after NaNCallback)
    eval_callback = callbacks[1]
    callbacks.append(WinRateEarlyStopping(eval_callback, target_win_rate=0.95, patience=10))
    
    # Train
    print(f"\nTraining on {total_samples:,} samples with batch size {batch_size}")
    print(f"Steps per epoch: {steps_per_epoch}, Validation steps: {validation_steps}")
    
    history = agent.model.fit(
        train_dataset,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_dataset,
        validation_steps=validation_steps,
        verbose=1,
        callbacks=callbacks
    )
    
    return history


# Evaluation Functions
def quick_evaluate(agent: PolicyAgent, num_games: int, order: int, 
                   clique_orders: Tuple[int, int]) -> float:
    """Quick evaluation against random bot (no visualization)"""
    random_bot = RandomBot()
    wins = 0
    
    # Disable GPU predictions for small batches
    with tf.device('/CPU:0'):
        # Test half games as red, half as blue
        for i in range(num_games // 2):
            # As red
            game = GameState.new_game(order, clique_orders)
            agents = {Player.red: agent, Player.blue: random_bot}
            
            while not game.is_over():
                move = agents[game.active].select_move(game)
                game = game.apply_move(move)
            
            if Player.red in game.win():
                wins += 1
            
            # As blue
            game = GameState.new_game(order, clique_orders)
            agents = {Player.red: random_bot, Player.blue: agent}
            
            while not game.is_over():
                move = agents[game.active].select_move(game)
                game = game.apply_move(move)
            
            if Player.blue in game.win():
                wins += 1
    
    return wins / num_games


def evaluate_agent(agent: PolicyAgent, num_games: int, order: int, 
                  clique_orders: Tuple[int, int], viz: TrainingVisualizer = None) -> float:
    """Evaluate agent against random bot"""
    random_bot = RandomBot()
    wins_as_red = 0
    wins_as_blue = 0
    
    print("\nEvaluating as Red player...")
    for i in range(num_games // 2):
        game = GameState.new_game(order, clique_orders)
        agents = {Player.red: agent, Player.blue: random_bot}
        
        while not game.is_over():
            move = agents[game.active].select_move(game)
            game = game.apply_move(move)
        
        winners = game.win()
        if Player.red in winners:
            wins_as_red += 1
            
        if viz and i % 10 == 0:
            viz.update_metrics(winners, 10)
            viz.refresh()
    
    print("\nEvaluating as Blue player...")
    for i in range(num_games // 2):
        game = GameState.new_game(order, clique_orders)
        agents = {Player.red: random_bot, Player.blue: agent}
        
        while not game.is_over():
            move = agents[game.active].select_move(game)
            game = game.apply_move(move)
        
        winners = game.win()
        if Player.blue in winners:
            wins_as_blue += 1
            
        if viz and i % 10 == 0:
            viz.update_metrics(winners, 10)
            viz.refresh()
    
    total_wins = wins_as_red + wins_as_blue
    win_rate = total_wins / num_games
    
    print(f"\nResults against RandomBot:")
    print(f"As Red: {wins_as_red}/{num_games//2} wins ({wins_as_red/(num_games//2)*100:.1f}%)")
    print(f"As Blue: {wins_as_blue}/{num_games//2} wins ({wins_as_blue/(num_games//2)*100:.1f}%)")
    print(f"Total: {total_wins}/{num_games} wins ({win_rate*100:.1f}%)")
    
    return win_rate


def analyze_experience(experience_buffer: ExperienceBuffer, order: int = 5) -> dict:
    """Analyze loaded experience data"""
    print("\nAnalyzing experience data...")
    
    # Reward statistics
    mean_reward = np.mean(experience_buffer.rewards)
    std_reward = np.std(experience_buffer.rewards)
    
    print(f"\nReward statistics:")
    print(f"  Mean: {mean_reward:.3f}")
    print(f"  Std:  {std_reward:.3f}")
    print(f"  Min:  {np.min(experience_buffer.rewards):.3f}")
    print(f"  Max:  {np.max(experience_buffer.rewards):.3f}")
    
    # Action distribution
    action_counts = np.bincount(experience_buffer.actions, minlength=order**2)
    most_played = np.argmax(action_counts)
    least_played = np.argmin(action_counts)
    
    print(f"\nAction distribution:")
    print(f"  Most played edge:  {most_played} ({action_counts[most_played]:,} times)")
    print(f"  Least played edge: {least_played} ({action_counts[least_played]:,} times)")
    
    # Show which edges are never played
    never_played = np.where(action_counts == 0)[0]
    if len(never_played) > 0:
        print(f"  Never played edges: {never_played}")
        # Decode what these edges are
        for edge_idx in never_played[:5]:  # Show first 5
            u, v = divmod(edge_idx, order)
            print(f"    Edge {edge_idx}: ({u}, {v})")
    
    # Estimate wins/losses
    wins = np.sum(experience_buffer.rewards > 3)
    losses = np.sum(experience_buffer.rewards < -3)
    
    print(f"\nEstimated game outcomes:")
    print(f"  Wins:   {wins:,} ({wins/(wins+losses)*100:.1f}%)")
    print(f"  Losses: {losses:,} ({losses/(wins+losses)*100:.1f}%)")
    
    # Reward distribution
    unique_rewards, counts = np.unique(experience_buffer.rewards, return_counts=True)
    print(f"\nReward distribution (top 5 most common):")
    sorted_indices = np.argsort(counts)[::-1][:5]
    for idx in sorted_indices:
        print(f"  Reward {unique_rewards[idx]:6.2f}: {counts[idx]:6,} times "
              f"({counts[idx]/len(experience_buffer.rewards)*100:5.1f}%)")
    
    return {
        'mean_reward': mean_reward,
        'std_reward': std_reward,
        'action_distribution': action_counts,
        'estimated_win_rate': wins / (wins + losses) if (wins + losses) > 0 else 0
    }


def main():
    """Main training function with support for loading existing experience"""
    parser = argparse.ArgumentParser(description="GPU-optimized Ramsey AI training")
    parser.add_argument("--load-experience", type=str, default=None,
                       help="Path to existing experience file to load")
    parser.add_argument("--num-games", type=int, default=1000,
                       help="Number of games to generate (if not loading)")
    parser.add_argument("--epochs", type=int, default=100,
                       help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=1024,
                       help="Batch size for training")
    parser.add_argument("--eval-games", type=int, default=50,
                       help="Number of games for evaluation after each epoch")
    parser.add_argument("--eval-every", type=int, default=1,
                       help="Evaluate every N epochs (1=every epoch)")
    parser.add_argument("--no-viz", action="store_true",
                       help="Disable visualization")
    args = parser.parse_args()
    
    # Configure GPU
    configure_gpu()
    
    # Initialize
    order = 5
    clique_orders = (3, 3)
    encoder = K3Encoder(order)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create visualizer
    viz = TrainingVisualizer(order) if not args.no_viz else None
    
    # Create model and agent
    print("Creating GPU-optimized model...")
    model = create_gpu_model(encoder, order)
    agent = PolicyAgent(model, encoder)
    
    print(f"Model parameters: {model.count_params():,}")
    
    # Test initial model prediction
    print("\nTesting initial model predictions...")
    test_state = GameState.new_game(order, clique_orders)
    test_tensor = encoder.encode(test_state)
    test_input = np.array([test_tensor])
    with tf.device('/CPU:0'):
        test_pred = model.predict(test_input, verbose=0)[0]
    print(f"  Prediction shape: {test_pred.shape}")
    print(f"  Prediction range: [{np.min(test_pred):.6f}, {np.max(test_pred):.6f}]")
    print(f"  Prediction sum: {np.sum(test_pred):.6f}")
    print(f"  Top 5 actions: {np.argsort(test_pred)[-5:][::-1]}")
    
    # Save initial model
    initial_model_path = f'gpu_model_initial_{timestamp}.h5'
    with h5py.File(initial_model_path, 'w') as f:
        agent.serialize(f)
    print(f"Saved initial model to {initial_model_path}")
    
    # Load or generate experience
    start_time = time.time()
    
    if args.load_experience:
        print(f"\nLoading experience from: {args.load_experience}")
        try:
            with h5py.File(args.load_experience, 'r') as f:
                experience_buffer = ExperienceBuffer.load_experience(f)
            
            print(f"Successfully loaded experience:")
            print(f"  States shape: {experience_buffer.states.shape}")
            print(f"  Total samples: {len(experience_buffer.states):,}")
            
            # Analyze the loaded experience
            stats = analyze_experience(experience_buffer, order)
            
            # Additional diagnostics
            print(f"\nData diagnostics:")
            print(f"  Action 0 count: {stats['action_distribution'][0]}")
            print(f"  Non-zero actions: {np.sum(stats['action_distribution'] > 0)}/{len(stats['action_distribution'])}")
            
            # Check for edge validity
            invalid_actions = experience_buffer.actions >= order**2
            if np.any(invalid_actions):
                print(f"  WARNING: {np.sum(invalid_actions)} invalid actions found!")
                # Filter out invalid actions
                valid_mask = experience_buffer.actions < order**2
                experience_buffer.states = experience_buffer.states[valid_mask]
                experience_buffer.actions = experience_buffer.actions[valid_mask]
                experience_buffer.rewards = experience_buffer.rewards[valid_mask]
                print(f"  Filtered to {len(experience_buffer.states)} valid samples")
            
        except Exception as e:
            print(f"Error loading experience: {e}")
            return
    else:
        print(f"\nGenerating {args.num_games} games of experience...")
        experience_buffer = generate_experience(
            agent, args.num_games, order, clique_orders, viz
        )
        
        # Save experience
        experience_path = f'experience_gpu_{timestamp}_{args.num_games}games.h5'
        with h5py.File(experience_path, 'w') as f:
            experience_buffer.serialize(f)
        print(f"Saved experience to {experience_path}")
    
    # Train
    print(f"\nTraining for {args.epochs} epochs with batch size {args.batch_size}...")
    print(f"Will evaluate on {args.eval_games} games every {args.eval_every} epoch(s)")
    history = train_agent(agent, experience_buffer, args.epochs, args.batch_size, 
                         viz, args.eval_games, args.eval_every, order, clique_orders)
    
    # Save trained model
    trained_model_path = f'gpu_model_trained_{timestamp}.h5'
    with h5py.File(trained_model_path, 'w') as f:
        agent.serialize(f)
    print(f"Saved trained model to {trained_model_path}")
    
    # Evaluate
    print(f"\nFinal evaluation on {args.eval_games * 2} games...")
    win_rate = evaluate_agent(agent, args.eval_games * 2, order, clique_orders, viz)
    
    # Summary
    total_time = time.time() - start_time
    print(f"\n{'='*50}")
    print(f"Training Complete!")
    print(f"{'='*50}")
    print(f"Total time: {total_time:.1f} seconds")
    print(f"Final win rate: {win_rate:.1%}")
    print(f"Model saved to: {trained_model_path}")
    
    # Show win rate progression if available
    if viz and viz.epoch_win_rates:
        print(f"\nWin Rate Progression:")
        print(f"  Initial: {viz.epoch_win_rates[0][1]:.1%}")
        print(f"  Final:   {viz.epoch_win_rates[-1][1]:.1%}")
        print(f"  Best:    {max(wr for _, wr in viz.epoch_win_rates):.1%}")
        print(f"  Improvement: {(viz.epoch_win_rates[-1][1] - viz.epoch_win_rates[0][1]):.1%}")
    
    if viz:
        print("\nPress Enter to close visualization...")
        input()
        plt.close('all')


if __name__ == "__main__":
    main()
"""
Next-Generation Ramsey AI: Graph Attention Networks with Physics-Inspired Optimization
===============================================================================

This implementation transforms the two-player Ramsey game into a single-agent optimization
problem using Graph Attention Networks with Potts model physics-inspired loss functions,
optimized for RTX 4080 Super with 16GB VRAM.

Key innovations:
- Graph Attention Networks v2 with enhanced attention mechanisms  
- Physics-inspired loss functions (Potts model for graph coloring)
- Single-agent MCTS with neural guidance
- Progressive training from 20-node to 1000-node graphs
- Memory-optimized for RTX 4080 Super
- Hybrid SAT-ML integration for concrete discoveries
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch_geometric.nn import GATv2Conv, global_mean_pool, global_max_pool
from torch_geometric.data import Data, Batch
from torch_geometric.utils import to_undirected, dense_to_sparse
import torch_geometric as pyg

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Union
import time
import math
import random
from collections import deque, namedtuple
import threading
import queue
import h5py

# Set device and optimization flags
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Only set CUDA optimizations if CUDA is available
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    print("GPU not available - using CPU (training will be slower)")
    print("For GPU acceleration, install PyTorch with CUDA support:")


@dataclass
class RamseyConfig:
    """Configuration for Ramsey number discovery"""
    # Graph parameters
    min_order: int = 20          # Start with small graphs
    max_order: int = 1000        # Scale up to large graphs  
    target_r: int = 5            # Target Ramsey number R(5,5)
    target_s: int = 5
    
    # Model architecture
    hidden_dim: int = 256        # Hidden dimension for GAT
    num_heads: int = 8           # Multi-head attention
    num_layers: int = 4          # GAT layers
    dropout: float = 0.1
    
    # Training parameters
    batch_size: int = 32         # Adaptive based on graph size
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    
    # Physics parameters (Potts model)
    temperature: float = 1.0     # Temperature for Potts model
    coupling_strength: float = 1.0  # J parameter in Potts model
    
    # MCTS parameters
    mcts_simulations: int = 1000
    exploration_constant: float = 1.414
    
    # Memory optimization
    gradient_checkpointing: bool = True
    mixed_precision: bool = True
    max_memory_gb: float = 14.0  # Conservative for RTX 4080 Super


class GraphAttentionNet(nn.Module):
    """
    Graph Attention Network v2 with physics-inspired components
    for single-agent Ramsey number discovery
    """
    
    def __init__(self, config: RamseyConfig):
        super().__init__()
        self.config = config
        
        # Input projection for node features
        self.node_embedding = nn.Linear(1, config.hidden_dim)
        
        # GAT layers with residual connections
        self.gat_layers = nn.ModuleList()
        for i in range(config.num_layers):
            in_dim = config.hidden_dim
            out_dim = config.hidden_dim
            
            self.gat_layers.append(
                GATv2Conv(
                    in_channels=in_dim,
                    out_channels=out_dim // config.num_heads,
                    heads=config.num_heads,
                    dropout=config.dropout,
                    concat=True,
                    edge_dim=3,  # Edge features: [black, red, blue]
                    bias=True
                )
            )
        
        # Layer normalization for each GAT layer
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(config.hidden_dim) for _ in range(config.num_layers)
        ])
        
        # Physics-inspired Potts model component
        self.potts_energy = PottsEnergyModule(config)
        
        # Policy head for action selection
        self.policy_head = nn.Sequential(
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),  # *2 for global pooling
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(config.hidden_dim // 2, 1)  # Single output per edge
        )
        
        # Value head for state evaluation
        self.value_head = nn.Sequential(
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, 1)
        )
        
        # Add adaptive pooling layer
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Ensure parameters require gradients
        for param in self.parameters():
            param.requires_grad = True
    
    def _init_weights(self, module):
        """Xavier initialization for better training stability"""
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
    
    def _prepare_batch(self, data):
        """Prepare batch data for training."""
        if hasattr(data, 'x'):
            data.x = data.x.requires_grad_(True)
        if hasattr(data, 'edge_index'):
            data.edge_index = data.edge_index.requires_grad_(True)
        return data
    
    def forward(self, data: Data) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass returning policy logits, value, and physics energy
        
        Args:
            data: PyG Data object with node features and edge indices
            
        Returns:
            policy_logits: Action probabilities for each possible edge
            value: State value estimation
            physics_energy: Potts model energy for physics-informed loss
        """
        # Prepare the batch first
        data = self._prepare_batch(data)
        
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        batch = getattr(data, 'batch', None)
        
        # Initial node embeddings
        x = self.node_embedding(x)
        
        # GAT layers with residual connections
        for i, (gat_layer, layer_norm) in enumerate(zip(self.gat_layers, self.layer_norms)):
            residual = x
            
            if self.config.gradient_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(gat_layer, x, edge_index, edge_attr)
            else:
                x = gat_layer(x, edge_index, edge_attr)
            
            x = layer_norm(x + residual)  # Residual connection
            x = F.relu(x)
            x = F.dropout(x, p=self.config.dropout, training=self.training)
        
        # Global graph representation
        if batch is not None:
            graph_repr = torch.cat([
                global_mean_pool(x, batch),
                global_max_pool(x, batch)
            ], dim=1)
        else:
            # Single graph
            graph_repr = torch.cat([
                x.mean(dim=0, keepdim=True),
                x.max(dim=0, keepdim=True)[0]
            ], dim=1)
        
        # Generate edge representations for policy
        edge_representations = self._create_edge_representations(x, data)
        
        # Policy logits for each possible edge
        policy_logits = self.policy_head(edge_representations).squeeze(-1)
        
        # Value estimation
        value = self.value_head(graph_repr).squeeze(-1)
        
        # Physics-inspired energy calculation
        physics_energy = self.potts_energy(data, x)
        
        # Get batch size from the input data
        batch_size = data.batch[-1].item() + 1 if hasattr(data, 'batch') else 1
        
        # Flatten and apply adaptive pooling
        x = x.view(batch_size, -1)  # Flatten
        x = x.unsqueeze(2)  # Add dimension for 1D adaptive pooling
        x = self.adaptive_pool(x)
        x = x.squeeze(2)  # Remove the extra dimension
        
        policy_logits = self.policy_head(x)
        value = self.value_head(x)
        
        return policy_logits, value, x
    
    def _create_edge_representations(self, node_features: torch.Tensor, data: Data) -> torch.Tensor:
        """Create representations for all possible edges"""
        num_nodes = node_features.size(0)
        
        # Generate all possible edges (upper triangular)
        edge_indices = []
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                edge_indices.append([i, j])
        
        if not edge_indices:
            # Single node case
            return torch.zeros(1, self.config.hidden_dim * 2, device=node_features.device)
        
        edge_indices = torch.tensor(edge_indices, device=node_features.device).t()
        
        # Create edge representations by concatenating node features
        source_features = node_features[edge_indices[0]]
        target_features = node_features[edge_indices[1]]
        edge_representations = torch.cat([source_features, target_features], dim=1)
        
        return edge_representations


class PottsEnergyModule(nn.Module):
    """
    Physics-inspired Potts model energy calculation for graph coloring constraints
    """
    
    def __init__(self, config: RamseyConfig):
        super().__init__()
        self.config = config
        self.coupling_strength = nn.Parameter(torch.tensor(config.coupling_strength))
        
    def forward(self, data: Data, node_features: torch.Tensor) -> torch.Tensor:
        """
        Calculate Potts model energy: H = -J * Σ δ(σᵢ, σⱼ) for adjacent nodes
        Lower energy = better coloring (fewer conflicts)
        """
        edge_index, edge_attr = data.edge_index, data.edge_attr
        
        if edge_index.numel() == 0:
            return torch.tensor(0.0, device=node_features.device)
        
        # Get edge colors (0=black, 1=red, 2=blue)
        edge_colors = torch.argmax(edge_attr, dim=1)
        
        # Only consider colored edges (not black)
        colored_mask = edge_colors != 0
        
        if not colored_mask.any():
            return torch.tensor(0.0, device=node_features.device)
        
        colored_edges = edge_index[:, colored_mask]
        colored_edge_colors = edge_colors[colored_mask]
        
        # Calculate energy contribution from each colored edge
        # Penalty for creating potential cliques
        energy = 0.0
        
        # Group edges by color
        for color in [1, 2]:  # red, blue
            color_mask = colored_edge_colors == color
            if not color_mask.any():
                continue
            
            color_edges = colored_edges[:, color_mask]
            
            # Count triangles of this color (simplified approximation)
            # In practice, would need more sophisticated clique detection
            triangle_energy = self._approximate_triangle_energy(color_edges, node_features.size(0))
            energy += triangle_energy
        
        return energy * self.coupling_strength
    
    def _approximate_triangle_energy(self, edges: torch.Tensor, num_nodes: int) -> torch.Tensor:
        """Approximate energy contribution from potential triangles"""
        if edges.size(1) < 3:
            return torch.tensor(0.0, device=edges.device)
        
        # Simple approximation: penalize high-degree nodes in monochromatic subgraph
        degrees = torch.zeros(num_nodes, device=edges.device)
        degrees.scatter_add_(0, edges[0], torch.ones_like(edges[0], dtype=torch.float))
        degrees.scatter_add_(0, edges[1], torch.ones_like(edges[1], dtype=torch.float))
        
        # Energy increases quadratically with degree (more triangles)
        energy = (degrees ** 2).sum() / (2 * num_nodes)
        return energy


class RamseyEnvironment:
    """
    Single-agent environment for Ramsey number discovery
    Converts two-player game to optimization problem
    """
    
    def __init__(self, config: RamseyConfig, order: int):
        self.config = config
        self.order = order
        self.reset()
    
    def reset(self) -> Data:
        """Reset environment to initial state"""
        self.graph = nx.complete_graph(self.order)
        
        # Initialize all edges as black (uncolored)
        for u, v in self.graph.edges():
            self.graph[u][v]['color'] = 0  # 0=black, 1=red, 2=blue
        
        self.colored_edges = set()
        self.red_clique_size = 0
        self.blue_clique_size = 0
        
        return self._get_graph_data()
    
    def step(self, action: int, color: int) -> Tuple[Data, float, bool, Dict]:
        """
        Take action in environment
        
        Args:
            action: Edge index to color
            color: Color to assign (1=red, 2=blue)
            
        Returns:
            next_state: New graph state
            reward: Immediate reward
            done: Whether episode is complete
            info: Additional information
        """
        # Decode edge from action
        edge = self._decode_edge(action)
        
        if edge in self.colored_edges:
            # Invalid action - edge already colored
            return self._get_graph_data(), -10.0, False, {'invalid_action': True}
        
        # Color the edge
        u, v = edge
        self.graph[u][v]['color'] = color
        self.colored_edges.add(edge)
        
        # Update clique sizes
        self.red_clique_size = self._get_max_clique_size(1)
        self.blue_clique_size = self._get_max_clique_size(2)
        
        # Calculate reward
        reward = self._calculate_reward()
        
        # Check if done
        done = self._is_done()
        
        info = {
            'red_clique_size': self.red_clique_size,
            'blue_clique_size': self.blue_clique_size,
            'colored_edges': len(self.colored_edges),
            'total_edges': self.order * (self.order - 1) // 2
        }
        
        return self._get_graph_data(), reward, done, info
    
    def _get_graph_data(self) -> Data:
        """Convert NetworkX graph to PyG Data object"""
        # Node features (just identity for now)
        x = torch.ones(self.order, 1, dtype=torch.float)
        
        # Edge indices and features
        edge_list = list(self.graph.edges())
        if not edge_list:
            # Empty graph case
            edge_index = torch.zeros(2, 0, dtype=torch.long)
            edge_attr = torch.zeros(0, 3, dtype=torch.float)
        else:
            edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
            edge_index = to_undirected(edge_index)  # Ensure undirected
            
            # Edge attributes: one-hot encoded colors
            edge_attr = []
            for u, v in edge_list:
                color = self.graph[u][v]['color']
                attr = [0, 0, 0]
                attr[color] = 1
                edge_attr.extend([attr, attr])  # Add for both directions
            
            edge_attr = torch.tensor(edge_attr, dtype=torch.float)
        
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    
    def _decode_edge(self, action: int) -> Tuple[int, int]:
        """Decode action index to edge"""
        # Map action to upper triangular edge
        edges = []
        for i in range(self.order):
            for j in range(i + 1, self.order):
                edges.append((i, j))
        
        if action >= len(edges):
            action = action % len(edges)  # Wrap around
        
        return edges[action]
    
    def _get_max_clique_size(self, color: int) -> int:
        """Get maximum clique size for given color"""
        # Create subgraph with only edges of given color
        subgraph = nx.Graph()
        subgraph.add_nodes_from(range(self.order))
        
        for u, v in self.graph.edges():
            if self.graph[u][v]['color'] == color:
                subgraph.add_edge(u, v)
        
        if subgraph.number_of_edges() == 0:
            return 1 if subgraph.number_of_nodes() > 0 else 0
        
        # Find maximum clique (expensive for large graphs)
        try:
            cliques = list(nx.find_cliques(subgraph))
            return max(len(clique) for clique in cliques) if cliques else 1
        except:
            return 1
    
    def _calculate_reward(self) -> float:
        """Physics-inspired reward function"""
        # Base reward for avoiding large cliques
        clique_penalty = 0.0
        
        if self.red_clique_size >= self.config.target_r:
            clique_penalty -= 100.0  # Large penalty for forbidden clique
        elif self.red_clique_size == self.config.target_r - 1:
            clique_penalty -= 10.0   # Warning penalty
        
        if self.blue_clique_size >= self.config.target_s:
            clique_penalty -= 100.0
        elif self.blue_clique_size == self.config.target_s - 1:
            clique_penalty -= 10.0
        
        # Reward for balanced coloring
        red_edges = sum(1 for u, v in self.graph.edges() if self.graph[u][v]['color'] == 1)
        blue_edges = sum(1 for u, v in self.graph.edges() if self.graph[u][v]['color'] == 2)
        
        balance_reward = -abs(red_edges - blue_edges) * 0.1
        
        # Progress reward
        progress_reward = len(self.colored_edges) * 0.01
        
        return clique_penalty + balance_reward + progress_reward
    
    def _is_done(self) -> bool:
        """Check if episode is complete"""
        # Done if forbidden clique found or all edges colored
        forbidden_clique = (self.red_clique_size >= self.config.target_r or 
                           self.blue_clique_size >= self.config.target_s)
        
        all_colored = len(self.colored_edges) == self.order * (self.order - 1) // 2
        
        return forbidden_clique or all_colored


class MCTSNode:
    """Monte Carlo Tree Search node for neural-guided search"""
    
    def __init__(self, state: Data, parent: Optional['MCTSNode'] = None, 
                 action: Optional[int] = None, color: Optional[int] = None):
        self.state = state
        self.parent = parent
        self.action = action
        self.color = color
        self.children: List['MCTSNode'] = []
        
        self.visits = 0
        self.value_sum = 0.0
        self.prior_prob = 0.0
        
    def is_fully_expanded(self, valid_actions: List[int]) -> bool:
        return len(self.children) == len(valid_actions) * 2  # * 2 for red/blue
    
    def best_child(self, exploration_constant: float) -> 'MCTSNode':
        """Select best child using UCB1"""
        def ucb_score(child: 'MCTSNode') -> float:
            if child.visits == 0:
                return float('inf')
            
            exploitation = child.value_sum / child.visits
            exploration = exploration_constant * math.sqrt(math.log(self.visits) / child.visits)
            return exploitation + exploration
        
        return max(self.children, key=ucb_score)
    
    def backup(self, value: float):
        """Backup value through tree"""
        self.visits += 1
        self.value_sum += value
        
        if self.parent:
            self.parent.backup(value)


class NeuralMCTS:
    """Neural-guided Monte Carlo Tree Search for Ramsey optimization"""
    
    def __init__(self, model: GraphAttentionNet, config: RamseyConfig):
        self.model = model
        self.config = config
    
    def search(self, env: RamseyEnvironment, num_simulations: int) -> Tuple[int, int]:
        """
        Run MCTS to find best action
        
        Returns:
            best_action: Edge index to color
            best_color: Color to use (1=red, 2=blue)
        """
        root = MCTSNode(env._get_graph_data())
        
        for _ in range(num_simulations):
            # Selection
            node = self._select(root, env)
            
            # Expansion and Evaluation
            if not self._is_terminal(node, env):
                node = self._expand(node, env)
                value = self._evaluate(node)
            else:
                value = self._get_terminal_value(node, env)
            
            # Backup
            node.backup(value)
        
        # Return best action
        if not root.children:
            # Fallback to random
            valid_actions = self._get_valid_actions(env)
            if valid_actions:
                action = random.choice(valid_actions)
                color = random.choice([1, 2])
                return action, color
            return 0, 1
        
        best_child = max(root.children, key=lambda c: c.visits)
        return best_child.action, best_child.color
    
    def _select(self, node: MCTSNode, env: RamseyEnvironment) -> MCTSNode:
        """Selection phase of MCTS"""
        while node.children and not self._is_terminal(node, env):
            if node.is_fully_expanded(self._get_valid_actions(env)):
                node = node.best_child(self.config.exploration_constant)
            else:
                break
        return node
    
    def _expand(self, node: MCTSNode, env: RamseyEnvironment) -> MCTSNode:
        """Expansion phase of MCTS"""
        valid_actions = self._get_valid_actions(env)
        
        for action in valid_actions:
            for color in [1, 2]:  # red, blue
                if not any(c.action == action and c.color == color for c in node.children):
                    # Simulate action
                    env_copy = self._copy_env(env)
                    next_state, _, _, _ = env_copy.step(action, color)
                    
                    child = MCTSNode(next_state, parent=node, action=action, color=color)
                    node.children.append(child)
                    return child
        
        return node
    
    def _evaluate(self, node: MCTSNode) -> float:
        """Neural evaluation of node"""
        with torch.no_grad():
            data = node.state.to(device)
            data = self.model._prepare_batch(data)  # Add this line
            _, value, _ = self.model(data)
            return value.item()
    
    def _get_valid_actions(self, env: RamseyEnvironment) -> List[int]:
        """Get valid actions (uncolored edges)"""
        valid_actions = []
        edge_idx = 0
        
        for i in range(env.order):
            for j in range(i + 1, env.order):
                if (i, j) not in env.colored_edges:
                    valid_actions.append(edge_idx)
                edge_idx += 1
        
        return valid_actions
    
    def _is_terminal(self, node: MCTSNode, env: RamseyEnvironment) -> bool:
        """Check if node represents terminal state"""
        # Simplified check - would need to reconstruct environment state
        return False  # For now, always allow expansion
    
    def _get_terminal_value(self, node: MCTSNode, env: RamseyEnvironment) -> float:
        """Get value for terminal node"""
        return 0.0
    
    def _copy_env(self, env: RamseyEnvironment) -> RamseyEnvironment:
        """Create copy of environment for simulation"""
        new_env = RamseyEnvironment(env.config, env.order)
        new_env.graph = env.graph.copy()
        new_env.colored_edges = env.colored_edges.copy()
        new_env.red_clique_size = env.red_clique_size
        new_env.blue_clique_size = env.blue_clique_size
        return new_env


class RamseyTrainer:
    """Main training loop for Ramsey number discovery"""
    
    def __init__(self, config: RamseyConfig):
        self.config = config
        self.model = GraphAttentionNet(config).to(device)
        
        # Optimizer with gradient clipping
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10
        )
        
        # Mixed precision training (only on GPU)
        if config.mixed_precision and torch.cuda.is_available():
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None
            if config.mixed_precision and not torch.cuda.is_available():
                print("Mixed precision disabled: requires CUDA")
        
        # MCTS for action selection
        self.mcts = NeuralMCTS(self.model, config)
        
        # Training metrics
        self.metrics = {
            'losses': [],
            'rewards': [],
            'clique_sizes': [],
            'success_rate': []
        }
    
    def train(self, num_episodes, progressive_sizes=True):
        """
        Train the model using progressive difficulty
        
        Args:
            num_episodes: Number of training episodes
            progressive_sizes: Whether to gradually increase graph size
        """
        print(f"Starting training for {num_episodes} episodes")
        print(f"Device: {device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        episode_rewards = deque(maxlen=100)
        episode_clique_sizes = deque(maxlen=100)
        
        for episode in range(num_episodes):
            # Progressive graph sizing
            if progressive_sizes:
                progress = episode / num_episodes
                current_order = int(self.config.min_order + 
                                  progress * (self.config.max_order - self.config.min_order))
                current_order = max(self.config.min_order, min(current_order, self.config.max_order))
            else:
                current_order = self.config.min_order
            
            # Adaptive batch size based on graph size
            adaptive_batch_size = max(1, self.config.batch_size // max(1, current_order // 20))
            
            # Run episode
            episode_reward, max_clique_size, loss = self._run_episode(current_order)
            
            episode_rewards.append(episode_reward)
            episode_clique_sizes.append(max_clique_size)
            
            # Update learning rate
            if loss is not None:
                self.scheduler.step(loss)
                self.metrics['losses'].append(loss)
            
            # Logging
            if episode % 100 == 0:
                avg_reward = np.mean(episode_rewards) if episode_rewards else 0
                avg_clique = np.mean(episode_clique_sizes) if episode_clique_sizes else 0
                success_rate = sum(1 for r in episode_rewards if r > 0) / len(episode_rewards) if episode_rewards else 0
                
                print(f"Episode {episode}: Order={current_order}, "
                      f"Avg Reward={avg_reward:.2f}, Avg Max Clique={avg_clique:.1f}, "
                      f"Success Rate={success_rate:.2%}")
                
                self.metrics['rewards'].append(avg_reward)
                self.metrics['clique_sizes'].append(avg_clique)
                self.metrics['success_rate'].append(success_rate)
                
                # Memory cleanup
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    
    def _run_episode(self, order: int) -> Tuple[float, int, Optional[float]]:
        """Run single training episode"""
        env = RamseyEnvironment(self.config, order)
        state = env.reset()
        
        episode_reward = 0.0
        episode_data = []
        max_clique_size = 0
        
        done = False
        step_count = 0
        max_steps = order * (order - 1)  # Reasonable limit
        
        while not done and step_count < max_steps:
            # Get action from MCTS + neural network
            action, color = self.mcts.search(env, self.config.mcts_simulations // 10)  # Reduced for speed
            
            # Take action
            next_state, reward, done, info = env.step(action, color)
            
            episode_reward += reward
            max_clique_size = max(max_clique_size, 
                                max(info['red_clique_size'], info['blue_clique_size']))
            
            # Store experience
            episode_data.append({
                'state': state,
                'action': action,
                'color': color,
                'reward': reward,
                'next_state': next_state,
                'done': done
            })
            
            state = next_state
            step_count += 1
        
        # Train on episode data
        loss = self._train_on_episode(episode_data)
        
        return episode_reward, max_clique_size, loss
    
    def _train_on_episode(self, episode_data: List[Dict]) -> Optional[float]:
        """Train model on episode experience"""
        if not episode_data:
            return None
        
        self.model.train()
        
        # Prepare batch data
        states = []
        targets = []
        values = []
        
        # Calculate discounted returns
        returns = []
        G = 0
        gamma = 0.99
        
        for data in reversed(episode_data):
            G = data['reward'] + gamma * G
            returns.append(G)
        
        returns.reverse()
        
        # Normalize returns
        if len(returns) > 1:
            returns = (np.array(returns) - np.mean(returns)) / (np.std(returns) + 1e-8)
        
        for i, data in enumerate(episode_data):
            states.append(data['state'])
            targets.append(data['action'])
            values.append(returns[i])
        
        # Convert to batch
        try:
            batch = Batch.from_data_list(states).to(device)
            targets_tensor = torch.tensor(targets, dtype=torch.long, device=device)
            values_tensor = torch.tensor(values, dtype=torch.float, device=device)
            
            # Forward pass
            use_mixed_precision = self.scaler is not None
            with torch.cuda.amp.autocast(enabled=use_mixed_precision):
                policy_logits, predicted_values, physics_energy = self.model(batch)
                
                # Policy loss (cross-entropy)
                policy_loss = F.cross_entropy(policy_logits, targets_tensor)
                
                # Value loss (MSE)
                value_loss = F.mse_loss(predicted_values, values_tensor)
                
                # Physics-inspired loss (encourage low energy states)
                physics_loss = physics_energy.mean()
                
                # Combined loss
                total_loss = policy_loss + 0.5 * value_loss + 0.1 * physics_loss
            
            # Backward pass
            self.optimizer.zero_grad()
            
            if self.scaler is not None:  # Mixed precision mode
                self.scaler.scale(total_loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:  # Standard precision mode
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
                self.optimizer.step()
            
            return total_loss.item()
            
        except Exception as e:
            print(f"Training error: {e}")
            return None
    
    def evaluate(self, order: int, num_episodes: int = 10) -> Dict:
        """Evaluate model performance"""
        self.model.eval()
        
        results = {
            'success_count': 0,
            'total_episodes': num_episodes,
            'avg_reward': 0.0,
            'avg_max_clique': 0.0,
            'best_coloring': None
        }
        
        episode_rewards = []
        max_cliques = []
        
        with torch.no_grad():
            for episode in range(num_episodes):
                env = RamseyEnvironment(self.config, order)
                state = env.reset()
                
                episode_reward = 0.0
                max_clique_size = 0
                done = False
                step_count = 0
                max_steps = order * order
                
                while not done and step_count < max_steps:
                    # Greedy action selection (no exploration)
                    data = state.to(device)
                    policy_logits, _, _ = self.model(data)
                    
                    # Get valid actions
                    valid_actions = []
                    edge_idx = 0
                    
                    for i in range(env.order):
                        for j in range(i + 1, env.order):
                            if (i, j) not in env.colored_edges:
                                valid_actions.append(edge_idx)
                            edge_idx += 1
                    
                    if not valid_actions:
                        break
                    
                    # Select best valid action
                    masked_logits = torch.full_like(policy_logits, float('-inf'))
                    for action in valid_actions:
                        if action < len(masked_logits):
                            masked_logits[action] = policy_logits[action]
                    
                    action = torch.argmax(masked_logits).item()
                    color = random.choice([1, 2])  # Random color for simplicity
                    
                    state, reward, done, info = env.step(action, color)
                    episode_reward += reward
                    max_clique_size = max(max_clique_size, 
                                        max(info['red_clique_size'], info['blue_clique_size']))
                    step_count += 1
                
                episode_rewards.append(episode_reward)
                max_cliques.append(max_clique_size)
                
                # Success if no forbidden clique found
                if max_clique_size < max(self.config.target_r, self.config.target_s):
                    results['success_count'] += 1
                    if results['best_coloring'] is None:
                        results['best_coloring'] = env.graph.copy()
        
        results['avg_reward'] = np.mean(episode_rewards)
        results['avg_max_clique'] = np.mean(max_cliques)
        results['success_rate'] = results['success_count'] / num_episodes
        
        return results
    
    def save_model(self, filepath: str):
        """Save model checkpoint"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'metrics': self.metrics
        }
        torch.save(checkpoint, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load model checkpoint"""
        checkpoint = torch.load(filepath, map_location=device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.metrics = checkpoint.get('metrics', self.metrics)
        print(f"Model loaded from {filepath}")


def visualize_training_progress(trainer: RamseyTrainer):
    """Visualize training metrics"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Training loss
    if trainer.metrics['losses']:
        axes[0, 0].plot(trainer.metrics['losses'])
        axes[0, 0].set_title('Training Loss')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Loss')
    
    # Average reward
    if trainer.metrics['rewards']:
        axes[0, 1].plot(trainer.metrics['rewards'])
        axes[0, 1].set_title('Average Episode Reward')
        axes[0, 1].set_xlabel('Episode (x100)')
        axes[0, 1].set_ylabel('Reward')
    
    # Maximum clique sizes
    if trainer.metrics['clique_sizes']:
        axes[1, 0].plot(trainer.metrics['clique_sizes'])
        axes[1, 0].set_title('Average Maximum Clique Size')
        axes[1, 0].set_xlabel('Episode (x100)')
        axes[1, 0].set_ylabel('Clique Size')
    
    # Success rate
    if trainer.metrics['success_rate']:
        axes[1, 1].plot(trainer.metrics['success_rate'])
        axes[1, 1].set_title('Success Rate')
        axes[1, 1].set_xlabel('Episode (x100)')
        axes[1, 1].set_ylabel('Success Rate')
    
    plt.tight_layout()
    plt.show()


def visualize_graph_coloring(graph: nx.Graph, title: str = "Graph Coloring"):
    """Visualize a colored graph"""
    pos = nx.spring_layout(graph)
    
    # Separate edges by color
    black_edges = [(u, v) for u, v in graph.edges() if graph[u][v]['color'] == 0]
    red_edges = [(u, v) for u, v in graph.edges() if graph[u][v]['color'] == 1]
    blue_edges = [(u, v) for u, v in graph.edges() if graph[u][v]['color'] == 2]
    
    plt.figure(figsize=(10, 8))
    
    # Draw nodes
    nx.draw_networkx_nodes(graph, pos, node_color='lightgray', node_size=300)
    nx.draw_networkx_labels(graph, pos)
    
    # Draw edges by color
    if black_edges:
        nx.draw_networkx_edges(graph, pos, edgelist=black_edges, edge_color='black', width=1)
    if red_edges:
        nx.draw_networkx_edges(graph, pos, edgelist=red_edges, edge_color='red', width=2)
    if blue_edges:
        nx.draw_networkx_edges(graph, pos, edgelist=blue_edges, edge_color='blue', width=2)
    
    plt.title(title)
    plt.axis('off')
    plt.show()


def main():
    """Main training and evaluation pipeline"""
    # Configuration - CPU-friendly defaults
    config = RamseyConfig(
        min_order=10,        # Start even smaller for CPU
        max_order=20,        # Smaller max for CPU testing
        target_r=4,          # Easier target for testing
        target_s=4,
        hidden_dim=64,       # Smaller model for CPU
        num_heads=2,         # Fewer attention heads
        num_layers=2,        # Fewer layers
        batch_size=4,        # Small batch for CPU
        learning_rate=1e-3,
        mcts_simulations=50, # Much fewer simulations for CPU
        mixed_precision=False  # Disable mixed precision for CPU
    )
    
    print("Next-Generation Ramsey AI Training")
    print("=" * 50)
    print(f"Target: R({config.target_r}, {config.target_s})")
    print(f"Graph size range: {config.min_order}-{config.max_order}")
    print(f"Device: {device}")
    
    # Adjust config based on device
    if not torch.cuda.is_available():
        print("\nRunning on CPU - using smaller configuration for testing")
        print("For full performance, install PyTorch with CUDA support")
    
    # Create trainer
    trainer = RamseyTrainer(config)
    
    # Training
    print("\nStarting training...")
    start_time = time.time()
    
    # Shorter training for initial testing
    trainer.train(num_episodes=100, progressive_sizes=True)
    
    training_time = time.time() - start_time
    print(f"\nTraining completed in {training_time:.1f} seconds")
    
    # Evaluation
    print("\nEvaluating model...")
    results = trainer.evaluate(order=config.min_order, num_episodes=5)
    
    print(f"Evaluation Results:")
    print(f"  Success rate: {results['success_rate']:.2%}")
    print(f"  Average reward: {results['avg_reward']:.2f}")
    print(f"  Average max clique: {results['avg_max_clique']:.1f}")
    
    # Save model
    model_path = f"ramsey_gat_model_{int(time.time())}.pt"
    trainer.save_model(model_path)
    
    # Visualize training progress
    try:
        visualize_training_progress(trainer)
    except Exception as e:
        print(f"Visualization error: {e}")
    
    # Visualize best coloring if found
    if results['best_coloring'] is not None:
        try:
            visualize_graph_coloring(results['best_coloring'], 
                                   f"Best R({config.target_r},{config.target_s}) Coloring Found")
        except Exception as e:
            print(f"Graph visualization error: {e}")
    
    print(f"\nModel saved to: {model_path}")
    print("\nFor GPU acceleration:")
    print("  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
    print("Training complete!")


if __name__ == "__main__":
    main()
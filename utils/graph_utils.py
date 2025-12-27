"""
Graph Utilities for GNN Processing
Converts NetworkX graphs to PyTorch Geometric format
"""

import torch
import numpy as np
import networkx as nx
import pandas as pd
from typing import Tuple, Dict, Optional


def prepare_gnn_data(
    graph: nx.DiGraph,
    node_features: pd.DataFrame
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Prepare graph data for GNN processing
    
    Args:
        graph: NetworkX directed graph
        node_features: DataFrame with node features
        
    Returns:
        x: Node feature tensor
        edge_index: Edge index tensor (2, num_edges)
        y: Label tensor
        train_mask: Training mask tensor
    """
    # Prepare node features
    feature_cols = ['revenue', 'debt_ratio', 'profit_margin', 
                    'years_in_business', 'credit_score', 'default_risk']
    
    # Normalize features
    x_data = node_features[feature_cols].values.astype(np.float32)
    x_mean = x_data.mean(axis=0)
    x_std = x_data.std(axis=0) + 1e-8
    x_normalized = (x_data - x_mean) / x_std
    
    x = torch.tensor(x_normalized, dtype=torch.float32)
    
    # Prepare edge index
    edges = list(graph.edges())
    if len(edges) > 0:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
    
    # Prepare labels
    y = torch.tensor(node_features['is_default'].values, dtype=torch.long)
    
    # Create train/test mask (80/20 split)
    n_nodes = len(node_features)
    indices = np.random.permutation(n_nodes)
    train_size = int(0.8 * n_nodes)
    
    train_mask = torch.zeros(n_nodes, dtype=torch.bool)
    train_mask[indices[:train_size]] = True
    
    test_mask = torch.zeros(n_nodes, dtype=torch.bool)
    test_mask[indices[train_size:]] = True
    
    return x, edge_index, y, train_mask, test_mask


def get_node_colors(
    node_features: pd.DataFrame,
    predictions: Optional[np.ndarray] = None
) -> list:
    """
    Get colors for nodes based on default status or predictions
    
    Args:
        node_features: DataFrame with node features
        predictions: Optional prediction probabilities
        
    Returns:
        List of color strings for each node
    """
    colors = []
    
    if predictions is not None:
        # Color based on prediction probability
        for prob in predictions:
            if prob > 0.7:
                colors.append('#ef4444')  # Red - high risk
            elif prob > 0.4:
                colors.append('#f97316')  # Orange - medium risk
            elif prob > 0.2:
                colors.append('#eab308')  # Yellow - low-medium risk
            else:
                colors.append('#22c55e')  # Green - low risk
    else:
        # Color based on actual default status
        for _, row in node_features.iterrows():
            if row['is_default'] == 1:
                colors.append('#ef4444')  # Red - defaulted
            else:
                colors.append('#22c55e')  # Green - healthy
    
    return colors


def get_edge_colors(graph: nx.DiGraph) -> list:
    """Get colors for edges based on relationship type"""
    edge_colors = []
    
    color_map = {
        'loan': '#3b82f6',        # Blue
        'guarantee': '#8b5cf6',   # Purple
        'supply_chain': '#06b6d4', # Cyan
        'investment': '#10b981',   # Emerald
        'subsidiary': '#6366f1',   # Indigo
    }
    
    for u, v in graph.edges():
        rel_type = graph.edges[u, v].get('relationship_type', 'loan')
        edge_colors.append(color_map.get(rel_type, '#94a3b8'))
    
    return edge_colors


def calculate_graph_statistics(
    graph: nx.DiGraph,
    node_features: pd.DataFrame
) -> Dict:
    """Calculate various statistics about the graph"""
    
    # Basic stats
    n_nodes = graph.number_of_nodes()
    n_edges = graph.number_of_edges()
    density = nx.density(graph)
    
    # Default statistics
    default_rate = node_features['is_default'].mean()
    
    # Degree statistics
    in_degrees = [d for n, d in graph.in_degree()]
    out_degrees = [d for n, d in graph.out_degree()]
    
    # Clustering coefficient (for undirected version)
    undirected = graph.to_undirected()
    avg_clustering = nx.average_clustering(undirected)
    
    return {
        'num_nodes': n_nodes,
        'num_edges': n_edges,
        'density': density,
        'default_rate': default_rate,
        'avg_in_degree': np.mean(in_degrees),
        'avg_out_degree': np.mean(out_degrees),
        'max_in_degree': max(in_degrees) if in_degrees else 0,
        'max_out_degree': max(out_degrees) if out_degrees else 0,
        'avg_clustering': avg_clustering,
    }

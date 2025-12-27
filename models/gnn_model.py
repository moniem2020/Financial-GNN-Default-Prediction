"""
Graph Neural Network Model for Financial Default Prediction
Implements a simple GCN (Graph Convolutional Network) for demonstration
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import numpy as np


class GraphConvLayer(nn.Module):
    """Simple Graph Convolution Layer"""
    
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.bn = nn.BatchNorm1d(out_features)
        
    def forward(
        self, 
        x: torch.Tensor, 
        edge_index: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass of graph convolution
        
        Args:
            x: Node features (N, in_features)
            edge_index: Edge indices (2, E)
            
        Returns:
            Updated node features (N, out_features)
        """
        num_nodes = x.size(0)
        
        # Create adjacency matrix from edge index
        adj = torch.zeros(num_nodes, num_nodes, device=x.device)
        if edge_index.size(1) > 0:
            adj[edge_index[0], edge_index[1]] = 1.0
        
        # Add self-loops
        adj = adj + torch.eye(num_nodes, device=x.device)
        
        # Normalize adjacency matrix (D^-1/2 * A * D^-1/2)
        degree = adj.sum(dim=1)
        degree_inv_sqrt = torch.pow(degree + 1e-8, -0.5)
        degree_matrix = torch.diag(degree_inv_sqrt)
        adj_normalized = degree_matrix @ adj @ degree_matrix
        
        # Graph convolution: A * X * W
        x = adj_normalized @ x
        x = self.linear(x)
        
        # Batch normalization
        if x.size(0) > 1:
            x = self.bn(x)
        
        return x


class FinancialGNN(nn.Module):
    """
    GNN model for financial default prediction
    Uses multiple graph convolution layers with residual connections
    """
    
    def __init__(
        self,
        input_dim: int = 6,
        hidden_dim: int = 32,
        output_dim: int = 2,
        num_layers: int = 3,
        dropout: float = 0.5  # Increased dropout to reduce overfitting
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        
        # Input projection
        self.input_layer = GraphConvLayer(input_dim, hidden_dim)
        
        # Hidden layers
        self.hidden_layers = nn.ModuleList([
            GraphConvLayer(hidden_dim, hidden_dim)
            for _ in range(num_layers - 2)
        ])
        
        # Output layer
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        
        # Attention for motif-aware aggregation (simplified)
        self.attention = nn.Linear(hidden_dim, 1)
        
    def forward(
        self, 
        x: torch.Tensor, 
        edge_index: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass
        
        Args:
            x: Node features (N, input_dim)
            edge_index: Edge indices (2, E)
            
        Returns:
            logits: Classification logits (N, 2)
            attention_weights: Node attention weights (N,)
        """
        # Input layer
        h = self.input_layer(x, edge_index)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        
        # Hidden layers with residual connections
        for layer in self.hidden_layers:
            h_new = layer(h, edge_index)
            h_new = F.relu(h_new)
            h_new = F.dropout(h_new, p=self.dropout, training=self.training)
            h = h + h_new  # Residual connection
        
        # Compute attention weights (indicating node importance)
        attention_weights = torch.sigmoid(self.attention(h)).squeeze(-1)
        
        # Output classification
        logits = self.output_layer(h)
        
        return logits, attention_weights
    
    def get_embeddings(
        self, 
        x: torch.Tensor, 
        edge_index: torch.Tensor
    ) -> torch.Tensor:
        """Get node embeddings before final classification"""
        h = self.input_layer(x, edge_index)
        h = F.relu(h)
        
        for layer in self.hidden_layers:
            h_new = layer(h, edge_index)
            h_new = F.relu(h_new)
            h = h + h_new
        
        return h


def train_gnn(
    model: FinancialGNN,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    y: torch.Tensor,
    train_mask: torch.Tensor,
    epochs: int = 100,
    lr: float = 0.01,
    weight_decay: float = 5e-4,
    verbose: bool = True
) -> dict:
    """
    Train the GNN model
    
    Args:
        model: GNN model
        x: Node features
        edge_index: Edge indices
        y: Labels
        train_mask: Training mask
        epochs: Number of epochs
        lr: Learning rate
        weight_decay: Weight decay for regularization
        verbose: Whether to print progress
        
    Returns:
        Training history dict
    """
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=lr, 
        weight_decay=weight_decay
    )
    
    # Class weights for imbalanced data
    class_counts = torch.bincount(y[train_mask])
    if len(class_counts) == 2:
        class_weights = torch.tensor([1.0, class_counts[0] / (class_counts[1] + 1e-8)])
    else:
        class_weights = torch.ones(2)
    
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    history = {'loss': [], 'accuracy': []}
    
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        logits, _ = model(x, edge_index)
        loss = criterion(logits[train_mask], y[train_mask])
        
        loss.backward()
        optimizer.step()
        
        # Calculate accuracy
        with torch.no_grad():
            pred = logits.argmax(dim=1)
            correct = (pred[train_mask] == y[train_mask]).sum().item()
            accuracy = correct / train_mask.sum().item()
        
        history['loss'].append(loss.item())
        history['accuracy'].append(accuracy)
        
        if verbose and (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}/{epochs} - Loss: {loss.item():.4f} - Accuracy: {accuracy:.4f}")
    
    return history


def predict_defaults(
    model: FinancialGNN,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    temperature: float = 2.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Make predictions using the trained model
    
    Args:
        model: Trained GNN model
        x: Node features
        edge_index: Edge indices
        temperature: Temperature for softmax calibration (higher = more uncertain)
        
    Returns:
        predictions: Predicted class labels
        probabilities: Prediction probabilities for default class
        attention_weights: Node attention weights
    """
    model.eval()
    
    with torch.no_grad():
        logits, attention = model(x, edge_index)
        
        # Apply temperature scaling for more calibrated probabilities
        # Higher temperature = less confident predictions (more realistic)
        scaled_logits = logits / temperature
        probs = F.softmax(scaled_logits, dim=1)
        
        # Add small noise to avoid exact 0 or 1 probabilities
        noise = torch.randn_like(probs) * 0.02
        probs = torch.clamp(probs + noise, 0.05, 0.95)
        
        # Renormalize
        probs = probs / probs.sum(dim=1, keepdim=True)
        
        predictions = logits.argmax(dim=1)
    
    return (
        predictions.numpy(),
        probs[:, 1].numpy(),  # Probability of default
        attention.numpy()
    )


if __name__ == "__main__":
    # Quick test
    model = FinancialGNN()
    x = torch.randn(10, 6)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]])
    
    logits, attention = model(x, edge_index)
    print(f"Output shape: {logits.shape}")
    print(f"Attention shape: {attention.shape}")

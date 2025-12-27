"""
Synthetic Financial Network Data Generator
Generates realistic financial networks with companies, relationships, and default labels
"""

import numpy as np
import networkx as nx
import pandas as pd
from typing import Tuple, Dict, List
import random


class FinancialNetworkGenerator:
    """Generate synthetic financial network data for GNN demonstration"""
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        np.random.seed(seed)
        random.seed(seed)
        
        # Company types and their characteristics
        self.company_types = {
            'large_corp': {'size_range': (1000, 5000), 'default_prob': 0.05},
            'medium_corp': {'size_range': (100, 1000), 'default_prob': 0.10},
            'small_business': {'size_range': (10, 100), 'default_prob': 0.20},
            'startup': {'size_range': (1, 10), 'default_prob': 0.35},
        }
        
        # Relationship types
        self.relationship_types = [
            'loan', 'guarantee', 'supply_chain', 'investment', 'subsidiary'
        ]
    
    def generate_network(
        self, 
        n_companies: int = 50, 
        edge_prob: float = 0.15,
        add_motifs: bool = True
    ) -> Tuple[nx.DiGraph, pd.DataFrame, pd.DataFrame]:
        """
        Generate a synthetic financial network
        
        Args:
            n_companies: Number of companies in the network
            edge_prob: Probability of edge between any two nodes
            add_motifs: Whether to add risk-indicating motifs
            
        Returns:
            graph: NetworkX directed graph
            node_features: DataFrame with node features
            edge_features: DataFrame with edge features
        """
        # Create base graph
        G = nx.erdos_renyi_graph(n_companies, edge_prob, directed=True, seed=self.seed)
        
        # Generate company features
        node_features = self._generate_node_features(n_companies)
        
        # Add motif patterns if requested
        if add_motifs:
            G, motif_nodes = self._add_risk_motifs(G, n_companies)
            # Increase default probability for motif nodes
            node_features.loc[motif_nodes, 'default_risk'] *= 1.5
        
        # Generate edge features
        edge_features = self._generate_edge_features(G)
        
        # Calculate final default labels based on features and connectivity
        node_features['is_default'] = self._calculate_defaults(G, node_features)
        
        # Add features to graph
        for node in G.nodes():
            for col in node_features.columns:
                G.nodes[node][col] = node_features.loc[node, col]
        
        for edge in G.edges():
            edge_data = edge_features[
                (edge_features['source'] == edge[0]) & 
                (edge_features['target'] == edge[1])
            ]
            if len(edge_data) > 0:
                for col in edge_data.columns:
                    if col not in ['source', 'target']:
                        G.edges[edge][col] = edge_data[col].values[0]
        
        return G, node_features, edge_features
    
    def _generate_node_features(self, n_companies: int) -> pd.DataFrame:
        """Generate features for each company node"""
        features = []
        
        for i in range(n_companies):
            company_type = random.choice(list(self.company_types.keys()))
            config = self.company_types[company_type]
            
            features.append({
                'node_id': i,
                'company_type': company_type,
                'revenue': np.random.uniform(*config['size_range']),
                'debt_ratio': np.random.uniform(0.1, 0.9),
                'profit_margin': np.random.uniform(-0.2, 0.5),
                'years_in_business': np.random.randint(1, 50),
                'credit_score': np.random.randint(300, 850),
                'default_risk': config['default_prob'] * np.random.uniform(0.5, 2.0),
            })
        
        return pd.DataFrame(features).set_index('node_id')
    
    def _generate_edge_features(self, G: nx.DiGraph) -> pd.DataFrame:
        """Generate features for each relationship edge"""
        edges = []
        
        for source, target in G.edges():
            edges.append({
                'source': source,
                'target': target,
                'relationship_type': random.choice(self.relationship_types),
                'transaction_volume': np.random.uniform(10, 1000),
                'relationship_duration': np.random.randint(1, 20),
                'risk_score': np.random.uniform(0, 1),
            })
        
        return pd.DataFrame(edges)
    
    def _add_risk_motifs(
        self, 
        G: nx.DiGraph, 
        n_companies: int
    ) -> Tuple[nx.DiGraph, List[int]]:
        """Add common risk-indicating motifs to the graph"""
        motif_nodes = set()
        
        # Add triangular motifs (circular transactions - often risky)
        n_triangles = max(1, n_companies // 10)
        for _ in range(n_triangles):
            nodes = random.sample(range(n_companies), 3)
            G.add_edge(nodes[0], nodes[1])
            G.add_edge(nodes[1], nodes[2])
            G.add_edge(nodes[2], nodes[0])
            motif_nodes.update(nodes)
        
        # Add star patterns (concentrated risk)
        n_stars = max(1, n_companies // 15)
        for _ in range(n_stars):
            available = list(set(range(n_companies)) - motif_nodes)
            if len(available) >= 4:
                center = random.choice(available)
                spokes = random.sample([n for n in available if n != center], min(3, len(available)-1))
                for spoke in spokes:
                    G.add_edge(center, spoke)
                    G.add_edge(spoke, center)
                motif_nodes.add(center)
                motif_nodes.update(spokes)
        
        return G, list(motif_nodes)
    
    def _calculate_defaults(
        self, 
        G: nx.DiGraph, 
        node_features: pd.DataFrame
    ) -> List[int]:
        """Calculate which companies default based on features and network structure"""
        defaults = []
        
        for node in range(len(node_features)):
            # Base probability from features
            base_prob = node_features.loc[node, 'default_risk']
            
            # Adjust based on neighbors (contagion effect)
            if node in G.nodes():
                neighbors = list(G.predecessors(node)) + list(G.successors(node))
                if neighbors:
                    neighbor_risks = node_features.loc[neighbors, 'default_risk'].mean()
                    base_prob = 0.7 * base_prob + 0.3 * neighbor_risks
            
            # High debt ratio increases default probability
            if node_features.loc[node, 'debt_ratio'] > 0.7:
                base_prob *= 1.3
            
            # Low credit score increases default probability
            if node_features.loc[node, 'credit_score'] < 500:
                base_prob *= 1.2
            
            defaults.append(1 if np.random.random() < min(base_prob, 0.9) else 0)
        
        return defaults


def generate_demo_network(
    n_companies: int = 50,
    seed: int = 42
) -> Tuple[nx.DiGraph, pd.DataFrame, pd.DataFrame]:
    """Convenience function to generate demo network"""
    generator = FinancialNetworkGenerator(seed=seed)
    return generator.generate_network(n_companies=n_companies)


if __name__ == "__main__":
    # Test the generator
    G, nodes, edges = generate_demo_network(30)
    print(f"Generated network with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    print(f"Default rate: {nodes['is_default'].mean():.2%}")

"""
Motif Detection for Financial Networks
Identifies common structural patterns that indicate financial risk
"""

import networkx as nx
from typing import List, Dict, Tuple, Set
from collections import defaultdict


class MotifDetector:
    """Detect and analyze motifs in financial networks"""
    
    # Define motif types and their risk implications
    MOTIF_RISK_INFO = {
        'triangle': {
            'description': 'Circular Transaction Pattern',
            'risk_level': 'High',
            'explanation': 'Triangular patterns often indicate circular transactions or mutual guarantees, which can amplify default risk across all connected entities.',
            'color': '#ef4444'  # Red
        },
        'star': {
            'description': 'Concentrated Dependency Pattern',
            'risk_level': 'Medium-High',
            'explanation': 'Star patterns show concentrated dependencies on a central entity. If the center fails, all connected entities are at risk.',
            'color': '#f97316'  # Orange
        },
        'chain': {
            'description': 'Sequential Risk Chain',
            'risk_level': 'Medium',
            'explanation': 'Chain patterns represent sequential dependencies. Risk propagates along the chain if any node defaults.',
            'color': '#eab308'  # Yellow
        },
        'bidirectional': {
            'description': 'Mutual Dependency',
            'risk_level': 'Medium',
            'explanation': 'Bidirectional edges indicate mutual dependencies between two entities, creating paired risk exposure.',
            'color': '#22c55e'  # Green
        }
    }
    
    def __init__(self, graph: nx.DiGraph):
        self.graph = graph
        self.motifs = defaultdict(list)
        
    def detect_all_motifs(self) -> Dict[str, List]:
        """Detect all types of motifs in the graph"""
        self.motifs['triangle'] = self._detect_triangles()
        self.motifs['star'] = self._detect_stars()
        self.motifs['chain'] = self._detect_chains()
        self.motifs['bidirectional'] = self._detect_bidirectional()
        
        return dict(self.motifs)
    
    def _detect_triangles(self) -> List[Tuple[int, int, int]]:
        """Detect triangular patterns (3-node cycles)"""
        triangles = []
        nodes = list(self.graph.nodes())
        
        for i, node1 in enumerate(nodes):
            successors = set(self.graph.successors(node1))
            for node2 in successors:
                for node3 in self.graph.successors(node2):
                    if node3 in self.graph.predecessors(node1) and node3 != node1:
                        triangle = tuple(sorted([node1, node2, node3]))
                        if triangle not in triangles:
                            triangles.append(triangle)
        
        return triangles[:10]  # Limit to first 10
    
    def _detect_stars(self, min_degree: int = 3) -> List[Dict]:
        """Detect star patterns (high-degree central nodes)"""
        stars = []
        
        for node in self.graph.nodes():
            out_degree = self.graph.out_degree(node)
            in_degree = self.graph.in_degree(node)
            
            if out_degree >= min_degree:
                stars.append({
                    'center': node,
                    'type': 'outgoing',
                    'spokes': list(self.graph.successors(node)),
                    'degree': out_degree
                })
            
            if in_degree >= min_degree:
                stars.append({
                    'center': node,
                    'type': 'incoming',
                    'spokes': list(self.graph.predecessors(node)),
                    'degree': in_degree
                })
        
        # Sort by degree and return top stars
        stars.sort(key=lambda x: x['degree'], reverse=True)
        return stars[:5]
    
    def _detect_chains(self, min_length: int = 3) -> List[List[int]]:
        """Detect chain patterns (sequential node connections)"""
        chains = []
        visited_in_chains = set()
        
        for start_node in self.graph.nodes():
            if start_node in visited_in_chains:
                continue
                
            # Try to build a chain from this node
            chain = [start_node]
            current = start_node
            
            while True:
                successors = [s for s in self.graph.successors(current) 
                            if s not in chain and self.graph.out_degree(s) <= 2]
                if not successors:
                    break
                    
                next_node = successors[0]
                chain.append(next_node)
                current = next_node
                
                if len(chain) >= 5:  # Limit chain length
                    break
            
            if len(chain) >= min_length:
                chains.append(chain)
                visited_in_chains.update(chain)
        
        return chains[:5]
    
    def _detect_bidirectional(self) -> List[Tuple[int, int]]:
        """Detect bidirectional edges (mutual relationships)"""
        bidirectional = []
        
        for edge in self.graph.edges():
            reverse = (edge[1], edge[0])
            if self.graph.has_edge(*reverse):
                pair = tuple(sorted([edge[0], edge[1]]))
                if pair not in bidirectional:
                    bidirectional.append(pair)
        
        return bidirectional[:10]
    
    def get_motif_nodes(self) -> Set[int]:
        """Get all nodes that are part of any motif"""
        motif_nodes = set()
        
        for triangle in self.motifs.get('triangle', []):
            motif_nodes.update(triangle)
        
        for star in self.motifs.get('star', []):
            motif_nodes.add(star['center'])
            motif_nodes.update(star['spokes'])
        
        for chain in self.motifs.get('chain', []):
            motif_nodes.update(chain)
        
        for pair in self.motifs.get('bidirectional', []):
            motif_nodes.update(pair)
        
        return motif_nodes
    
    def get_motif_summary(self) -> Dict:
        """Get summary statistics about detected motifs"""
        return {
            'triangles_count': len(self.motifs.get('triangle', [])),
            'stars_count': len(self.motifs.get('star', [])),
            'chains_count': len(self.motifs.get('chain', [])),
            'bidirectional_count': len(self.motifs.get('bidirectional', [])),
            'total_motif_nodes': len(self.get_motif_nodes()),
            'motif_info': self.MOTIF_RISK_INFO
        }


if __name__ == "__main__":
    from data_generator import generate_demo_network
    
    G, nodes, edges = generate_demo_network(50)
    detector = MotifDetector(G)
    motifs = detector.detect_all_motifs()
    summary = detector.get_motif_summary()
    
    print("Motif Detection Summary:")
    for key, value in summary.items():
        if key != 'motif_info':
            print(f"  {key}: {value}")

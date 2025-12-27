"""
Visualization Components for GNN Demo
Creates interactive Plotly charts and network visualizations
"""

import plotly.graph_objects as go
import plotly.express as px
import networkx as nx
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple


# Color palette
COLORS = {
    'primary': '#6366f1',      # Indigo
    'secondary': '#8b5cf6',    # Violet
    'success': '#22c55e',      # Green
    'danger': '#ef4444',       # Red
    'warning': '#f97316',      # Orange
    'info': '#06b6d4',         # Cyan
    'background': '#0f172a',   # Slate 900
    'surface': '#1e293b',      # Slate 800
    'text': '#f8fafc',         # Slate 50
    'muted': '#94a3b8',        # Slate 400
}


def create_network_visualization(
    graph: nx.DiGraph,
    node_features: pd.DataFrame,
    predictions: Optional[np.ndarray] = None,
    highlight_nodes: Optional[set] = None,
    title: str = "Financial Network Graph"
) -> go.Figure:
    """
    Create an interactive network visualization
    
    Args:
        graph: NetworkX graph
        node_features: DataFrame with node features
        predictions: Optional prediction probabilities
        highlight_nodes: Set of nodes to highlight
        title: Chart title
        
    Returns:
        Plotly figure
    """
    # Get layout
    pos = nx.spring_layout(graph, seed=42, k=2/np.sqrt(len(graph.nodes())))
    
    # Create edge traces
    edge_x = []
    edge_y = []
    for edge in graph.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color=COLORS['muted']),
        hoverinfo='none',
        mode='lines',
        opacity=0.5
    )
    
    # Create node traces
    node_x = []
    node_y = []
    node_colors = []
    node_sizes = []
    node_text = []
    
    for node in graph.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        
        # Color based on predictions or actual defaults
        if predictions is not None:
            prob = predictions[node]
            if prob > 0.7:
                color = COLORS['danger']
            elif prob > 0.4:
                color = COLORS['warning']
            else:
                color = COLORS['success']
        else:
            color = COLORS['danger'] if node_features.loc[node, 'is_default'] == 1 else COLORS['success']
        
        # Highlight motif nodes
        if highlight_nodes and node in highlight_nodes:
            node_sizes.append(20)
        else:
            node_sizes.append(12)
        
        node_colors.append(color)
        
        # Hover text
        text = f"<b>Company {node}</b><br>"
        text += f"Type: {node_features.loc[node, 'company_type']}<br>"
        text += f"Revenue: ${node_features.loc[node, 'revenue']:.0f}M<br>"
        text += f"Debt Ratio: {node_features.loc[node, 'debt_ratio']:.1%}<br>"
        text += f"Credit Score: {node_features.loc[node, 'credit_score']:.0f}<br>"
        if predictions is not None:
            text += f"<b>Default Risk: {predictions[node]:.1%}</b>"
        else:
            text += f"Default: {'Yes' if node_features.loc[node, 'is_default'] == 1 else 'No'}"
        node_text.append(text)
    
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        text=node_text,
        marker=dict(
            size=node_sizes,
            color=node_colors,
            line=dict(width=1, color='white'),
        )
    )
    
    # Create figure
    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            title=dict(
                text=title,
                font=dict(size=20, color=COLORS['text']),
                x=0.5
            ),
            showlegend=False,
            hovermode='closest',
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            paper_bgcolor=COLORS['background'],
            plot_bgcolor=COLORS['background'],
            margin=dict(l=20, r=20, t=60, b=20),
            height=500
        )
    )
    
    return fig


def create_training_chart(history: Dict) -> go.Figure:
    """Create training progress chart"""
    epochs = list(range(1, len(history['loss']) + 1))
    
    fig = go.Figure()
    
    # Loss trace
    fig.add_trace(go.Scatter(
        x=epochs, y=history['loss'],
        name='Loss',
        line=dict(color=COLORS['danger'], width=2),
        yaxis='y'
    ))
    
    # Accuracy trace
    fig.add_trace(go.Scatter(
        x=epochs, y=history['accuracy'],
        name='Accuracy',
        line=dict(color=COLORS['success'], width=2),
        yaxis='y2'
    ))
    
    fig.update_layout(
        title=dict(
            text='Training Progress',
            font=dict(size=18, color=COLORS['text']),
            x=0.5
        ),
        xaxis=dict(
            title='Epoch',
            gridcolor=COLORS['surface'],
            color=COLORS['muted']
        ),
        yaxis=dict(
            title='Loss',
            gridcolor=COLORS['surface'],
            color=COLORS['muted']
        ),
        yaxis2=dict(
            title='Accuracy',
            overlaying='y',
            side='right',
            gridcolor=COLORS['surface'],
            color=COLORS['muted']
        ),
        paper_bgcolor=COLORS['background'],
        plot_bgcolor=COLORS['background'],
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='center',
            x=0.5,
            font=dict(color=COLORS['text'])
        ),
        height=350,
        margin=dict(l=60, r=60, t=80, b=40)
    )
    
    return fig


def create_prediction_chart(
    predictions: np.ndarray,
    actual: np.ndarray
) -> go.Figure:
    """Create prediction distribution chart"""
    
    fig = go.Figure()
    
    # Non-default predictions
    non_default_preds = predictions[actual == 0]
    fig.add_trace(go.Histogram(
        x=non_default_preds,
        name='Healthy Companies',
        marker_color=COLORS['success'],
        opacity=0.7,
        nbinsx=20
    ))
    
    # Default predictions
    default_preds = predictions[actual == 1]
    fig.add_trace(go.Histogram(
        x=default_preds,
        name='Defaulted Companies',
        marker_color=COLORS['danger'],
        opacity=0.7,
        nbinsx=20
    ))
    
    fig.update_layout(
        title=dict(
            text='Prediction Distribution by Actual Status',
            font=dict(size=18, color=COLORS['text']),
            x=0.5
        ),
        xaxis=dict(
            title='Default Probability',
            gridcolor=COLORS['surface'],
            color=COLORS['muted']
        ),
        yaxis=dict(
            title='Count',
            gridcolor=COLORS['surface'],
            color=COLORS['muted']
        ),
        barmode='overlay',
        paper_bgcolor=COLORS['background'],
        plot_bgcolor=COLORS['background'],
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='center',
            x=0.5,
            font=dict(color=COLORS['text'])
        ),
        height=350,
        margin=dict(l=60, r=20, t=80, b=40)
    )
    
    return fig


def create_motif_visualization(
    motif_summary: Dict
) -> go.Figure:
    """Create motif type distribution chart"""
    
    motif_types = ['Triangles', 'Stars', 'Chains', 'Bidirectional']
    counts = [
        motif_summary['triangles_count'],
        motif_summary['stars_count'],
        motif_summary['chains_count'],
        motif_summary['bidirectional_count']
    ]
    colors = [COLORS['danger'], COLORS['warning'], COLORS['info'], COLORS['success']]
    
    fig = go.Figure(data=[
        go.Bar(
            x=motif_types,
            y=counts,
            marker_color=colors,
            text=counts,
            textposition='auto',
            textfont=dict(color='white', size=14)
        )
    ])
    
    fig.update_layout(
        title=dict(
            text='Detected Motif Patterns',
            font=dict(size=18, color=COLORS['text']),
            x=0.5
        ),
        xaxis=dict(
            title='Motif Type',
            gridcolor=COLORS['surface'],
            color=COLORS['muted']
        ),
        yaxis=dict(
            title='Count',
            gridcolor=COLORS['surface'],
            color=COLORS['muted']
        ),
        paper_bgcolor=COLORS['background'],
        plot_bgcolor=COLORS['background'],
        height=300,
        margin=dict(l=60, r=20, t=60, b=40)
    )
    
    return fig


def create_feature_importance_chart() -> go.Figure:
    """Create feature importance visualization (simulated)"""
    
    features = ['Debt Ratio', 'Credit Score', 'Network Motifs', 
                'Neighbor Risk', 'Revenue', 'Profit Margin']
    importance = [0.28, 0.24, 0.18, 0.15, 0.09, 0.06]
    
    fig = go.Figure(data=[
        go.Bar(
            x=importance,
            y=features,
            orientation='h',
            marker=dict(
                color=importance,
                colorscale=[[0, COLORS['info']], [1, COLORS['primary']]],
            ),
            text=[f'{v:.0%}' for v in importance],
            textposition='outside',
            textfont=dict(color=COLORS['text'], size=12)
        )
    ])
    
    fig.update_layout(
        title=dict(
            text='Feature Importance in Default Prediction',
            font=dict(size=18, color=COLORS['text']),
            x=0.5
        ),
        xaxis=dict(
            title='Importance',
            gridcolor=COLORS['surface'],
            color=COLORS['muted'],
            range=[0, 0.35]
        ),
        yaxis=dict(
            gridcolor=COLORS['surface'],
            color=COLORS['muted']
        ),
        paper_bgcolor=COLORS['background'],
        plot_bgcolor=COLORS['background'],
        height=300,
        margin=dict(l=120, r=60, t=60, b=40)
    )
    
    return fig


def create_network_stats_chart(stats: Dict) -> go.Figure:
    """Create network statistics gauge charts"""
    
    fig = go.Figure()
    
    # Default Rate Gauge
    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=stats['default_rate'] * 100,
        title={'text': "Default Rate (%)", 'font': {'color': COLORS['text'], 'size': 14}},
        number={'suffix': '%', 'font': {'color': COLORS['text']}},
        gauge={
            'axis': {'range': [0, 50], 'tickcolor': COLORS['muted']},
            'bar': {'color': COLORS['danger']},
            'bgcolor': COLORS['surface'],
            'bordercolor': COLORS['muted'],
            'steps': [
                {'range': [0, 15], 'color': COLORS['success']},
                {'range': [15, 30], 'color': COLORS['warning']},
                {'range': [30, 50], 'color': COLORS['danger']}
            ],
        },
        domain={'x': [0, 0.45], 'y': [0, 1]}
    ))
    
    # Network Density Gauge
    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=stats['density'] * 100,
        title={'text': "Network Density (%)", 'font': {'color': COLORS['text'], 'size': 14}},
        number={'suffix': '%', 'font': {'color': COLORS['text']}},
        gauge={
            'axis': {'range': [0, 50], 'tickcolor': COLORS['muted']},
            'bar': {'color': COLORS['primary']},
            'bgcolor': COLORS['surface'],
            'bordercolor': COLORS['muted'],
        },
        domain={'x': [0.55, 1], 'y': [0, 1]}
    ))
    
    fig.update_layout(
        paper_bgcolor=COLORS['background'],
        height=250,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    return fig

"""
Financial Default Prediction via Motif-Preserving GNN
Interactive Demo Application

Run with: streamlit run app.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import torch
import time

# Import custom modules
from utils.data_generator import FinancialNetworkGenerator
from utils.motif_detector import MotifDetector
from utils.graph_utils import prepare_gnn_data, calculate_graph_statistics
from models.gnn_model import FinancialGNN, train_gnn, predict_defaults
from components.visualizations import (
    create_network_visualization,
    create_training_chart,
    create_prediction_chart,
    create_motif_visualization,
    create_feature_importance_chart,
    create_network_stats_chart
)


# Page configuration
st.set_page_config(
    page_title="Financial GNN Demo",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load custom CSS
def load_css():
    with open("static/style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

try:
    load_css()
except:
    pass  # CSS file might not exist in some environments


def main():
    # Header
    st.markdown("""
        <div style="text-align: center; padding: 20px 0;">
            <h1 style="font-size: 2.5rem; margin-bottom: 10px;">
                🔮 Financial Default Prediction
            </h1>
            <p style="color: #94a3b8; font-size: 1.2rem;">
                via Motif-Preserving Graph Neural Networks
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### ⚙️ Configuration")
        
        n_companies = st.slider(
            "Number of Companies",
            min_value=20,
            max_value=100,
            value=50,
            step=10,
            help="Number of companies in the financial network"
        )
        
        seed = st.number_input(
            "Random Seed",
            min_value=1,
            max_value=9999,
            value=42,
            help="Seed for reproducible results"
        )
        
        st.markdown("---")
        st.markdown("### 🧠 GNN Settings")
        
        hidden_dim = st.select_slider(
            "Hidden Dimensions",
            options=[16, 32, 64, 128],
            value=32
        )
        
        num_layers = st.slider(
            "GNN Layers",
            min_value=2,
            max_value=5,
            value=3
        )
        
        epochs = st.slider(
            "Training Epochs",
            min_value=50,
            max_value=300,
            value=100,
            step=25
        )
        
        st.markdown("---")
        st.markdown("""
            <div style="text-align: center; padding: 20px 0;">
                <p style="color: #6366f1; font-size: 0.9rem;">
                    Built with ❤️ for GNN Research
                </p>
            </div>
        """, unsafe_allow_html=True)
    
    # Initialize session state
    if 'graph' not in st.session_state:
        st.session_state.graph = None
        st.session_state.node_features = None
        st.session_state.edge_features = None
        st.session_state.model = None
        st.session_state.predictions = None
        st.session_state.history = None
    
    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Network Generation", 
        "🔍 Motif Analysis",
        "🤖 GNN Prediction",
        "📚 Learn More"
    ])
    
    # Tab 1: Network Generation
    with tab1:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("### 🏢 Financial Network")
            
            if st.button("🔄 Generate New Network", key="gen_network"):
                with st.spinner("Generating financial network..."):
                    generator = FinancialNetworkGenerator(seed=seed)
                    graph, node_features, edge_features = generator.generate_network(
                        n_companies=n_companies
                    )
                    
                    st.session_state.graph = graph
                    st.session_state.node_features = node_features
                    st.session_state.edge_features = edge_features
                    st.session_state.model = None
                    st.session_state.predictions = None
                    
                    time.sleep(0.5)  # Visual effect
                
                st.success(f"✅ Generated network with {n_companies} companies!")
            
            if st.session_state.graph is not None:
                fig = create_network_visualization(
                    st.session_state.graph,
                    st.session_state.node_features,
                    title="Financial Network - Node Color = Default Status"
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### 📈 Network Statistics")
            
            if st.session_state.graph is not None:
                stats = calculate_graph_statistics(
                    st.session_state.graph,
                    st.session_state.node_features
                )
                
                # Stats metrics
                st.metric("🏢 Companies", stats['num_nodes'])
                st.metric("🔗 Relationships", stats['num_edges'])
                st.metric("📊 Default Rate", f"{stats['default_rate']:.1%}")
                st.metric("🕸️ Clustering", f"{stats['avg_clustering']:.3f}")
                
                st.markdown("---")
                
                # Gauge charts
                fig = create_network_stats_chart(stats)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("👆 Generate a network to see statistics")
    
    # Tab 2: Motif Analysis
    with tab2:
        st.markdown("### 🔍 Structural Pattern (Motif) Detection")
        
        if st.session_state.graph is None:
            st.warning("⚠️ Please generate a network first!")
        else:
            # Detect motifs
            detector = MotifDetector(st.session_state.graph)
            motifs = detector.detect_all_motifs()
            summary = detector.get_motif_summary()
            motif_nodes = detector.get_motif_nodes()
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.markdown("#### Detected Motifs")
                fig = create_motif_visualization(summary)
                st.plotly_chart(fig, use_container_width=True)
                
                # Motif explanations
                for motif_type, info in MotifDetector.MOTIF_RISK_INFO.items():
                    count = summary.get(f'{motif_type}s_count', 0) if motif_type != 'bidirectional' else summary.get('bidirectional_count', 0)
                    
                    with st.expander(f"{info['description']} ({count} found)", expanded=False):
                        st.markdown(f"""
                            <div style="padding: 10px;">
                                <p><strong>Risk Level:</strong> 
                                    <span style="color: {info['color']}; font-weight: bold;">
                                        {info['risk_level']}
                                    </span>
                                </p>
                                <p style="color: #94a3b8;">{info['explanation']}</p>
                            </div>
                        """, unsafe_allow_html=True)
            
            with col2:
                st.markdown("#### Network with Highlighted Motifs")
                
                fig = create_network_visualization(
                    st.session_state.graph,
                    st.session_state.node_features,
                    highlight_nodes=motif_nodes,
                    title="Motif Nodes Highlighted (Larger Size)"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                st.info(f"""
                    📌 **{len(motif_nodes)} nodes** are part of structural motifs 
                    ({len(motif_nodes)/len(st.session_state.node_features):.1%} of network)
                """)
    
    # Tab 3: GNN Prediction
    with tab3:
        st.markdown("### 🤖 Graph Neural Network Prediction")
        
        if st.session_state.graph is None:
            st.warning("⚠️ Please generate a network first!")
        else:
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.markdown("#### Model Training")
                
                if st.button("🚀 Train GNN Model", key="train_model"):
                    # Prepare data
                    x, edge_index, y, train_mask, test_mask = prepare_gnn_data(
                        st.session_state.graph,
                        st.session_state.node_features
                    )
                    
                    # Create model
                    model = FinancialGNN(
                        input_dim=6,
                        hidden_dim=hidden_dim,
                        num_layers=num_layers
                    )
                    
                    # Training progress
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Train with progress updates
                    history = {'loss': [], 'accuracy': []}
                    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
                    criterion = torch.nn.CrossEntropyLoss()
                    
                    model.train()
                    for epoch in range(epochs):
                        optimizer.zero_grad()
                        logits, _ = model(x, edge_index)
                        loss = criterion(logits[train_mask], y[train_mask])
                        loss.backward()
                        optimizer.step()
                        
                        with torch.no_grad():
                            pred = logits.argmax(dim=1)
                            acc = (pred[train_mask] == y[train_mask]).float().mean()
                        
                        history['loss'].append(loss.item())
                        history['accuracy'].append(acc.item())
                        
                        progress = (epoch + 1) / epochs
                        progress_bar.progress(progress)
                        status_text.text(f"Epoch {epoch+1}/{epochs} - Loss: {loss.item():.4f} - Accuracy: {acc.item():.2%}")
                    
                    # Get predictions
                    predictions, probs, attention = predict_defaults(model, x, edge_index)
                    
                    # Calculate test accuracy
                    test_acc = (predictions[test_mask.numpy()] == y.numpy()[test_mask.numpy()]).mean()
                    
                    st.session_state.model = model
                    st.session_state.predictions = probs
                    st.session_state.history = history
                    
                    progress_bar.empty()
                    status_text.empty()
                    
                    st.success(f"✅ Model trained! Test Accuracy: {test_acc:.2%}")
                
                if st.session_state.history is not None:
                    fig = create_training_chart(st.session_state.history)
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown("#### Prediction Results")
                
                if st.session_state.predictions is not None:
                    # Prediction distribution
                    fig = create_prediction_chart(
                        st.session_state.predictions,
                        st.session_state.node_features['is_default'].values
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Feature importance
                    st.markdown("#### Feature Importance")
                    fig = create_feature_importance_chart()
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("👆 Train the model to see prediction results")
            
            # Network with predictions
            if st.session_state.predictions is not None:
                st.markdown("---")
                st.markdown("#### Network Visualization with Predicted Risk")
                
                fig = create_network_visualization(
                    st.session_state.graph,
                    st.session_state.node_features,
                    predictions=st.session_state.predictions,
                    title="Predicted Default Risk (Green=Low, Red=High)"
                )
                st.plotly_chart(fig, use_container_width=True)
    
    # Tab 4: Learn More
    with tab4:
        st.markdown("### 📚 Understanding GNN for Financial Default Prediction")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("""
                #### What is a Graph Neural Network?
                
                Graph Neural Networks (GNNs) are a class of neural networks designed to work 
                with graph-structured data. Unlike traditional neural networks that assume 
                independent data points, GNNs can capture:
                
                - **Structural relationships** between entities
                - **Information propagation** through connections
                - **Complex patterns** (motifs) in network topology
                
                ---
                
                #### Why Use GNNs for Default Prediction?
                
                Traditional credit scoring considers companies in isolation. But in reality:
                
                🔗 **Companies are connected** through loans, guarantees, and supply chains
                
                📊 **Risk spreads** through the network when one company defaults
                
                🔍 **Structural patterns** (like circular transactions) indicate hidden risks
                
                GNNs capture these relationships that traditional models miss!
            """)
        
        with col2:
            st.markdown("""
                #### Key Concepts in This Demo
                
                **1. Motif Patterns**
                
                Motifs are recurring subgraph structures that have special meaning:
                - 🔺 **Triangles**: Circular dependencies, mutual risk
                - ⭐ **Stars**: Concentrated exposure to a central entity
                - 🔗 **Chains**: Sequential risk propagation
                
                **2. Message Passing**
                
                GNNs work by passing messages between connected nodes:
                1. Each node collects information from neighbors
                2. Information is aggregated and transformed
                3. After several layers, each node has 'awareness' of its neighborhood
                
                **3. Motif-Preserving Architecture**
                
                This approach ensures the GNN :
                - Detects important structural patterns
                - Weighs motif participation in predictions
                - Captures both local features and network effects
            """)
        
        st.markdown("---")
        
        st.markdown("""
            #### 🎓 Key Takeaways
            
            <div style="display: flex; gap: 20px; flex-wrap: wrap;">
                <div style="flex: 1; min-width: 250px; background: rgba(99, 102, 241, 0.1); 
                            border-radius: 12px; padding: 16px; border-left: 4px solid #6366f1;">
                    <strong>Network Structure Matters</strong><br>
                    <span style="color: #94a3b8;">Companies connected to risky entities become risky themselves</span>
                </div>
                <div style="flex: 1; min-width: 250px; background: rgba(249, 115, 22, 0.1); 
                            border-radius: 12px; padding: 16px; border-left: 4px solid #f97316;">
                    <strong>Motifs Reveal Hidden Risk</strong><br>
                    <span style="color: #94a3b8;">Triangular and star patterns often indicate problematic structures</span>
                </div>
                <div style="flex: 1; min-width: 250px; background: rgba(34, 197, 94, 0.1); 
                            border-radius: 12px; padding: 16px; border-left: 4px solid #22c55e;">
                    <strong>GNNs Combine Everything</strong><br>
                    <span style="color: #94a3b8;">Node features + network structure = better predictions</span>
                </div>
            </div>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()

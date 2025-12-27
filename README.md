# Financial Default Prediction via Motif-Preserving GNN

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-2.0+-000000?style=for-the-badge&logo=flask&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

**An interactive demo application showcasing Graph Neural Networks for financial default prediction**

[Features](#-features) • [Installation](#-installation) • [Usage](#-usage) • [Architecture](#-architecture) • [API Reference](#-api-reference)

</div>

---

## 🎯 Overview

This project demonstrates how **Graph Neural Networks (GNNs)** can be used to predict financial defaults by analyzing the network structure of financial relationships. Unlike traditional credit scoring that treats companies in isolation, GNNs capture:

- **Network Effects**: How risk spreads through financial connections
- **Motif Patterns**: Structural patterns (triangles, stars, chains) that indicate hidden risks
- **Relationship Dynamics**: Loans, guarantees, and supply chain dependencies

## ✨ Features

| Feature | Description |
|---------|-------------|
| 📊 **Network Generation** | Create synthetic financial networks with configurable parameters |
| 🔍 **Motif Detection** | Identify risk-indicating patterns (triangles, stars, chains, bidirectional) |
| 🤖 **GNN Training** | Train a Graph Convolutional Network with real-time progress |
| 📈 **Visualization** | Interactive Plotly graphs with color-coded risk levels |
| 🎨 **Modern UI** | Beautiful dark-themed interface with glassmorphism effects |

## 🖼️ Screenshots

### Network Visualization
The application displays financial networks with color-coded nodes:
- 🟢 **Green**: Healthy companies (low default risk)
- 🟡 **Yellow**: Medium risk companies
- 🟠 **Orange**: Elevated risk companies  
- 🔴 **Red**: High risk / defaulted companies

### Motif Analysis
Detect and visualize structural patterns:
- **🔺 Triangles**: Circular transaction patterns (high risk)
- **⭐ Stars**: Concentrated dependencies (medium-high risk)
- **🔗 Chains**: Sequential risk propagation (medium risk)
- **↔️ Bidirectional**: Mutual dependencies (medium risk)

## 📦 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

```bash
# Clone or navigate to the project
cd d:\GNN

# Install dependencies
pip install -r requirements.txt

# Or install manually
pip install flask torch networkx plotly pandas numpy scikit-learn
```

## 🚀 Usage

### Starting the Application

```bash
# Run the Flask server
python flask_app.py

# The app will be available at:
# http://localhost:5000
```

### Application Workflow

1. **Generate Network**: Configure the number of companies and generate a synthetic financial network
2. **Analyze Motifs**: Switch to the Motif Analysis tab to detect structural patterns
3. **Train GNN**: Configure model parameters and train the neural network
4. **View Predictions**: See color-coded default risk predictions on the network

## 🏗️ Architecture

### Project Structure

```
d:\GNN\
├── flask_app.py              # Flask REST API backend
├── templates/
│   └── index.html            # Beautiful dark-themed frontend
├── models/
│   ├── __init__.py
│   └── gnn_model.py          # GNN architecture (GCN layers)
├── utils/
│   ├── __init__.py
│   ├── data_generator.py     # Synthetic network generation
│   ├── graph_utils.py        # Graph processing utilities
│   └── motif_detector.py     # Structural pattern detection
├── components/
│   ├── __init__.py
│   └── visualizations.py     # Plotly chart generation
├── static/
│   └── style.css             # Additional CSS styles
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

### GNN Model Architecture

```
Input Features (6) → GCN Layer → ReLU → Dropout
                          ↓
                    GCN Layer (with residual) → ReLU → Dropout
                          ↓
                    GCN Layer (with residual) → ReLU → Dropout
                          ↓
                    Linear → Output (2 classes)
```

**Node Features:**
| Feature | Description | Range |
|---------|-------------|-------|
| Revenue | Company size in millions | 1 - 5000 |
| Debt Ratio | Leverage indicator | 0.1 - 0.9 |
| Profit Margin | Profitability | -0.2 - 0.5 |
| Years in Business | Company age | 1 - 50 |
| Credit Score | Credit rating | 300 - 850 |
| Default Risk | Base risk probability | 0.05 - 0.35 |

## 📡 API Reference

### Endpoints

#### `POST /api/generate-network`
Generate a new financial network.

**Request Body:**
```json
{
  "n_companies": 50,
  "seed": 42
}
```

**Response:**
```json
{
  "success": true,
  "nodes": [...],
  "edges": [...],
  "stats": {
    "num_nodes": 50,
    "num_edges": 381,
    "default_rate": 0.24,
    "density": 0.156,
    "avg_clustering": 0.234
  }
}
```

#### `GET /api/detect-motifs`
Detect motif patterns in the current network.

**Response:**
```json
{
  "success": true,
  "summary": {
    "triangles_count": 5,
    "stars_count": 3,
    "chains_count": 2,
    "bidirectional_count": 8
  },
  "motif_nodes": [1, 5, 12, ...],
  "motifs": {...}
}
```

#### `POST /api/train-model`
Train the GNN model.

**Request Body:**
```json
{
  "hidden_dim": 32,
  "num_layers": 3,
  "epochs": 100
}
```

**Response:**
```json
{
  "success": true,
  "history": {
    "loss": [...],
    "accuracy": [...]
  },
  "test_accuracy": 0.85,
  "predictions": [...]
}
```

## 🧠 Technical Details

### Graph Convolution Layer

The model uses a normalized graph convolution:

```
H^(l+1) = σ(D^(-1/2) Â D^(-1/2) H^(l) W^(l))
```

Where:
- `Â = A + I` (adjacency with self-loops)
- `D` is the degree matrix
- `σ` is ReLU activation
- `W` is the learnable weight matrix

### Motif Detection Algorithms

- **Triangles**: Find 3-node cycles using adjacency traversal
- **Stars**: Identify nodes with high in/out degree
- **Chains**: Build sequential paths with degree constraints
- **Bidirectional**: Find mutual edges in directed graph

## 📚 Learn More

### What is a Graph Neural Network?

GNNs are neural networks that operate on graph-structured data. They work through **message passing**, where each node aggregates information from its neighbors to update its representation.

### Why Use GNNs for Default Prediction?

1. **Network Effects**: A company's risk is influenced by its financial partners
2. **Contagion**: Defaults can cascade through the network
3. **Hidden Patterns**: Motifs reveal structural risks not visible in individual features

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- This demo is inspired by research on motif-preserving GNNs for financial networks
- Built with Flask, PyTorch, NetworkX, and Plotly

---

<div align="center">

**Made with ❤️ for GNN Research**

</div>

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from sklearn.cluster import KMeans
import numpy as np
from typing import Optional

class RationalActivation(nn.Module):
    """
    Rational activation function as described in the paper.
    R(x) = P(x) / Q(x) = (sum(a_k * x^k)) / (1 + |sum(b_k * x^k)|)
    """
    def __init__(self, numerator_degree: int = 5, denominator_degree: int = 4):
        super().__init__()
        self.numerator_degree = numerator_degree
        self.denominator_degree = denominator_degree
        
        # Initialize coefficients for numerator P(x)
        self.a = nn.Parameter(torch.randn(numerator_degree + 1) * 0.1)
        # Initialize coefficients for denominator Q(x) (excluding the constant 1)
        self.b = nn.Parameter(torch.randn(denominator_degree) * 0.1)
        
    def forward(self, x):
        # Compute numerator P(x) = sum(a_k * x^k)
        numerator = torch.zeros_like(x)
        for k in range(self.numerator_degree + 1):
            numerator += self.a[k] * (x ** k)
        
        # Compute denominator Q(x) = 1 + |sum(b_k * x^k)|
        denominator_sum = torch.zeros_like(x)
        for k in range(1, self.denominator_degree + 1):
            denominator_sum += self.b[k-1] * (x ** k)
        
        denominator = 1 + torch.abs(denominator_sum)
        
        return numerator / denominator

class CNAModule(nn.Module):
    """
    Cluster-Normalize-Activate Module for Graph Neural Networks
    """
    def __init__(self, 
                 num_features: int,
                 num_clusters: int = 8,
                 eps: float = 1e-5,
                 numerator_degree: int = 5,
                 denominator_degree: int = 4):
        super().__init__()
        
        self.num_features = num_features
        self.num_clusters = num_clusters
        self.eps = eps
        
        # Create separate rational activation functions for each cluster
        self.activations = nn.ModuleList([
            RationalActivation(numerator_degree, denominator_degree) 
            for _ in range(num_clusters)
        ])
        
        # Initialize k-means for clustering (will be updated during forward pass)
        self.kmeans = None
        
    def cluster_nodes(self, x):
        """
        Step 1: Cluster node features using k-means
        """
        # Convert to numpy for sklearn
        x_np = x.detach().cpu().numpy()
        
        # Initialize or update k-means
        if self.kmeans is None:
            self.kmeans = KMeans(n_clusters=self.num_clusters, random_state=42, n_init=10)
        
        # Perform clustering
        cluster_labels = self.kmeans.fit_predict(x_np)
        
        return torch.tensor(cluster_labels, dtype=torch.long, device=x.device)
    
    def normalize_clusters(self, x, cluster_labels):
        """
        Step 2: Normalize features within each cluster separately
        """
        normalized_x = torch.zeros_like(x)
        
        for cluster_id in range(self.num_clusters):
            # Get mask for current cluster
            cluster_mask = (cluster_labels == cluster_id)
            
            if cluster_mask.sum() == 0:
                continue
                
            # Get features for nodes in this cluster
            cluster_features = x[cluster_mask]
            
            # Compute mean and variance for this cluster
            cluster_mean = cluster_features.mean(dim=0, keepdim=True)
            cluster_var = cluster_features.var(dim=0, keepdim=True, unbiased=False)
            
            # Normalize: (x - μ) / sqrt(σ² + ε)
            normalized_cluster = (cluster_features - cluster_mean) / torch.sqrt(cluster_var + self.eps)
            
            # Store normalized features
            normalized_x[cluster_mask] = normalized_cluster
            
        return normalized_x
    
    def activate_clusters(self, x, cluster_labels):
        """
        Step 3: Apply separate learned activation functions to each cluster
        """
        activated_x = torch.zeros_like(x)
        
        for cluster_id in range(self.num_clusters):
            # Get mask for current cluster
            cluster_mask = (cluster_labels == cluster_id)
            
            if cluster_mask.sum() == 0:
                continue
                
            # Apply cluster-specific activation function
            cluster_features = x[cluster_mask]
            activated_cluster = self.activations[cluster_id](cluster_features)
            
            # Store activated features
            activated_x[cluster_mask] = activated_cluster
            
        return activated_x
    
    def forward(self, x):
        """
        Forward pass: Cluster → Normalize → Activate
        """
        # Step 1: Cluster
        cluster_labels = self.cluster_nodes(x)
        
        # Step 2: Normalize
        normalized_x = self.normalize_clusters(x, cluster_labels)
        
        # Step 3: Activate
        activated_x = self.activate_clusters(normalized_x, cluster_labels)
        
        return activated_x

class GCNWithCNA(nn.Module):
    """
    Example GCN layer with CNA module replacing standard activation
    """
    def __init__(self, 
                 in_features: int, 
                 out_features: int, 
                 num_clusters: int = 8,
                 bias: bool = True):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        
        # Linear transformation
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        
        # CNA module instead of standard activation
        self.cna = CNAModule(out_features, num_clusters)
        
    def forward(self, x, edge_index):
        """
        Forward pass with message passing and CNA activation
        """
        # Linear transformation
        x = self.linear(x)
        
        # Simple message passing (mean aggregation)
        row, col = edge_index
        deg = torch.bincount(row, minlength=x.size(0)).float()
        deg_inv = deg.pow(-1)
        deg_inv[deg_inv == float('inf')] = 0
        
        # Aggregate messages
        out = torch.zeros_like(x)
        out.index_add_(0, row, x[col])
        out = deg_inv.view(-1, 1) * out
        
        # Apply CNA instead of standard activation (like ReLU)
        out = self.cna(out)
        
        return out

class MultiLayerGCNWithCNA(nn.Module):
    """
    Multi-layer GCN with CNA modules
    """
    def __init__(self, 
                 input_dim: int,
                 hidden_dim: int,
                 output_dim: int,
                 num_layers: int = 2,
                 num_clusters: int = 8,
                 dropout: float = 0.5):
        super().__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Build layers
        self.layers = nn.ModuleList()
        
        # Input layer
        self.layers.append(GCNWithCNA(input_dim, hidden_dim, num_clusters))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.layers.append(GCNWithCNA(hidden_dim, hidden_dim, num_clusters))
        
        # Output layer (without CNA for final prediction)
        self.layers.append(nn.Linear(hidden_dim, output_dim))
        
    def forward(self, x, edge_index):
        # Apply CNA layers
        for i in range(self.num_layers - 1):
            x = self.layers[i](x, edge_index)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Final linear layer
        x = self.layers[-1](x)
        
        return x

# Example usage and testing
if __name__ == "__main__":
    # Test the CNA module
    print("Testing CNA Module...")
    
    # Create sample data
    num_nodes = 100
    num_features = 64
    num_clusters = 8
    
    x = torch.randn(num_nodes, num_features)
    
    # Test CNA module
    cna = CNAModule(num_features, num_clusters)
    output = cna(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Number of clusters: {num_clusters}")
    
    # Test with a simple graph
    from torch_geometric.data import Data
    
    # Create a simple graph
    edge_index = torch.tensor([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]], dtype=torch.long)
    data = Data(x=x[:4], edge_index=edge_index)
    
    # Test GCN with CNA
    model = MultiLayerGCNWithCNA(
        input_dim=num_features,
        hidden_dim=32,
        output_dim=7,  # 7 classes like Cora dataset
        num_layers=4,
        num_clusters=8
    )
    
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
    
    print(f"\nGCN with CNA output shape: {out.shape}")
    print("CNA module implementation complete!")
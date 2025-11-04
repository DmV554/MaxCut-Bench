"""
GNN Policy para CLR: Aprende a samplear hiperplanos inteligentemente.

La política parametriza una distribución sobre S^(d-1) condicional al grafo
y a los vectores SDP. Para simplificar, usamos:
    - GCN de 3 capas para procesar el grafo
    - MLP para proyectar a parámetros de distribución
    - Von Mises-Fisher como distribución output (por ahora)

Alternativa más simple para MVP: output directamente un hiperplano (determinista)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, global_add_pool
from torch_geometric.data import Data, Batch
import numpy as np
from typing import Dict, Optional, Tuple


class GNNPolicy(nn.Module):
    """
    Red neuronal que aprende política de redondeo para CLR.
    
    Architecture:
        Input: Grafo + vectores SDP como node features
        GCN: 3 capas, 64 hidden dims
        Pooling: Global mean
        MLP: 2 capas para output
        Output: Hiperplano en S^(d-1) (normalizado)
    
    Para simplificar MVP, output es determinista (no distribución).
    Versión futura: parametrizar von Mises-Fisher para muestreo.
    """
    
    def __init__(self,
                 node_feature_dim: int = 1,  # Dim de features originales
                 sdp_vector_dim: int = 20,   # Dim de vectores SDP (≈ n para grafos chicos)
                 hidden_dim: int = 64,
                 num_layers: int = 3,
                 dropout: float = 0.1,
                 pooling: str = 'mean'):  # 'mean' o 'add'
        """
        Args:
            node_feature_dim: Dimensión de features originales de nodos
            sdp_vector_dim: Dimensión de vectores SDP (input principal)
            hidden_dim: Dimensión de embeddings ocultos
            num_layers: Número de capas GCN
            dropout: Dropout rate
            pooling: Tipo de pooling global
        """
        super(GNNPolicy, self).__init__()
        
        self.node_feature_dim = node_feature_dim
        self.sdp_vector_dim = sdp_vector_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.pooling = pooling
        
        # Input projection: (node_features + sdp_vectors) → hidden_dim
        input_dim = node_feature_dim + sdp_vector_dim
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # GCN layers
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        
        self.dropout = nn.Dropout(dropout)
        
        # Global pooling
        if pooling == 'mean':
            self.pool = global_mean_pool
        elif pooling == 'add':
            self.pool = global_add_pool
        else:
            raise ValueError(f"Pooling desconocido: {pooling}")
        
        # Output MLP: graph embedding → hyperplane parameters
        # Output dim = sdp_vector_dim (el hiperplano debe vivir en R^d)
        self.output_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, sdp_vector_dim)
        )
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
    
    def forward(self, data: Data) -> torch.Tensor:
        """
        Forward pass: grafo → hiperplano.
        
        Args:
            data: PyG Data object con:
                - x: node features [n, node_feature_dim + sdp_vector_dim]
                - edge_index: [2, num_edges]
                - batch: batch vector (para batching)
        
        Returns:
            hyperplane: [batch_size, sdp_vector_dim] (sin normalizar aún)
        """
        x, edge_index = data.x, data.edge_index
        batch = data.batch if hasattr(data, 'batch') else torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        
        # Input projection
        x = self.input_proj(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        # GCN layers con residual connections
        for i, conv in enumerate(self.convs):
            x_new = conv(x, edge_index)
            x_new = F.relu(x_new)
            x_new = self.dropout(x_new)
            
            # Residual connection (excepto primera capa)
            if i > 0:
                x = x + x_new
            else:
                x = x_new
        
        # Global pooling: [n, hidden_dim] → [batch_size, hidden_dim]
        graph_embedding = self.pool(x, batch)
        
        # Output MLP: graph_embedding → hyperplane
        hyperplane = self.output_mlp(graph_embedding)
        
        # NO normalizamos aquí, lo haremos al samplear
        # Esto permite gradientes más estables
        return hyperplane
    
    def sample(self, graph_data: Dict, sdp_vectors: np.ndarray) -> np.ndarray:
        """
        Samplea un hiperplano usando la política aprendida.
        
        Esta es la interfaz que usa MixedSampler.
        
        Args:
            graph_data: Dict con 'adjacency_matrix' y opcionalmente 'features'
            sdp_vectors: np.ndarray (n, d) vectores SDP
        
        Returns:
            hyperplane: np.ndarray (d,) normalizado en S^(d-1)
        """
        self.eval()
        with torch.no_grad():
            # Preparar PyG Data object
            pyg_data = self._prepare_data(graph_data, sdp_vectors)
            
            # Forward pass
            hyperplane_unnormalized = self.forward(pyg_data)  # [1, d]
            
            # Normalizar a esfera
            hyperplane = hyperplane_unnormalized.squeeze(0).cpu().numpy()
            norm = np.linalg.norm(hyperplane)
            
            if norm < 1e-8:
                # Caso degenerado: retornar dirección aleatoria
                hyperplane = np.random.randn(len(hyperplane))
                norm = np.linalg.norm(hyperplane)
            
            hyperplane = hyperplane / norm
            
        return hyperplane
    
    def _prepare_data(self, graph_data: Dict, sdp_vectors: np.ndarray) -> Data:
        """
        Convierte grafo + vectores SDP a PyG Data object.
        
        Args:
            graph_data: Dict con 'adjacency_matrix'
            sdp_vectors: np.ndarray (n, d)
        
        Returns:
            PyG Data object listo para forward pass
        """
        adjacency_matrix = graph_data['adjacency_matrix']
        n = adjacency_matrix.shape[0]
        
        # Edge index de matriz de adyacencia
        edge_index = []
        for i in range(n):
            for j in range(n):
                if adjacency_matrix[i, j] != 0:
                    edge_index.append([i, j])
        
        if len(edge_index) == 0:
            # Grafo sin aristas: crear self-loops
            edge_index = [[i, i] for i in range(n)]
        
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        
        # Node features: [constant_feature, sdp_vectors]
        # Por ahora, solo usamos los vectores SDP
        node_features = torch.tensor(sdp_vectors, dtype=torch.float32)
        
        # Si hay features adicionales en graph_data, concatenarlas
        if 'features' in graph_data:
            extra_features = torch.tensor(graph_data['features'], dtype=torch.float32)
            node_features = torch.cat([extra_features, node_features], dim=1)
        else:
            # Agregar feature constante (grado o 1s)
            degrees = adjacency_matrix.sum(axis=1, keepdims=True)
            degree_features = torch.tensor(degrees, dtype=torch.float32)
            node_features = torch.cat([degree_features, node_features], dim=1)
        
        data = Data(x=node_features, edge_index=edge_index)
        data = data.to(self.device)
        
        return data
    
    def compute_hyperplane_batch(self, batch_data: Batch) -> torch.Tensor:
        """
        Computa hiperplanos para un batch de grafos.
        
        Args:
            batch_data: PyG Batch object
        
        Returns:
            hyperplanes: [batch_size, d] sin normalizar
        """
        return self.forward(batch_data)


def create_gnn_policy(sdp_vector_dim: int = 20,
                      hidden_dim: int = 64,
                      num_layers: int = 3,
                      dropout: float = 0.1,
                      pretrained_path: Optional[str] = None) -> GNNPolicy:
    """
    Factory function para crear GNN policy.
    
    Args:
        sdp_vector_dim: Dimensión de vectores SDP
        hidden_dim: Dimensión de embeddings
        num_layers: Número de capas GCN
        dropout: Dropout rate
        pretrained_path: Path a modelo preentrenado (.pth)
    
    Returns:
        GNNPolicy (cargado si pretrained_path es provisto)
    """
    policy = GNNPolicy(
        node_feature_dim=1,  # Solo grado por ahora
        sdp_vector_dim=sdp_vector_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout
    )
    
    if pretrained_path is not None:
        checkpoint = torch.load(pretrained_path, map_location=policy.device)
        policy.load_state_dict(checkpoint['model_state_dict'])
        print(f"Modelo cargado desde {pretrained_path}")
    
    return policy


if __name__ == '__main__':
    # Test básico
    print("Testing GNNPolicy...")
    
    # Crear política
    policy = create_gnn_policy(sdp_vector_dim=20, hidden_dim=32, num_layers=2)
    print(f"✅ Política creada: {sum(p.numel() for p in policy.parameters())} parámetros")
    
    # Test con grafo dummy
    n = 20
    d = 20
    adjacency = np.eye(n) + np.random.rand(n, n) * 0.5
    adjacency = (adjacency + adjacency.T) / 2  # Simetrizar
    np.fill_diagonal(adjacency, 0)
    
    sdp_vectors = np.random.randn(n, d)
    sdp_vectors /= np.linalg.norm(sdp_vectors, axis=1, keepdims=True)
    
    graph_data = {'adjacency_matrix': adjacency}
    
    # Samplear hiperplano
    hyperplane = policy.sample(graph_data, sdp_vectors)
    print(f"✅ Hiperplano generado: shape={hyperplane.shape}, norm={np.linalg.norm(hyperplane):.6f}")
    
    assert hyperplane.shape == (d,), f"Shape incorrecto: {hyperplane.shape}"
    assert abs(np.linalg.norm(hyperplane) - 1.0) < 1e-5, "Hiperplano no normalizado"
    
    print("✅ Tests pasados")

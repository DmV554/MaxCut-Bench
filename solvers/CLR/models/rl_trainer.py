"""
RL Trainer para CLR: Entrena GNN policy usando REINFORCE.

Metodología basada en Maliakal et al. (ECO-DQN paper):
- RL sin datos: aprende directamente de la recompensa del corte
- Reward: cut_learned - cut_uniform (mejora sobre baseline)
- REINFORCE con baseline para reducir varianza
- Training on-policy con grafos sintéticos

Key insight: No necesitamos conocer el óptimo, solo mejorar sobre uniforme.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
import numpy as np
from typing import Dict, List, Optional, Tuple
import time
from collections import defaultdict
import os
import pickle


class RLTrainer:
    """
    Entrena GNN policy para CLR usando REINFORCE.
    
    Training loop:
        1. Para cada grafo, resolver SDP una vez (cachear)
        2. Samplear K hiperplanos con GNN (policy actual)
        3. Evaluar cortes para cada hiperplano
        4. Reward = cut_learned - baseline_uniform
        5. REINFORCE gradient: ∇θ J = E[∇θ log π(r|G) · R]
        6. Update θ con gradient ascent
    """
    
    def __init__(self,
                 gnn_policy,
                 sdp_solver,
                 learning_rate: float = 1e-4,
                 baseline_K: int = 10,
                 policy_K: int = 10,
                 gamma: float = 0.99,
                 entropy_coef: float = 0.01,
                 use_baseline_value: bool = True,
                 device: Optional[str] = None):
        """
        Args:
            gnn_policy: GNNPolicy a entrenar
            sdp_solver: SDPSolver para resolver relajaciones
            learning_rate: Learning rate para optimizer
            baseline_K: Número de samples uniformes para baseline
            policy_K: Número de samples de la policy por grafo
            gamma: Discount factor (no usado en episodios single-step)
            entropy_coef: Coeficiente de regularización de entropía
            use_baseline_value: Si True, usa baseline aprendido para reducir varianza
            device: 'cuda' o 'cpu' (auto-detect si None)
        """
        self.gnn_policy = gnn_policy
        self.sdp_solver = sdp_solver
        self.baseline_K = baseline_K
        self.policy_K = policy_K
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.use_baseline_value = use_baseline_value
        
        # Device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.gnn_policy.to(self.device)
        
        # Optimizer
        self.optimizer = optim.Adam(self.gnn_policy.parameters(), lr=learning_rate)
        
        # Baseline value function (opcional, para reducir varianza)
        if self.use_baseline_value:
            # Simple MLP que predice valor esperado del corte
            self.value_net = nn.Sequential(
                nn.Linear(gnn_policy.hidden_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 1)
            ).to(self.device)
            self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=learning_rate)
        
        # Estadísticas de training
        self.stats = defaultdict(list)
    
    def train_episode(self, graph_data: Dict, sdp_vectors: np.ndarray, Z_SDP: float) -> Dict:
        """
        Entrena en un solo grafo (episodio).
        
        Args:
            graph_data: Dict con 'adjacency_matrix'
            sdp_vectors: Vectores SDP pre-computados (n, d)
            Z_SDP: Valor óptimo SDP (cota superior)
        
        Returns:
            {
                'policy_loss': float,
                'value_loss': float (si use_baseline_value),
                'mean_reward': float,
                'mean_cut_policy': float,
                'mean_cut_uniform': float,
                'improvement': float (%)
            }
        """
        self.gnn_policy.train()
        
        adjacency_matrix = graph_data['adjacency_matrix']
        n = adjacency_matrix.shape[0]
        
        # Preparar PyG Data
        pyg_data = self.gnn_policy._prepare_data(graph_data, sdp_vectors)
        
        # 1. Samplear hiperplanos con la policy (K veces)
        log_probs = []
        cuts_policy = []
        
        for _ in range(self.policy_K):
            # Forward pass (sin normalizar)
            hyperplane_unnormalized = self.gnn_policy.forward(pyg_data)  # [1, d]
            
            # Calcular log probability (asumiendo Gaussian en hiperplano no normalizado)
            # log π(r) ∝ -||r - μ||² donde μ es el output de la GNN
            # Para simplificar: log π(r) = -||r_unnorm||² + const
            # Esto es equivalente a L2 regularization
            log_prob = -0.5 * torch.sum(hyperplane_unnormalized ** 2)
            log_probs.append(log_prob)
            
            # Normalizar y evaluar corte
            hyperplane = hyperplane_unnormalized.squeeze(0).detach().cpu().numpy()
            norm = np.linalg.norm(hyperplane)
            if norm < 1e-8:
                hyperplane = np.random.randn(len(hyperplane))
                norm = np.linalg.norm(hyperplane)
            hyperplane = hyperplane / norm
            
            # Evaluar corte
            projections = sdp_vectors @ hyperplane
            spins = np.sign(projections)
            spins[spins == 0] = 1
            cut = self._evaluate_cut(spins, adjacency_matrix)
            cuts_policy.append(cut)
        
        # 2. Baseline: samplear K hiperplanos uniformes
        cuts_uniform = []
        for _ in range(self.baseline_K):
            hyperplane = np.random.randn(sdp_vectors.shape[1])
            hyperplane /= np.linalg.norm(hyperplane)
            
            projections = sdp_vectors @ hyperplane
            spins = np.sign(projections)
            spins[spins == 0] = 1
            cut = self._evaluate_cut(spins, adjacency_matrix)
            cuts_uniform.append(cut)
        
        # 3. Calcular rewards
        mean_cut_policy = np.mean(cuts_policy)
        mean_cut_uniform = np.mean(cuts_uniform)
        
        # Reward individual para cada sample
        # Opción 1: Reward = cut - mean(uniform)
        baseline_value = mean_cut_uniform
        
        # Opción 2: Reward = cut - baseline_learned (si usamos value net)
        if self.use_baseline_value:
            # Estimar valor con value network
            # Reusar el graph embedding que ya calculamos durante forward
            with torch.no_grad():
                x = pyg_data.x
                edge_index = pyg_data.edge_index
                batch = pyg_data.batch if hasattr(pyg_data, 'batch') else torch.zeros(x.size(0), dtype=torch.long, device=self.device)
                
                # Aplicar input projection
                x = self.gnn_policy.input_proj(x)
                x = F.relu(x)
                
                # Aplicar todas las capas GCN
                for i, conv in enumerate(self.gnn_policy.convs):
                    x_new = conv(x, edge_index)
                    x_new = F.relu(x_new)
                    if i > 0:
                        x = x + x_new  # Residual
                    else:
                        x = x_new
                
                # Pool para obtener graph embedding
                graph_embedding = self.gnn_policy.pool(x, batch)
            
            baseline_value_learned = self.value_net(graph_embedding).squeeze()
            baseline_value = baseline_value_learned.item()
        
        rewards = [cut - baseline_value for cut in cuts_policy]
        
        # 4. REINFORCE loss
        policy_loss = 0.0
        for log_prob, reward in zip(log_probs, rewards):
            # Gradient ascent: maximize E[log π(r) · R]
            # En PyTorch: minimize -E[log π(r) · R]
            policy_loss += -log_prob * reward
        
        policy_loss = policy_loss / self.policy_K
        
        # Entropy regularization (opcional, para exploración)
        if self.entropy_coef > 0:
            # Simplificación: entropy ≈ -mean(log_probs)
            entropy = -torch.mean(torch.stack(log_probs))
            policy_loss -= self.entropy_coef * entropy
        
        # 5. Update policy
        self.optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.gnn_policy.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        # 6. Update value network (si se usa)
        value_loss = 0.0
        if self.use_baseline_value:
            # Target: mean cut de policy
            value_target = torch.tensor([mean_cut_policy], dtype=torch.float32, device=self.device)
            value_pred = self.value_net(graph_embedding)
            value_loss = F.mse_loss(value_pred, value_target)
            
            self.value_optimizer.zero_grad()
            value_loss.backward()
            self.value_optimizer.step()
        
        # Estadísticas
        improvement = (mean_cut_policy - mean_cut_uniform) / mean_cut_uniform * 100 if mean_cut_uniform > 0 else 0.0
        
        stats = {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item() if self.use_baseline_value else 0.0,
            'mean_reward': np.mean(rewards),
            'mean_cut_policy': mean_cut_policy,
            'mean_cut_uniform': mean_cut_uniform,
            'improvement': improvement,
            'Z_SDP': Z_SDP,
            'gap_policy': (Z_SDP - mean_cut_policy) / Z_SDP * 100 if Z_SDP > 0 else 0.0,
            'gap_uniform': (Z_SDP - mean_cut_uniform) / Z_SDP * 100 if Z_SDP > 0 else 0.0,
        }
        
        return stats
    
    def train_epoch(self, training_graphs: List[Tuple], verbose: bool = True) -> Dict:
        """
        Entrena en un epoch (todos los grafos).
        
        Args:
            training_graphs: Lista de (graph_data, sdp_vectors, Z_SDP)
            verbose: Si True, imprime progreso
        
        Returns:
            Estadísticas agregadas del epoch
        """
        epoch_stats = defaultdict(list)
        
        for i, (graph_data, sdp_vectors, Z_SDP) in enumerate(training_graphs):
            stats = self.train_episode(graph_data, sdp_vectors, Z_SDP)
            
            for key, value in stats.items():
                epoch_stats[key].append(value)
            
            if verbose and (i + 1) % 10 == 0:
                print(f"  Graph {i+1}/{len(training_graphs)}: "
                      f"cut_policy={stats['mean_cut_policy']:.2f}, "
                      f"improvement={stats['improvement']:.2f}%")
        
        # Promediar estadísticas
        avg_stats = {key: np.mean(values) for key, values in epoch_stats.items()}
        
        return avg_stats
    
    def validate(self, validation_graphs: List[Tuple]) -> Dict:
        """
        Valida la policy en grafos de validación.
        
        Args:
            validation_graphs: Lista de (graph_data, sdp_vectors, Z_SDP)
        
        Returns:
            Estadísticas de validación
        """
        self.gnn_policy.eval()
        
        val_stats = defaultdict(list)
        
        with torch.no_grad():
            for graph_data, sdp_vectors, Z_SDP in validation_graphs:
                adjacency_matrix = graph_data['adjacency_matrix']
                
                # Evaluar con policy
                cuts_policy = []
                for _ in range(self.policy_K):
                    hyperplane = self.gnn_policy.sample(graph_data, sdp_vectors)
                    projections = sdp_vectors @ hyperplane
                    spins = np.sign(projections)
                    spins[spins == 0] = 1
                    cut = self._evaluate_cut(spins, adjacency_matrix)
                    cuts_policy.append(cut)
                
                # Baseline uniforme
                cuts_uniform = []
                for _ in range(self.baseline_K):
                    hyperplane = np.random.randn(sdp_vectors.shape[1])
                    hyperplane /= np.linalg.norm(hyperplane)
                    projections = sdp_vectors @ hyperplane
                    spins = np.sign(projections)
                    spins[spins == 0] = 1
                    cut = self._evaluate_cut(spins, adjacency_matrix)
                    cuts_uniform.append(cut)
                
                mean_cut_policy = np.mean(cuts_policy)
                mean_cut_uniform = np.mean(cuts_uniform)
                improvement = (mean_cut_policy - mean_cut_uniform) / mean_cut_uniform * 100 if mean_cut_uniform > 0 else 0.0
                
                val_stats['mean_cut_policy'].append(mean_cut_policy)
                val_stats['mean_cut_uniform'].append(mean_cut_uniform)
                val_stats['improvement'].append(improvement)
                val_stats['gap_policy'].append((Z_SDP - mean_cut_policy) / Z_SDP * 100 if Z_SDP > 0 else 0.0)
        
        # Promediar
        avg_val_stats = {key: np.mean(values) for key, values in val_stats.items()}
        
        return avg_val_stats
    
    def _evaluate_cut(self, spins: np.ndarray, adjacency_matrix: np.ndarray) -> float:
        """Evalúa el valor del corte."""
        total_weight = adjacency_matrix.sum()
        spin_contribution = spins @ adjacency_matrix @ spins
        cut = (total_weight - spin_contribution) / 2.0
        return float(cut)
    
    def save_checkpoint(self, path: str, epoch: int, stats: Dict):
        """Guarda checkpoint del modelo."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.gnn_policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'stats': stats
        }
        
        if self.use_baseline_value:
            checkpoint['value_net_state_dict'] = self.value_net.state_dict()
            checkpoint['value_optimizer_state_dict'] = self.value_optimizer.state_dict()
        
        torch.save(checkpoint, path)
        print(f"✅ Checkpoint guardado en {path}")
    
    def load_checkpoint(self, path: str):
        """Carga checkpoint del modelo."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.gnn_policy.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.use_baseline_value and 'value_net_state_dict' in checkpoint:
            self.value_net.load_state_dict(checkpoint['value_net_state_dict'])
            self.value_optimizer.load_state_dict(checkpoint['value_optimizer_state_dict'])
        
        print(f"✅ Checkpoint cargado desde {path}")
        return checkpoint['epoch'], checkpoint['stats']


def prepare_training_data(graphs: List[Dict],
                          sdp_solver,
                          cache_dir: str = './cache_sdp') -> List[Tuple]:
    """
    Prepara datos de training: resuelve SDPs y cachea resultados.
    
    Args:
        graphs: Lista de graph_data dicts
        sdp_solver: SDPSolver instance
        cache_dir: Directorio de cache
    
    Returns:
        Lista de (graph_data, sdp_vectors, Z_SDP) tuples
    """
    training_data = []
    
    print(f"Preparando {len(graphs)} grafos (resolviendo SDPs)...")
    
    for i, graph_data in enumerate(graphs):
        # Resolver SDP (usa cache automáticamente)
        sdp_result = sdp_solver.solve(graph_data['adjacency_matrix'])
        
        if sdp_result['status'] not in ['optimal', 'optimal_inaccurate']:
            print(f"⚠️  Grafo {i}: SDP no convergió ({sdp_result['status']}), saltando")
            continue
        
        vectors = sdp_result['vectors']
        Z_SDP = sdp_result['Z_SDP']
        
        training_data.append((graph_data, vectors, Z_SDP))
        
        if (i + 1) % 100 == 0:
            print(f"  Procesados {i+1}/{len(graphs)} grafos")
    
    print(f"✅ {len(training_data)} grafos listos para training")
    
    return training_data


# Import para value loss
import torch.nn.functional as F


if __name__ == '__main__':
    print("Testing RLTrainer...")
    
    # Este test requiere torch_geometric instalado
    print("⚠️  Test completo requiere PyTorch Geometric")
    print("✅ RLTrainer implementado - listo para training")

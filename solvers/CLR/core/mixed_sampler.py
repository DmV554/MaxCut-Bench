"""
Mixed Sampler para CLR: Mezcla convexa entre muestreo uniforme y política aprendida.

Este es el componente CLAVE que diferencia CLR de métodos puros:
    π_λ(r) = λ · Uniform(S^(n-1)) + (1-λ) · π_learned(r | G, {x_i*})

Donde:
- λ ∈ [0,1] controla el trade-off conservatismo/agresividad
- λ = 1.0: Reduce a pGW puro (baseline uniforme)
- λ = 0.0: Confía completamente en la GNN aprendida
- λ ∈ (0,1): Mezcla híbrida con garantías parciales
"""
import numpy as np
from typing import Dict, Optional, Callable
import time


class MixedSampler:
    """
    Implementa el muestreo híbrido π_λ para CLR.
    
    La mezcla convexa permite interpolar suavemente entre:
    - Garantías teóricas (λ alto)
    - Performance empírico (λ bajo)
    """
    
    def __init__(self, 
                 lambda_mix: float = 0.5,
                 learned_policy: Optional[Callable] = None,
                 seed: Optional[int] = None):
        """
        Args:
            lambda_mix: Peso de muestreo uniforme en [0,1]
                       λ=1.0 → solo uniforme (pGW)
                       λ=0.0 → solo aprendida (arriesgado)
            learned_policy: Función que samplea de π_learned
                           Signature: learned_policy(graph, vectors) → hyperplane
                           Si None, usa solo muestreo uniforme
            seed: Semilla para reproducibilidad
        """
        if not 0 <= lambda_mix <= 1:
            raise ValueError(f"lambda_mix debe estar en [0,1], recibido: {lambda_mix}")
        
        self.lambda_mix = lambda_mix
        self.learned_policy = learned_policy
        self.rng = np.random.RandomState(seed)
        
        # Estadísticas de uso
        self.stats = {
            'uniform_samples': 0,
            'learned_samples': 0,
            'total_samples': 0
        }
    
    def sample_hyperplane(self,
                         vectors: np.ndarray,
                         graph_data: Optional[Dict] = None) -> np.ndarray:
        """
        Samplea un hiperplano de la mezcla π_λ.
        
        Args:
            vectors: Vectores SDP (n, d) de la relajación
            graph_data: Información del grafo (opcional, para π_learned)
                       Debe contener: adjacency_matrix, features, etc.
        
        Returns:
            hyperplane: Vector en S^(d-1) para redondeo
        """
        d = vectors.shape[1]
        
        # Decidir de qué distribución samplear
        use_uniform = self.rng.rand() < self.lambda_mix
        
        if use_uniform or self.learned_policy is None:
            # Muestreo uniforme en la esfera
            hyperplane = self._sample_uniform_sphere(d)
            self.stats['uniform_samples'] += 1
        else:
            # Muestreo de la política aprendida
            hyperplane = self.learned_policy(graph_data, vectors)
            self.stats['learned_samples'] += 1
        
        self.stats['total_samples'] += 1
        
        return hyperplane
    
    def sample_multiple(self,
                       vectors: np.ndarray,
                       K: int,
                       graph_data: Optional[Dict] = None,
                       return_stats: bool = False) -> Dict:
        """
        Samplea K hiperplanos y retorna el mejor corte (best-of-K).
        
        Esta es la implementación completa de CLR sampling.
        
        Args:
            vectors: Vectores SDP (n, d)
            K: Número de samples (típicamente 50)
            graph_data: Info del grafo (debe incluir adjacency_matrix)
            return_stats: Si True, retorna estadísticas detalladas
            
        Returns:
            {
                'best_cut': float,
                'best_hyperplane': np.ndarray,
                'best_spins': np.ndarray,
                'all_cuts': list (si return_stats),
                'sampling_strategy': dict (si return_stats)
            }
        """
        if graph_data is None or 'adjacency_matrix' not in graph_data:
            raise ValueError("graph_data debe contener 'adjacency_matrix'")
        
        adjacency_matrix = graph_data['adjacency_matrix']
        n = adjacency_matrix.shape[0]
        
        if vectors.shape[0] != n:
            raise ValueError(f"Dimensiones incompatibles: vectors ({vectors.shape[0]}) "
                           f"vs adjacency ({n})")
        
        start_time = time.time()
        
        best_cut = -np.inf
        best_hyperplane = None
        best_spins = None
        all_cuts = [] if return_stats else None
        
        # Reset stats para este batch
        self.stats = {
            'uniform_samples': 0,
            'learned_samples': 0,
            'total_samples': 0
        }
        
        for _ in range(K):
            # Samplear hiperplano de π_λ
            hyperplane = self.sample_hyperplane(vectors, graph_data)
            
            # Redondeo: s_i = sign(x_i^T · r)
            projections = vectors @ hyperplane
            spins = np.sign(projections)
            spins[spins == 0] = 1  # Resolver empates
            
            # Evaluar corte: Σ w_ij * (1 - s_i*s_j)/2
            cut = self._evaluate_cut(spins, adjacency_matrix)
            
            if return_stats:
                all_cuts.append(cut)
            
            if cut > best_cut:
                best_cut = cut
                best_hyperplane = hyperplane
                best_spins = spins.copy()
        
        sampling_time = time.time() - start_time
        
        result = {
            'best_cut': best_cut,
            'best_hyperplane': best_hyperplane,
            'best_spins': best_spins,
            'sampling_time': sampling_time
        }
        
        if return_stats:
            result['all_cuts'] = all_cuts
            result['mean_cut'] = np.mean(all_cuts)
            result['std_cut'] = np.std(all_cuts)
            result['min_cut'] = np.min(all_cuts)
            result['max_cut'] = np.max(all_cuts)
            result['sampling_strategy'] = self.stats.copy()
        
        return result
    
    def _sample_uniform_sphere(self, d: int) -> np.ndarray:
        """
        Samplea uniformemente en S^(d-1).
        
        Método: Marsaglia (1972)
        - Samplear z ~ N(0, I_d)
        - Normalizar: r = z / ||z||
        """
        z = self.rng.randn(d)
        norm = np.linalg.norm(z)
        
        # Manejo de caso degenerado
        if norm < 1e-10:
            z = self.rng.randn(d)
            norm = np.linalg.norm(z)
        
        return z / norm
    
    def _evaluate_cut(self, spins: np.ndarray, adjacency_matrix: np.ndarray) -> float:
        """
        Evalúa el valor del corte para una asignación de spins.
        
        Cut = Σ_{(i,j)} w_ij * (1 - s_i*s_j)/2
            = (Σ w_ij - s^T W s) / 2
        
        Args:
            spins: Vector {-1, +1}^n
            adjacency_matrix: Matriz W (n, n), simétrica
            
        Returns:
            cut_value: Suma de pesos de aristas cortadas
        """
        n = len(spins)
        
        # Versión eficiente: cut = (sum(W) - s^T W s) / 2
        total_weight = adjacency_matrix.sum()
        spin_contribution = spins @ adjacency_matrix @ spins
        cut = (total_weight - spin_contribution) / 2.0
        
        return float(cut)
    
    def get_mixing_ratio(self) -> Dict:
        """
        Retorna estadísticas de uso de cada estrategia.
        
        Returns:
            {
                'lambda_mix': float,
                'uniform_fraction': float,
                'learned_fraction': float,
                'total_samples': int
            }
        """
        total = self.stats['total_samples']
        
        if total == 0:
            uniform_frac = 0.0
            learned_frac = 0.0
        else:
            uniform_frac = self.stats['uniform_samples'] / total
            learned_frac = self.stats['learned_samples'] / total
        
        return {
            'lambda_mix': self.lambda_mix,
            'uniform_fraction': uniform_frac,
            'learned_fraction': learned_frac,
            'total_samples': total
        }
    
    def set_lambda(self, lambda_mix: float):
        """Actualiza el parámetro de mezcla."""
        if not 0 <= lambda_mix <= 1:
            raise ValueError(f"lambda_mix debe estar en [0,1], recibido: {lambda_mix}")
        self.lambda_mix = lambda_mix
    
    def set_policy(self, learned_policy: Callable):
        """Actualiza la política aprendida."""
        self.learned_policy = learned_policy


def create_clr_sampler(lambda_mix: float = 0.5,
                       gnn_policy=None,
                       seed: Optional[int] = None) -> MixedSampler:
    """
    Factory function para crear un sampler CLR configurado.
    
    Args:
        lambda_mix: Parámetro de mezcla λ
        gnn_policy: Modelo GNN entrenado (objeto con método .sample())
        seed: Semilla para reproducibilidad
        
    Returns:
        MixedSampler configurado
    """
    # Convertir GNN policy a callable si es necesario
    if gnn_policy is not None and hasattr(gnn_policy, 'sample'):
        learned_policy = lambda graph, vectors: gnn_policy.sample(graph, vectors)
    else:
        learned_policy = gnn_policy
    
    return MixedSampler(
        lambda_mix=lambda_mix,
        learned_policy=learned_policy,
        seed=seed
    )

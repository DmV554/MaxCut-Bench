"""
Baseline Solvers para comparación con CLR
"""
import numpy as np
from typing import Dict, Optional
import time
import sys
import os

# Add parent directory to path to import core modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.sdp_solver import SDPSolver, evaluate_hyperplane_cut


def probabilistic_gw(adjacency_matrix: np.ndarray,
                    K: int = 50,
                    sdp_solver: Optional[SDPSolver] = None,
                    return_details: bool = False) -> Dict:
    """
    Probabilistic Goemans-Williamson: best-of-K sampling uniforme.
    
    Este es el BASELINE CRÍTICO que CLR debe superar.
    
    Args:
        adjacency_matrix: Matriz de adyacencia (n, n)
        K: Número de muestras (típicamente 50)
        sdp_solver: Instancia del solver SDP (o se crea uno nuevo)
        return_details: Si True, retorna información detallada
        
    Returns:
        {
            'cut': float,           # Mejor corte encontrado
            'Z_SDP': float,         # Cota superior SDP
            'best_hyperplane': np.ndarray,
            'best_spins': np.ndarray,
            'time_sdp': float,
            'time_sampling': float,
            'time_total': float
        }
    """
    # Resolver SDP (o usar solver provisto)
    if sdp_solver is None:
        sdp_solver = SDPSolver(verbose=False)
    
    sdp_result = sdp_solver.solve(adjacency_matrix)
    
    if sdp_result['status'] not in ['optimal', 'optimal_inaccurate']:
        raise ValueError(f"SDP no convergió: {sdp_result['status']}")
    
    vectors = sdp_result['vectors']
    Z_SDP = sdp_result['Z_SDP']
    time_sdp = sdp_result['solve_time']
    
    # Samplear K hiperplanos uniformes en S^(d-1)
    d = vectors.shape[1]
    
    start_sampling = time.time()
    
    best_cut = -np.inf
    best_hyperplane = None
    best_spins = None
    all_cuts = []
    
    for _ in range(K):
        # Samplear hiperplano uniforme en la esfera
        hyperplane = np.random.randn(d)
        hyperplane /= np.linalg.norm(hyperplane)
        
        # Evaluar corte
        cut = evaluate_hyperplane_cut(vectors, hyperplane, adjacency_matrix)
        all_cuts.append(cut)
        
        if cut > best_cut:
            best_cut = cut
            best_hyperplane = hyperplane
            # Guardar spins para esta solución
            spins = np.sign(vectors @ hyperplane)
            spins[spins == 0] = 1
            best_spins = spins
    
    time_sampling = time.time() - start_sampling
    
    result = {
        'cut': best_cut,
        'Z_SDP': Z_SDP,
        'best_hyperplane': best_hyperplane,
        'best_spins': best_spins,
        'time_sdp': time_sdp,
        'time_sampling': time_sampling,
        'time_total': time_sdp + time_sampling
    }
    
    if return_details:
        result['all_cuts'] = all_cuts
        result['mean_cut'] = np.mean(all_cuts)
        result['std_cut'] = np.std(all_cuts)
        result['min_cut'] = np.min(all_cuts)
        result['max_cut'] = np.max(all_cuts)
    
    return result


def goemans_williamson(adjacency_matrix: np.ndarray,
                      sdp_solver: Optional[SDPSolver] = None) -> Dict:
    """
    Goemans-Williamson clásico: 1 muestra uniforme.
    
    Args:
        adjacency_matrix: Matriz de adyacencia (n, n)
        sdp_solver: Instancia del solver SDP
        
    Returns:
        Mismo formato que probabilistic_gw
    """
    return probabilistic_gw(adjacency_matrix, K=1, sdp_solver=sdp_solver)

"""
SDP Solver for MaxCut - Corrected Implementation
"""
import numpy as np
import cvxpy as cp
import hashlib
import pickle
import os
import time
from typing import Dict, Any, Optional


class SDPSolver:
    """
    Resuelve la relajación SDP de MaxCut correctamente.
    Corrige problemas del solver original en solvers/SDP/evaluate.py
    """
    
    def __init__(self, 
                 solver='SCS',  # SCS es gratuito, MOSEK requiere licencia
                 cache_dir='./cache_sdp',
                 verbose=False):
        """
        Args:
            solver: 'SCS' (gratis), 'MOSEK' (requiere licencia), 'CVXOPT'
            cache_dir: Directorio para cachear soluciones SDP
            verbose: Si True, imprime información de debugging
        """
        self.solver = solver
        self.cache_dir = cache_dir
        self.verbose = verbose
        
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
    
    def solve(self, 
              adjacency_matrix: np.ndarray,
              use_cache: bool = True) -> Dict[str, Any]:
        """
        Resuelve la relajación SDP de MaxCut.
        
        Args:
            adjacency_matrix: Matriz de adyacencia (n×n) con pesos de espín {-1, +1}
            use_cache: Si True, intenta cargar solución cacheada
            
        Returns:
            {
                'vectors': np.ndarray (n, n),     # Vectores x_i como filas
                'Z_SDP': float,                   # Valor óptimo (cota superior)
                'matrix': np.ndarray (n, n),      # Matriz X = V·V^T
                'solve_time': float,
                'status': str
            }
        """
        n = adjacency_matrix.shape[0]
        
        # Verificar cache
        if use_cache and self.cache_dir:
            cache_key = self._compute_cache_key(adjacency_matrix)
            cache_path = os.path.join(self.cache_dir, f"{cache_key}.pkl")
            
            if os.path.exists(cache_path):
                if self.verbose:
                    print(f"[SDP] Cargando de cache: {cache_key}")
                with open(cache_path, 'rb') as f:
                    return pickle.load(f)
        
        # Formular problema SDP
        # Para grafos con pesos de espín {-1, +1}:
        # maximize: (1/4) * Σ_ij w_ij (1 - X_ij)
        # subject to: X_ii = 1 ∀i, X ⪰ 0
        X = cp.Variable((n, n), PSD=True)
        
        # Objetivo: maximizar el corte esperado
        objective = 0.25 * cp.sum(cp.multiply(adjacency_matrix, 1 - X))
        
        # Restricción: diagonal = 1 (vectores unitarios)
        constraints = [cp.diag(X) == 1]
        
        problem = cp.Problem(cp.Maximize(objective), constraints)
        
        # Resolver
        start_time = time.time()
        
        try:
            problem.solve(solver=self.solver, verbose=self.verbose)
            solve_time = time.time() - start_time
            
            if problem.status not in ['optimal', 'optimal_inaccurate']:
                if self.verbose:
                    print(f"[SDP] Warning: status = {problem.status}")
                return {
                    'vectors': None,
                    'Z_SDP': None,
                    'matrix': None,
                    'solve_time': solve_time,
                    'status': problem.status
                }
            
            # Extraer solución
            X_opt = X.value
            Z_SDP = problem.value
            
            # CORRECCIÓN CRÍTICA: Obtener vectores por factorización
            # El código original usaba matrix.value directamente (INCORRECTO)
            # Necesitamos factorizar X = V·V^T
            try:
                # Intentar Cholesky (más rápido si X es exactamente PSD)
                L = np.linalg.cholesky(X_opt)
                vectors = L.T  # Cada fila es un vector x_i
            except np.linalg.LinAlgError:
                # Si falla, usar eigendecomposition (más robusto)
                eigenvalues, eigenvectors = np.linalg.eigh(X_opt)
                eigenvalues = np.maximum(eigenvalues, 0)  # Forzar no-negatividad
                vectors = eigenvectors @ np.diag(np.sqrt(eigenvalues))
            
            result = {
                'vectors': vectors,
                'Z_SDP': Z_SDP,
                'matrix': X_opt,
                'solve_time': solve_time,
                'status': problem.status
            }
            
            # Guardar en cache
            if use_cache and self.cache_dir:
                with open(cache_path, 'wb') as f:
                    pickle.dump(result, f)
                if self.verbose:
                    print(f"[SDP] Guardado en cache: {cache_key}")
            
            return result
            
        except Exception as e:
            solve_time = time.time() - start_time
            if self.verbose:
                print(f"[SDP] Error: {e}")
            
            return {
                'vectors': None,
                'Z_SDP': None,
                'matrix': None,
                'solve_time': solve_time,
                'status': 'error'
            }
    
    def _compute_cache_key(self, adjacency_matrix: np.ndarray) -> str:
        """Genera un hash único del grafo."""
        matrix_bytes = adjacency_matrix.tobytes()
        return hashlib.sha256(matrix_bytes).hexdigest()[:16]


def evaluate_hyperplane_cut(vectors: np.ndarray, 
                            hyperplane: np.ndarray,
                            adjacency_matrix: np.ndarray) -> float:
    """
    Evalúa el valor del corte para un hiperplano dado.
    
    Args:
        vectors: Vectores SDP (n, d) donde cada fila es x_i
        hyperplane: Vector normal del hiperplano (d,)
        adjacency_matrix: Matriz de adyacencia (n, n)
        
    Returns:
        Valor del corte
    """
    # Asignar espines según el lado del hiperplano
    spins = np.sign(vectors @ hyperplane)
    
    # Manejar caso de proyección cero (asignar +1 por defecto)
    spins[spins == 0] = 1
    
    # Calcular valor del corte
    # cut = (1/4) * Σ_ij w_ij (1 - s_i*s_j)
    cut = 0.25 * np.sum(adjacency_matrix * (1 - np.outer(spins, spins)))
    
    return cut

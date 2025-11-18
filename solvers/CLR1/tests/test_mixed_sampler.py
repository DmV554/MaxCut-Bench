"""
Test para Mixed Sampler: Verificar mezcla convexa y best-of-K sampling.
"""
import numpy as np
import sys
import os
from scipy import sparse

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.sdp_solver import SDPSolver
from core.mixed_sampler import MixedSampler, create_clr_sampler


def test_mixed_sampler_lambda_1():
    """Test λ=1.0: Debe usar solo muestreo uniforme."""
    print("="*60)
    print("TEST 1: λ=1.0 (Solo uniforme)")
    print("="*60)
    
    # Grafo triángulo simple
    adjacency = np.array([
        [0, 1, 1],
        [1, 0, 1],
        [1, 1, 0]
    ], dtype=float)
    
    # Resolver SDP
    solver = SDPSolver(verbose=False)
    sdp_result = solver.solve(adjacency)
    vectors = sdp_result['vectors']
    
    # Crear sampler con λ=1.0 (solo uniforme)
    sampler = MixedSampler(lambda_mix=1.0, seed=42)
    
    graph_data = {'adjacency_matrix': adjacency}
    
    # Samplear con K=20
    result = sampler.sample_multiple(vectors, K=20, graph_data=graph_data, return_stats=True)
    
    print(f"Mejor corte: {result['best_cut']:.2f}")
    print(f"Corte medio: {result['mean_cut']:.2f}")
    print(f"Std cortes: {result['std_cut']:.2f}")
    print(f"Tiempo: {result['sampling_time']:.4f}s")
    
    # Verificar estadísticas
    stats = result['sampling_strategy']
    print(f"\nEstrategia de muestreo:")
    print(f"  Uniforme: {stats['uniform_samples']}/{stats['total_samples']}")
    print(f"  Aprendida: {stats['learned_samples']}/{stats['total_samples']}")
    
    assert stats['uniform_samples'] == 20, "Con λ=1.0 debe usar solo uniforme"
    assert stats['learned_samples'] == 0, "Con λ=1.0 no debe usar aprendida"
    
    print("✅ Test 1 pasado\n")


def test_mixed_sampler_with_dummy_policy():
    """Test λ=0.5 con política dummy."""
    print("="*60)
    print("TEST 2: λ=0.5 (Mezcla con política dummy)")
    print("="*60)
    
    # Política dummy: siempre retorna [1, 0, 0, ...]
    def dummy_policy(graph_data, vectors):
        d = vectors.shape[1]
        hyperplane = np.zeros(d)
        hyperplane[0] = 1.0
        return hyperplane
    
    # Grafo cuadrado K4
    adjacency = np.array([
        [0, 1, 1, 1],
        [1, 0, 1, 1],
        [1, 1, 0, 1],
        [1, 1, 1, 0]
    ], dtype=float)
    
    solver = SDPSolver(verbose=False)
    sdp_result = solver.solve(adjacency)
    vectors = sdp_result['vectors']
    
    # Crear sampler con λ=0.5
    sampler = MixedSampler(lambda_mix=0.5, learned_policy=dummy_policy, seed=42)
    
    graph_data = {'adjacency_matrix': adjacency}
    
    # Samplear con K=50
    result = sampler.sample_multiple(vectors, K=50, graph_data=graph_data, return_stats=True)
    
    print(f"Mejor corte: {result['best_cut']:.2f}")
    print(f"Z_SDP: {sdp_result['Z_SDP']:.2f}")
    print(f"Gap: {(sdp_result['Z_SDP'] - result['best_cut']) / sdp_result['Z_SDP'] * 100:.2f}%")
    
    # Verificar mezcla
    stats = result['sampling_strategy']
    print(f"\nEstrategia de muestreo:")
    print(f"  Uniforme: {stats['uniform_samples']}/{stats['total_samples']}")
    print(f"  Aprendida: {stats['learned_samples']}/{stats['total_samples']}")
    
    # Con λ=0.5, esperamos ~25 de cada tipo (con varianza)
    assert stats['total_samples'] == 50
    assert stats['uniform_samples'] + stats['learned_samples'] == 50
    
    # Tolerancia: entre 15 y 35 de cada tipo (bastante amplio)
    assert 15 <= stats['uniform_samples'] <= 35, f"Uniforme fuera de rango: {stats['uniform_samples']}"
    
    print("✅ Test 2 pasado\n")


def test_best_of_K_improves():
    """Test: Verificar que best-of-K mejora con K más grande."""
    print("="*60)
    print("TEST 3: Best-of-K mejora con K")
    print("="*60)
    
    # Grafo BA_20
    data_path = 'data/testing/BA_20'
    
    if not os.path.exists(data_path):
        print("⚠️  Datos BA_20 no encontrados, saltando test")
        return
    
    # Cargar primer grafo
    files = sorted([f for f in os.listdir(data_path) if f.endswith('.npz')])
    if not files:
        print("⚠️  No hay archivos .npz, saltando test")
        return
    
    graph_file = os.path.join(data_path, files[0])
    # Cargar como sparse matrix
    adjacency_sparse = sparse.load_npz(graph_file)
    adjacency = adjacency_sparse.toarray()
    
    # Resolver SDP
    solver = SDPSolver(verbose=False)
    sdp_result = solver.solve(adjacency)
    vectors = sdp_result['vectors']
    Z_SDP = sdp_result['Z_SDP']
    
    print(f"Grafo: {files[0]}, n={adjacency.shape[0]}")
    print(f"Z_SDP: {Z_SDP:.2f}\n")
    
    # Probar diferentes valores de K
    sampler = MixedSampler(lambda_mix=1.0, seed=42)  # Solo uniforme
    graph_data = {'adjacency_matrix': adjacency}
    
    K_values = [1, 10, 50]
    results = []
    
    for K in K_values:
        result = sampler.sample_multiple(vectors, K=K, graph_data=graph_data, return_stats=True)
        cut = result['best_cut']
        gap = (Z_SDP - cut) / Z_SDP * 100
        
        results.append(cut)
        print(f"K={K:3d}: cut={cut:.2f}, gap={gap:.2f}%")
    
    # Verificar mejora monótona
    assert results[1] >= results[0], "K=10 debe ser ≥ K=1"
    assert results[2] >= results[1], "K=50 debe ser ≥ K=10"
    
    print("\n✅ Test 3 pasado\n")


def test_lambda_ablation():
    """Test: Explorar efecto de λ en el performance."""
    print("="*60)
    print("TEST 4: Ablation de λ")
    print("="*60)
    
    # Política dummy: prefiere primer eje
    def biased_policy(graph_data, vectors):
        d = vectors.shape[1]
        hyperplane = np.random.randn(d)
        hyperplane[0] += 2.0  # Bias hacia primer eje
        return hyperplane / np.linalg.norm(hyperplane)
    
    # Grafo pequeño
    adjacency = np.array([
        [0, 1, 1, 0],
        [1, 0, 1, 1],
        [1, 1, 0, 1],
        [0, 1, 1, 0]
    ], dtype=float)
    
    solver = SDPSolver(verbose=False)
    sdp_result = solver.solve(adjacency)
    vectors = sdp_result['vectors']
    Z_SDP = sdp_result['Z_SDP']
    
    graph_data = {'adjacency_matrix': adjacency}
    
    print(f"Z_SDP: {Z_SDP:.2f}\n")
    
    lambda_values = [0.0, 0.25, 0.5, 0.75, 1.0]
    
    for lam in lambda_values:
        sampler = MixedSampler(lambda_mix=lam, learned_policy=biased_policy, seed=42)
        result = sampler.sample_multiple(vectors, K=30, graph_data=graph_data, return_stats=True)
        
        cut = result['best_cut']
        gap = (Z_SDP - cut) / Z_SDP * 100
        stats = result['sampling_strategy']
        
        print(f"λ={lam:.2f}: cut={cut:.2f}, gap={gap:.2f}%, "
              f"uniform={stats['uniform_samples']}/{stats['total_samples']}")
    
    print("\n✅ Test 4 pasado\n")


if __name__ == '__main__':
    print("\n" + "="*60)
    print("TESTS DE MIXED SAMPLER")
    print("="*60 + "\n")
    
    try:
        test_mixed_sampler_lambda_1()
        test_mixed_sampler_with_dummy_policy()
        test_best_of_K_improves()
        test_lambda_ablation()
        
        print("="*60)
        print("✅ TODOS LOS TESTS PASARON")
        print("="*60)
        
    except AssertionError as e:
        print(f"\n❌ Test falló: {e}")
        raise
    except Exception as e:
        print(f"\n❌ Error inesperado: {e}")
        raise

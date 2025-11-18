"""
Test del SDP Solver Corregido
"""
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.sdp_solver import SDPSolver, evaluate_hyperplane_cut


def test_triangle():
    """
    Test en un triángulo simple con pesos +1.
    
    Grafo: 0 -- 1
           |  / 
           | /
           2
    
    Óptimo: corte de 2 aristas = 2
    Z_SDP conocido ≈ 2.18 (de la literatura)
    """
    print("\n" + "="*60)
    print("TEST 1: Triángulo con pesos +1")
    print("="*60)
    
    adjacency = np.array([
        [0, 1, 1],
        [1, 0, 1],
        [1, 1, 0]
    ], dtype=float)
    
    solver = SDPSolver(verbose=True)
    result = solver.solve(adjacency)
    
    print(f"\nStatus: {result['status']}")
    print(f"Z_SDP = {result['Z_SDP']:.4f}")
    print(f"Tiempo: {result['solve_time']:.3f}s")
    
    # Verificaciones
    assert result['status'] in ['optimal', 'optimal_inaccurate'], "SDP no convergió"
    assert result['Z_SDP'] >= 2.0 - 1e-6, "Z_SDP debe ser >= óptimo (2.0)"
    
    # Z_SDP teórico para triángulo ≈ 2.18
    expected_sdp = 2.18
    print(f"Z_SDP esperado: {expected_sdp:.4f}")
    print(f"Diferencia: {abs(result['Z_SDP'] - expected_sdp):.4f}")
    
    # Validar que los vectores se extrajeron correctamente
    assert result['vectors'] is not None, "Vectores no extraídos"
    vectors = result['vectors']
    print(f"Shape vectores: {vectors.shape}")
    
    # Reconstruir matriz X = V·V^T
    X_reconstructed = vectors @ vectors.T
    print(f"Diagonal de X (debe ser todos 1s): {np.diag(X_reconstructed)}")
    assert np.allclose(np.diag(X_reconstructed), 1.0, atol=1e-6), "Diagonal debe ser 1"
    
    # Test de redondeo
    hyperplane = np.array([1, 0, 0])  # Hiperplano fijo para reproducibilidad
    cut = evaluate_hyperplane_cut(vectors, hyperplane, adjacency)
    print(f"Corte con hiperplano de prueba: {cut:.4f}")
    assert cut <= result['Z_SDP'] + 1e-6, "Corte no puede exceder Z_SDP"
    
    print("\n✅ TEST 1 PASADO")
    return True


def test_k4():
    """
    Test en K4 (grafo completo de 4 nodos).
    
    Óptimo: 4 aristas en el corte
    """
    print("\n" + "="*60)
    print("TEST 2: K4 (grafo completo 4 nodos)")
    print("="*60)
    
    # K4 con pesos +1
    adjacency = np.array([
        [0, 1, 1, 1],
        [1, 0, 1, 1],
        [1, 1, 0, 1],
        [1, 1, 1, 0]
    ], dtype=float)
    
    solver = SDPSolver(verbose=True)
    result = solver.solve(adjacency)
    
    print(f"\nStatus: {result['status']}")
    print(f"Z_SDP = {result['Z_SDP']:.4f}")
    print(f"Óptimo verdadero: 4.0")
    print(f"Ratio Z_SDP/OPT: {result['Z_SDP']/4.0:.4f}")
    
    assert result['status'] in ['optimal', 'optimal_inaccurate']
    assert result['Z_SDP'] >= 4.0 - 1e-6, "Z_SDP debe ser >= 4.0"
    
    print("\n✅ TEST 2 PASADO")
    return True


def test_real_graph_ba20():
    """
    Test en un grafo real del dataset BA_20.
    """
    print("\n" + "="*60)
    print("TEST 3: Grafo real de BA_20")
    print("="*60)
    
    from scipy.sparse import load_npz
    
    graph_path = 'data/training/BA_20/BA_20vertices_graph_77.npz'
    
    if not os.path.exists(graph_path):
        print(f"⚠️  Archivo no encontrado: {graph_path}")
        print("Saltando test...")
        return True
    
    adjacency = load_npz(graph_path).toarray()
    
    print(f"Grafo: {adjacency.shape[0]} nodos")
    print(f"Aristas no-cero: {np.count_nonzero(adjacency)}")
    
    solver = SDPSolver(verbose=True)
    result = solver.solve(adjacency, use_cache=False)  # Forzar resolver
    
    print(f"\nStatus: {result['status']}")
    print(f"Z_SDP = {result['Z_SDP']:.4f}")
    print(f"Tiempo: {result['solve_time']:.3f}s")
    
    assert result['status'] in ['optimal', 'optimal_inaccurate']
    assert result['Z_SDP'] > 0, "Z_SDP debe ser positivo"
    
    # Test de caching
    print("\nTest de caching...")
    result2 = solver.solve(adjacency, use_cache=True)
    print(f"Tiempo con cache: {result2['solve_time']:.6f}s (debe ser ~0)")
    assert result2['solve_time'] < 0.1, "Cache debe ser rápido"
    assert np.isclose(result2['Z_SDP'], result['Z_SDP']), "Cache debe dar mismo resultado"
    
    print("\n✅ TEST 3 PASADO")
    return True


if __name__ == '__main__':
    print("\n" + "="*60)
    print("VALIDACIÓN DEL SDP SOLVER CORREGIDO")
    print("="*60)
    
    try:
        test_triangle()
        test_k4()
        test_real_graph_ba20()
        
        print("\n" + "="*60)
        print("✅ TODOS LOS TESTS PASADOS")
        print("="*60)
        print("\nEl SDP solver está funcionando correctamente.")
        print("Puedes proceder con la implementación de CLR.")
        
    except Exception as e:
        print("\n" + "="*60)
        print("❌ TEST FALLÓ")
        print("="*60)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

"""
Script para inspeccionar la estructura de los datos .npz
"""
import numpy as np
from scipy.sparse import load_npz
import os

# Cargar un grafo de ejemplo
test_file = 'data/training/BA_20/BA_20vertices_graph_77.npz'

if os.path.exists(test_file):
    print(f"Inspeccionando: {test_file}\n")
    print("="*60)
    
    # Cargar el grafo
    graph_sparse = load_npz(test_file)
    graph_dense = graph_sparse.toarray()
    
    print(f"Formato sparse: {type(graph_sparse)}")
    print(f"Shape: {graph_sparse.shape}")
    print(f"Número de vértices: {graph_dense.shape[0]}")
    print(f"Número de aristas no-cero: {np.count_nonzero(graph_dense)}")
    print(f"Es simétrica: {np.allclose(graph_dense, graph_dense.T)}")
    print(f"Tiene diagonal: {np.count_nonzero(np.diag(graph_dense))}")
    print(f"Rango de pesos: [{graph_dense[graph_dense>0].min():.3f}, {graph_dense[graph_dense>0].max():.3f}]")
    print(f"¿Pesos binarios?: {set(graph_dense[graph_dense>0].flatten())}")
    
    print("\nMatriz de adyacencia (primeros 10x10):")
    print(graph_dense[:10, :10])
    
    print("\n" + "="*60)
    print("\nINFORMACIÓN PARA CLR:")
    print("- Los grafos están almacenados como matrices sparse en .npz")
    print("- Son matrices de adyacencia simétricas (sin diagonal)")
    print("- Representan grafos no dirigidos")
    print("- Necesitamos convertir esto a formato compatible con SDP solver")
else:
    print(f"No se encontró el archivo: {test_file}")

"""
Training script para CLR: Entrena GNN policy offline en grafos sintéticos.

Usage:
    python train.py --distribution BA_20 --num_graphs 1000 --epochs 50
    python train.py --distribution ER_200 --num_graphs 5000 --epochs 100 --gpu
"""
import argparse
import os
import sys
import numpy as np
import pickle
from scipy import sparse
import time
from typing import List, Dict
import json

# Add parent directory to path
sys.path.insert(0, os.path.dirname(__file__))

try:
    import torch
    from models.gnn_policy import GNNPolicy, create_gnn_policy
    from models.rl_trainer import RLTrainer, prepare_training_data
    from core.sdp_solver import SDPSolver
    TORCH_AVAILABLE = True
except ImportError:
    print("⚠️  PyTorch/PyTorch Geometric no instalado")
    print("   Instalar con: pip install torch torch-geometric")
    TORCH_AVAILABLE = False


def generate_synthetic_graphs(distribution: str, num_graphs: int, **kwargs) -> List[Dict]:
    """
    Genera grafos sintéticos para training.
    
    Args:
        distribution: 'BA', 'ER', 'WS', etc.
        num_graphs: Número de grafos a generar
        **kwargs: Parámetros específicos (n, m, p, etc.)
    
    Returns:
        Lista de graph_data dicts
    """
    import networkx as nx
    
    graphs = []
    
    # Parámetros por defecto según distribución
    if distribution.startswith('BA'):
        n = kwargs.get('n', 20)
        m = kwargs.get('m', 2)
        
        for i in range(num_graphs):
            G = nx.barabasi_albert_graph(n, m, seed=i)
            adjacency = nx.to_scipy_sparse_array(G, format='csr', dtype=float)
            
            # Convertir a matriz densa para compatibilidad
            adjacency_dense = adjacency.toarray()
            
            graph_data = {
                'adjacency_matrix': adjacency_dense,
                'distribution': distribution,
                'n': n,
                'id': i
            }
            graphs.append(graph_data)
    
    elif distribution.startswith('ER'):
        n = kwargs.get('n', 20)
        p = kwargs.get('p', 0.3)
        
        for i in range(num_graphs):
            G = nx.erdos_renyi_graph(n, p, seed=i)
            adjacency = nx.to_scipy_sparse_array(G, format='csr', dtype=float)
            adjacency_dense = adjacency.toarray()
            
            graph_data = {
                'adjacency_matrix': adjacency_dense,
                'distribution': distribution,
                'n': n,
                'id': i
            }
            graphs.append(graph_data)
    
    elif distribution.startswith('WS'):
        n = kwargs.get('n', 20)
        k = kwargs.get('k', 4)
        p = kwargs.get('p', 0.3)
        
        for i in range(num_graphs):
            G = nx.watts_strogatz_graph(n, k, p, seed=i)
            adjacency = nx.to_scipy_sparse_array(G, format='csr', dtype=float)
            adjacency_dense = adjacency.toarray()
            
            graph_data = {
                'adjacency_matrix': adjacency_dense,
                'distribution': distribution,
                'n': n,
                'id': i
            }
            graphs.append(graph_data)
    
    else:
        raise ValueError(f"Distribución desconocida: {distribution}")
    
    print(f"✅ Generados {len(graphs)} grafos {distribution}")
    
    return graphs


def load_graphs_from_disk(data_path: str, max_graphs: int = None) -> List[Dict]:
    """
    Carga grafos desde disco (formato .npz del benchmark).
    
    Args:
        data_path: Path a directorio con archivos .npz
        max_graphs: Máximo número de grafos a cargar
    
    Returns:
        Lista de graph_data dicts
    """
    files = sorted([f for f in os.listdir(data_path) if f.endswith('.npz')])
    
    if max_graphs is not None:
        files = files[:max_graphs]
    
    graphs = []
    
    for i, filename in enumerate(files):
        filepath = os.path.join(data_path, filename)
        
        # Cargar como sparse matrix
        adjacency_sparse = sparse.load_npz(filepath)
        adjacency_dense = adjacency_sparse.toarray()
        
        graph_data = {
            'adjacency_matrix': adjacency_dense,
            'filename': filename,
            'id': i
        }
        graphs.append(graph_data)
        
        if (i + 1) % 100 == 0:
            print(f"  Cargados {i+1}/{len(files)} grafos")
    
    print(f"✅ Cargados {len(graphs)} grafos desde {data_path}")
    
    return graphs


def train_clr(args):
    """Main training function."""
    
    if not TORCH_AVAILABLE:
        print("❌ PyTorch no disponible. Abortando.")
        return
    
    print("="*60)
    print("TRAINING CLR POLICY")
    print("="*60)
    print(f"Distribution: {args.distribution}")
    print(f"Num graphs: {args.num_graphs}")
    print(f"Epochs: {args.epochs}")
    print(f"Device: {'cuda' if args.gpu and torch.cuda.is_available() else 'cpu'}")
    print("="*60 + "\n")
    
    # 1. Generar o cargar grafos
    if args.load_from_disk:
        print(f"Cargando grafos desde {args.data_path}...")
        graphs = load_graphs_from_disk(args.data_path, max_graphs=args.num_graphs)
    else:
        print(f"Generando {args.num_graphs} grafos sintéticos...")
        graphs = generate_synthetic_graphs(
            args.distribution,
            args.num_graphs,
            n=args.n,
            m=args.m,
            p=args.p
        )
    
    # Split train/val
    num_train = int(len(graphs) * 0.8)
    train_graphs = graphs[:num_train]
    val_graphs = graphs[num_train:]
    
    print(f"Train: {len(train_graphs)} grafos")
    print(f"Val: {len(val_graphs)} grafos\n")
    
    # 2. Crear SDP solver
    print("Creando SDP solver...")
    sdp_solver = SDPSolver(
        solver=args.sdp_solver,
        cache_dir=args.cache_dir,
        verbose=False
    )
    
    # 3. Preparar datos (resolver SDPs)
    print("\nPreparando datos de training (resolviendo SDPs)...")
    start_time = time.time()
    train_data = prepare_training_data(train_graphs, sdp_solver, args.cache_dir)
    val_data = prepare_training_data(val_graphs, sdp_solver, args.cache_dir)
    prep_time = time.time() - start_time
    print(f"✅ Preparación completada en {prep_time:.2f}s\n")
    
    # 4. Crear GNN policy
    print("Creando GNN policy...")
    # Inferir dimensión de vectores SDP del primer grafo
    _, first_vectors, _ = train_data[0]
    sdp_vector_dim = first_vectors.shape[1]
    
    gnn_policy = create_gnn_policy(
        sdp_vector_dim=sdp_vector_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout
    )
    
    num_params = sum(p.numel() for p in gnn_policy.parameters())
    print(f"✅ GNN creada: {num_params:,} parámetros\n")
    
    # 5. Crear RL trainer
    print("Creando RL trainer...")
    trainer = RLTrainer(
        gnn_policy=gnn_policy,
        sdp_solver=sdp_solver,
        learning_rate=args.lr,
        baseline_K=args.baseline_K,
        policy_K=args.policy_K,
        entropy_coef=args.entropy_coef,
        use_baseline_value=args.use_value_baseline,
        device='cuda' if args.gpu else 'cpu'
    )
    print("✅ Trainer listo\n")
    
    # 6. Training loop
    print("="*60)
    print("INICIANDO ENTRENAMIENTO")
    print("="*60 + "\n")
    
    best_val_improvement = -float('inf')
    best_epoch = 0
    
    training_history = {
        'train': [],
        'val': []
    }
    
    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")
        print("-" * 40)
        
        # Train
        train_stats = trainer.train_epoch(train_data, verbose=False)
        training_history['train'].append(train_stats)
        
        print(f"Train: cut_policy={train_stats['mean_cut_policy']:.2f}, "
              f"cut_uniform={train_stats['mean_cut_uniform']:.2f}, "
              f"improvement={train_stats['improvement']:.2f}%")
        
        # Validate
        if (epoch + 1) % args.val_every == 0:
            print("Validando...")
            val_stats = trainer.validate(val_data)
            training_history['val'].append(val_stats)
            
            print(f"Val: cut_policy={val_stats['mean_cut_policy']:.2f}, "
                  f"cut_uniform={val_stats['mean_cut_uniform']:.2f}, "
                  f"improvement={val_stats['improvement']:.2f}%")
            
            # Save best model
            if val_stats['improvement'] > best_val_improvement:
                best_val_improvement = val_stats['improvement']
                best_epoch = epoch + 1
                
                save_path = os.path.join(args.save_dir, args.distribution, 'policy_best.pth')
                trainer.save_checkpoint(save_path, epoch, val_stats)
                print(f"✅ Mejor modelo guardado (improvement={best_val_improvement:.2f}%)")
        
        print()
    
    # 7. Save final model
    final_path = os.path.join(args.save_dir, args.distribution, 'policy_final.pth')
    trainer.save_checkpoint(final_path, args.epochs, train_stats)
    
    # 8. Save training history
    history_path = os.path.join(args.save_dir, args.distribution, 'training_history.pkl')
    os.makedirs(os.path.dirname(history_path), exist_ok=True)
    with open(history_path, 'wb') as f:
        pickle.dump(training_history, f)
    
    print("\n" + "="*60)
    print("ENTRENAMIENTO COMPLETADO")
    print("="*60)
    print(f"Mejor improvement (val): {best_val_improvement:.2f}% en epoch {best_epoch}")
    print(f"Modelo guardado en: {args.save_dir}/{args.distribution}/")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(description='Train CLR policy')
    
    # Data
    parser.add_argument('--distribution', type=str, default='BA_20',
                       help='Distribución de grafos (BA_20, ER_200, etc.)')
    parser.add_argument('--num_graphs', type=int, default=1000,
                       help='Número de grafos de training')
    parser.add_argument('--load_from_disk', action='store_true',
                       help='Cargar grafos desde disco en vez de generar')
    parser.add_argument('--data_path', type=str, default='data/training/BA_20',
                       help='Path a grafos (si load_from_disk=True)')
    
    # Graph parameters (para generación sintética)
    parser.add_argument('--n', type=int, default=20, help='Número de nodos')
    parser.add_argument('--m', type=int, default=2, help='Parámetro m para BA')
    parser.add_argument('--p', type=float, default=0.3, help='Probabilidad para ER/WS')
    
    # Model
    parser.add_argument('--hidden_dim', type=int, default=64, help='Hidden dimension')
    parser.add_argument('--num_layers', type=int, default=3, help='Number of GCN layers')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    
    # Training
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--baseline_K', type=int, default=10,
                       help='Number of uniform samples for baseline')
    parser.add_argument('--policy_K', type=int, default=10,
                       help='Number of policy samples per graph')
    parser.add_argument('--entropy_coef', type=float, default=0.01,
                       help='Entropy regularization coefficient')
    parser.add_argument('--use_value_baseline', action='store_true',
                       help='Use learned value baseline')
    
    # SDP
    parser.add_argument('--sdp_solver', type=str, default='SCS',
                       choices=['SCS', 'MOSEK', 'CVXOPT'],
                       help='SDP solver backend')
    parser.add_argument('--cache_dir', type=str, default='./cache_sdp',
                       help='Directory for SDP cache')
    
    # Validation & Saving
    parser.add_argument('--val_every', type=int, default=5,
                       help='Validate every N epochs')
    parser.add_argument('--save_dir', type=str, default='solvers/CLR/pretrained',
                       help='Directory to save models')
    
    # Device
    parser.add_argument('--gpu', action='store_true', help='Use GPU if available')
    
    args = parser.parse_args()
    
    # Train
    train_clr(args)


if __name__ == '__main__':
    main()

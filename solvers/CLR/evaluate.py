"""
Evaluation script para CLR: Compatible con MaxCut-Bench framework.

Usage:
    python evaluate.py --test_distribution BA_800vertices_weighted --train_distribution BA_20
    python evaluate.py --test_distribution ER_200vertices_weighted --lambda_mix 0.5 --K 50
"""
import argparse
import os
import sys
import numpy as np
import pandas as pd
from scipy import sparse
import time
from typing import List, Dict
import pickle

# Add parent directory to path
sys.path.insert(0, os.path.dirname(__file__))

try:
    import torch
    from models.gnn_policy import GNNPolicy, create_gnn_policy
    from core.sdp_solver import SDPSolver
    from core.mixed_sampler import MixedSampler, create_clr_sampler
    from core.quality_certificate import compute_quality_certificate
    TORCH_AVAILABLE = True
except ImportError:
    print("⚠️  PyTorch/PyTorch Geometric no instalado")
    print("   Instalar con: pip install torch torch-geometric")
    TORCH_AVAILABLE = False


def load_test_graphs(data_path: str) -> List[Dict]:
    """
    Carga grafos de testing desde disco.
    
    Args:
        data_path: Path a directorio con archivos .npz
    
    Returns:
        Lista de (filename, adjacency_matrix)
    """
    files = sorted([f for f in os.listdir(data_path) if f.endswith('.npz')])
    
    graphs = []
    
    for filename in files:
        filepath = os.path.join(data_path, filename)
        
        # Cargar como sparse matrix
        adjacency_sparse = sparse.load_npz(filepath)
        adjacency_dense = adjacency_sparse.toarray()
        
        graphs.append({
            'filename': filename,
            'adjacency_matrix': adjacency_dense
        })
    
    print(f"✅ Cargados {len(graphs)} grafos de testing")
    
    return graphs


def evaluate_clr(args):
    """Main evaluation function."""
    
    if not TORCH_AVAILABLE:
        print("❌ PyTorch no disponible. Usando solo baseline pGW.")
        use_learned_policy = False
    else:
        use_learned_policy = (args.lambda_mix < 1.0)
    
    print("="*60)
    print("EVALUANDO CLR")
    print("="*60)
    print(f"Test distribution: {args.test_distribution}")
    print(f"Train distribution: {args.train_distribution}")
    print(f"Lambda: {args.lambda_mix}")
    print(f"K: {args.K}")
    print(f"Use learned policy: {use_learned_policy}")
    print("="*60 + "\n")
    
    # 1. Load test graphs
    data_path = os.path.join('data/testing', args.test_distribution)
    if not os.path.exists(data_path):
        print(f"❌ Path no encontrado: {data_path}")
        return
    
    test_graphs = load_test_graphs(data_path)
    n_tests = len(test_graphs)
    
    # 2. Create SDP solver
    print("Creando SDP solver...")
    sdp_solver = SDPSolver(
        solver=args.sdp_solver,
        cache_dir=args.cache_dir,
        verbose=False
    )
    
    # 3. Load GNN policy (si se usa)
    gnn_policy = None
    if use_learned_policy:
        print(f"Cargando GNN policy de {args.train_distribution}...")
        pretrained_path = os.path.join(
            args.pretrained_dir,
            args.train_distribution,
            'policy_best.pth'
        )
        
        if not os.path.exists(pretrained_path):
            print(f"⚠️  Modelo no encontrado: {pretrained_path}")
            print("   Usando solo baseline uniforme (λ=1.0)")
            args.lambda_mix = 1.0
            use_learned_policy = False
        else:
            # Inferir dimensión de primer grafo
            sdp_result = sdp_solver.solve(test_graphs[0]['adjacency_matrix'])
            sdp_dim = sdp_result['vectors'].shape[1]
            
            gnn_policy = create_gnn_policy(
                sdp_vector_dim=sdp_dim,
                hidden_dim=args.hidden_dim,
                num_layers=args.num_layers,
                dropout=args.dropout,
                pretrained_path=pretrained_path
            )
            print("✅ GNN policy cargada")
    
    # 4. Create CLR sampler
    print("Creando CLR sampler...")
    sampler = create_clr_sampler(
        lambda_mix=args.lambda_mix,
        gnn_policy=gnn_policy,
        seed=args.seed
    )
    print("✅ Sampler listo\n")
    
    # 5. Evaluate on all test graphs
    print("="*60)
    print("EVALUANDO EN GRAFOS DE TEST")
    print("="*60 + "\n")
    
    results = []
    
    for i, graph_data in enumerate(test_graphs):
        start_time = time.time()
        
        adjacency_matrix = graph_data['adjacency_matrix']
        filename = graph_data['filename']
        
        # Resolver SDP
        sdp_result = sdp_solver.solve(adjacency_matrix)
        
        if sdp_result['status'] not in ['optimal', 'optimal_inaccurate']:
            print(f"⚠️  Grafo {i+1}/{n_tests}: SDP no convergió, saltando")
            continue
        
        vectors = sdp_result['vectors']
        Z_SDP = sdp_result['Z_SDP']
        time_sdp = sdp_result['solve_time']
        
        # CLR sampling
        graph_dict = {'adjacency_matrix': adjacency_matrix}
        clr_result = sampler.sample_multiple(
            vectors,
            K=args.K,
            graph_data=graph_dict,
            return_stats=True
        )
        
        cut = clr_result['best_cut']
        time_sampling = clr_result['sampling_time']
        time_total = time_sdp + time_sampling
        
        # Quality certificate
        cert = compute_quality_certificate(cut, Z_SDP)
        
        # Mixing ratio
        mixing_stats = sampler.get_mixing_ratio()
        
        # Store results
        result = {
            'filename': filename,
            'cut': cut,
            'time': time_total,
            'time_sdp': time_sdp,
            'time_sampling': time_sampling,
            'Z_SDP': Z_SDP,
            'gap_absolute': cert['gap_absolute'],
            'gap_relative': cert['gap_relative'],
            'approx_ratio': cert['approx_ratio'],
            'certification_level': cert['certification_level'],
            'lambda_mix': args.lambda_mix,
            'K': args.K,
            'uniform_fraction': mixing_stats['uniform_fraction'],
            'learned_fraction': mixing_stats['learned_fraction'],
            'mean_cut': clr_result.get('mean_cut', cut),
            'std_cut': clr_result.get('std_cut', 0.0)
        }
        
        results.append(result)
        
        # Print progress
        if (i + 1) % 10 == 0 or (i + 1) == n_tests:
            print(f"Grafo {i+1}/{n_tests}: cut={cut:.2f}, "
                  f"gap={cert['gap_relative']*100:.2f}%, "
                  f"time={time_total:.3f}s")
    
    # 6. Convert to DataFrame
    df_results = pd.DataFrame(results)
    
    # Add metadata columns (compatible con benchmark)
    df_results['Train Distribution'] = args.train_distribution
    df_results['Test Distribution'] = args.test_distribution
    
    # 7. Save results
    save_folder = os.path.join('results', args.test_distribution)
    os.makedirs(save_folder, exist_ok=True)
    
    save_path = os.path.join(save_folder, 'CLR')
    df_results.to_pickle(save_path)
    
    print(f"\n✅ Resultados guardados en {save_path}")
    
    # 8. Print summary
    print("\n" + "="*60)
    print("RESUMEN DE RESULTADOS")
    print("="*60)
    print(f"Grafos evaluados: {len(df_results)}")
    print(f"\nCortes:")
    print(f"  Mean: {df_results['cut'].mean():.2f}")
    print(f"  Std: {df_results['cut'].std():.2f}")
    print(f"  Min: {df_results['cut'].min():.2f}")
    print(f"  Max: {df_results['cut'].max():.2f}")
    print(f"\nGaps relativos:")
    print(f"  Mean: {df_results['gap_relative'].mean()*100:.2f}%")
    print(f"  Std: {df_results['gap_relative'].std()*100:.2f}%")
    print(f"\nTiempo promedio: {df_results['time'].mean():.3f}s")
    print(f"  SDP: {df_results['time_sdp'].mean():.3f}s")
    print(f"  Sampling: {df_results['time_sampling'].mean():.3f}s")
    print(f"\nCertificación:")
    print(df_results['certification_level'].value_counts())
    print("="*60)
    
    return df_results


def main():
    parser = argparse.ArgumentParser(description='Evaluate CLR on test distribution')
    
    # Data
    parser.add_argument('--test_distribution', type=str, required=True,
                       help='Test distribution (e.g., BA_800vertices_weighted)')
    parser.add_argument('--train_distribution', type=str, default=None,
                       help='Train distribution (e.g., BA_20). If None, uses test_distribution.')
    
    # CLR parameters
    parser.add_argument('--lambda_mix', type=float, default=0.5,
                       help='Mixing parameter λ ∈ [0,1]. 1.0=pure uniform, 0.0=pure learned')
    parser.add_argument('--K', type=int, default=50,
                       help='Number of samples for best-of-K')
    
    # Model
    parser.add_argument('--hidden_dim', type=int, default=64, help='Hidden dimension')
    parser.add_argument('--num_layers', type=int, default=3, help='Number of GCN layers')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--pretrained_dir', type=str, default='solvers/CLR/pretrained',
                       help='Directory with pretrained models')
    
    # SDP
    parser.add_argument('--sdp_solver', type=str, default='SCS',
                       choices=['SCS', 'MOSEK', 'CVXOPT'],
                       help='SDP solver backend')
    parser.add_argument('--cache_dir', type=str, default='./cache_sdp',
                       help='Directory for SDP cache')
    
    # Other
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    # Default train_distribution = test_distribution
    if args.train_distribution is None:
        args.train_distribution = args.test_distribution
    
    # Evaluate
    evaluate_clr(args)


if __name__ == '__main__':
    main()

"""
Multi-testing script: Evalúa CLR en múltiples distribuciones de test.

Usage:
    python evaluate_multi.py --train_distribution BA_20
    python evaluate_multi.py --train_distribution BA_20 --test_distributions BA_200vertices_weighted ER_200vertices_weighted
"""
import argparse
import os
import subprocess
import sys
from typing import List

# Distribuciones de test disponibles por defecto
DEFAULT_TEST_DISTRIBUTIONS = [
    'BA_20',
    'BA_200vertices_weighted',
    'BA_800vertices_unweighted',
    'BA_800vertices_weighted',
    'dense_MC_100_200vertices_unweighted',
    'ER_200vertices_weighted',
    'ER_800vertices_unweighted',
    'ER_800vertices_weighted',
    'HomleKim_800vertices_unweighted',
    'HomleKim_800vertices_weighted',
    'Physics',
    'planar_800vertices_unweighted',
    'planar_800vertices_weighted',
    'SK_spin_70_100vertices_weighted',
    'torodial_800vertices_weighted',
    'WattsStrogatz_800vertices_unweighted',
    'WattsStrogatz_800vertices_weighted'
]


def get_available_test_distributions() -> List[str]:
    """Lista todas las distribuciones de test disponibles."""
    # Navegar a la raíz del proyecto
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))  # MaxCut-Bench/
    test_dir = os.path.join(project_root, 'data', 'testing')
    
    if not os.path.exists(test_dir):
        print(f"⚠️  Directorio de testing no encontrado: {test_dir}")
        return []
    
    distributions = [d for d in os.listdir(test_dir) 
                     if os.path.isdir(os.path.join(test_dir, d))]
    
    return sorted(distributions)


def run_evaluation(train_distribution: str,
                   test_distribution: str,
                   lambda_mix: float = 0.5,
                   K: int = 50,
                   **kwargs) -> bool:
    """
    Ejecuta evaluación en una distribución de test.
    
    Returns:
        True si exitoso, False si falla
    """
    print("\n" + "="*80)
    print(f"EVALUANDO: {test_distribution}")
    print("="*80)
    
    # Determinar ruta correcta del script evaluate.py
    script_dir = os.path.dirname(os.path.abspath(__file__))
    evaluate_script = os.path.join(script_dir, 'evaluate.py')
    
    cmd = [
        sys.executable,  # python
        evaluate_script,
        '--test_distribution', test_distribution,
        '--train_distribution', train_distribution,
        '--lambda_mix', str(lambda_mix),
        '--K', str(K)
    ]
    
    # Agregar parámetros adicionales
    for key, value in kwargs.items():
        cmd.extend([f'--{key}', str(value)])
    
    try:
        result = subprocess.run(cmd, check=True)
        print(f"✅ {test_distribution} completado")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {test_distribution} falló con código {e.returncode}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Evaluate CLR on multiple test distributions')
    
    # Required
    parser.add_argument('--train_distribution', type=str, required=True,
                       help='Train distribution (e.g., BA_20)')
    
    # Test distributions
    parser.add_argument('--test_distributions', type=str, nargs='+',
                       help='Test distributions (default: all available)')
    parser.add_argument('--exclude', type=str, nargs='+', default=[],
                       help='Distributions to exclude')
    
    # CLR parameters
    parser.add_argument('--lambda_mix', type=float, default=0.5,
                       help='Mixing parameter λ')
    parser.add_argument('--K', type=int, default=50,
                       help='Number of samples')
    
    # Model
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--dropout', type=float, default=0.1)
    
    # SDP
    parser.add_argument('--sdp_solver', type=str, default='SCS')
    parser.add_argument('--cache_dir', type=str, default='./cache_sdp')
    
    args = parser.parse_args()
    
    # Determinar qué distribuciones testear
    if args.test_distributions is not None:
        test_distributions = args.test_distributions
    else:
        # Usar todas las disponibles
        test_distributions = get_available_test_distributions()
    
    # Filtrar excluidas
    test_distributions = [d for d in test_distributions if d not in args.exclude]
    
    if not test_distributions:
        print("❌ No hay distribuciones de test para evaluar")
        return
    
    print("="*80)
    print("MULTI-TESTING CLR")
    print("="*80)
    print(f"Train distribution: {args.train_distribution}")
    print(f"Test distributions: {len(test_distributions)}")
    print(f"Lambda: {args.lambda_mix}")
    print(f"K: {args.K}")
    print("="*80)
    print("\nDistribuciones a evaluar:")
    for i, dist in enumerate(test_distributions, 1):
        print(f"  {i}. {dist}")
    print()
    
    # Ejecutar evaluaciones
    results = {}
    
    for dist in test_distributions:
        success = run_evaluation(
            train_distribution=args.train_distribution,
            test_distribution=dist,
            lambda_mix=args.lambda_mix,
            K=args.K,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            dropout=args.dropout,
            sdp_solver=args.sdp_solver,
            cache_dir=args.cache_dir
        )
        results[dist] = success
    
    # Resumen final
    print("\n" + "="*80)
    print("RESUMEN FINAL")
    print("="*80)
    
    successful = [d for d, s in results.items() if s]
    failed = [d for d, s in results.items() if not s]
    
    print(f"\n✅ Exitosos: {len(successful)}/{len(test_distributions)}")
    for dist in successful:
        print(f"  ✓ {dist}")
    
    if failed:
        print(f"\n❌ Fallidos: {len(failed)}/{len(test_distributions)}")
        for dist in failed:
            print(f"  ✗ {dist}")
    
    print("\n" + "="*80)
    print(f"Resultados guardados en: results/<distribution>/CLR")
    print("="*80)
    
    # Exit code
    sys.exit(0 if not failed else 1)


if __name__ == '__main__':
    main()

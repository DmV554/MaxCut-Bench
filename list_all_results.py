#!/usr/bin/env python
"""
Script para listar y explorar todos los resultados disponibles.
Muestra una vista general de qué solvers se ejecutaron en qué datasets.

Uso:
    python list_all_results.py
"""

import pandas as pd
from pathlib import Path
from collections import defaultdict


def read_result(filepath):
    """Lee un archivo de resultado pickle"""
    try:
        with open(filepath, 'rb') as f:
            return pd.read_pickle(f)
    except:
        return None


def main():
    results_dir = Path('results')
    
    if not results_dir.exists():
        print("❌ No existe el directorio 'results'")
        return
    
    # Recopilar información
    datasets = defaultdict(list)
    
    for dataset_dir in sorted(results_dir.iterdir()):
        if not dataset_dir.is_dir():
            continue
        
        dataset_name = dataset_dir.name
        
        for result_file in dataset_dir.iterdir():
            solver_name = result_file.name
            df = read_result(result_file)
            
            if df is not None:
                info = {
                    'solver': solver_name,
                    'instances': len(df),
                }
                
                # Algunos resultados tienen columnas diferentes
                if 'cut' in df.columns:
                    info['avg_cut'] = df['cut'].mean()
                    info['best_cut'] = df['cut'].max()
                else:
                    info['avg_cut'] = 0
                    info['best_cut'] = 0
                
                if 'time' in df.columns:
                    info['avg_time'] = df['time'].mean()
                else:
                    info['avg_time'] = 0
                
                datasets[dataset_name].append(info)
    
    if not datasets:
        print("❌ No se encontraron resultados")
        return
    
    print(f"\n{'='*80}")
    print(f"🔍 RESUMEN DE TODOS LOS RESULTADOS")
    print(f"{'='*80}\n")
    
    # Mostrar por dataset
    for dataset_name, solvers in sorted(datasets.items()):
        print(f"\n📁 Dataset: {dataset_name}")
        print(f"{'─'*80}")
        
        if solvers:
            df_summary = pd.DataFrame(solvers)
            df_summary = df_summary.sort_values('avg_cut', ascending=False)
            
            # Formato mejorado
            print(f"{'Solver':<25} {'Instances':>10} {'Avg Cut':>12} {'Best Cut':>12} {'Avg Time (s)':>15}")
            print(f"{'─'*25} {' ':<10} {' ':<12} {' ':<12} {' ':<15}")
            
            for _, row in df_summary.iterrows():
                print(f"{row['solver']:<25} {row['instances']:>10} "
                      f"{row['avg_cut']:>12.2f} {row['best_cut']:>12.0f} "
                      f"{row['avg_time']:>15.6f}")
        else:
            print("  (sin resultados)")
    
    # Resumen global
    print(f"\n{'='*80}")
    print(f"📊 RESUMEN GLOBAL")
    print(f"{'='*80}")
    print(f"  Total de datasets con resultados: {len(datasets)}")
    
    all_solvers = set()
    for solvers in datasets.values():
        for s in solvers:
            all_solvers.add(s['solver'])
    
    print(f"  Total de solvers únicos: {len(all_solvers)}")
    print(f"\n  Solvers disponibles:")
    for solver in sorted(all_solvers):
        print(f"    - {solver}")
    
    print(f"\n💡 Usa 'python view_results.py --dataset <DATASET>' para ver detalles\n")


if __name__ == '__main__':
    main()

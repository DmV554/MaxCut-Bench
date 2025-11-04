#!/usr/bin/env python
"""
Script para visualizar los resultados de los experimentos de MaxCut.
Los resultados se guardan como archivos pickle de Pandas DataFrames.

Uso:
    python view_results.py --dataset BA_20 --solver "Standard Greedy"
    python view_results.py --dataset BA_20  # muestra todos los solvers
"""

import pandas as pd
import pickle
import argparse
from pathlib import Path


def read_result(filepath):
    """Lee un archivo de resultado pickle"""
    with open(filepath, 'rb') as f:
        return pd.read_pickle(f)


def main():
    parser = argparse.ArgumentParser(description='Visualizar resultados de MaxCut')
    parser.add_argument('--dataset', type=str, default='BA_20', 
                        help='Dataset a visualizar (ej: BA_20)')
    parser.add_argument('--solver', type=str, default=None,
                        help='Solver específico (ej: "Standard Greedy"). Si no se especifica, muestra todos.')
    parser.add_argument('--format', type=str, default='table', choices=['table', 'csv', 'summary'],
                        help='Formato de salida: table, csv, o summary')
    args = parser.parse_args()
    
    results_dir = Path('results') / args.dataset
    
    if not results_dir.exists():
        print(f"❌ No se encontró el directorio: {results_dir}")
        print(f"\n📁 Directorios disponibles:")
        for d in Path('results').iterdir():
            if d.is_dir():
                print(f"   - {d.name}")
        return
    
    # Encontrar archivos de resultados
    result_files = list(results_dir.iterdir())
    
    if not result_files:
        print(f"❌ No hay resultados en {results_dir}")
        return
    
    print(f"\n{'='*70}")
    print(f"📊 Resultados para dataset: {args.dataset}")
    print(f"{'='*70}\n")
    
    # Filtrar por solver si se especificó
    if args.solver:
        result_files = [f for f in result_files if f.name == args.solver]
        if not result_files:
            print(f"❌ No se encontró el solver: {args.solver}")
            print(f"\n🔍 Solvers disponibles:")
            for f in results_dir.iterdir():
                print(f"   - {f.name}")
            return
    
    # Leer y mostrar resultados
    all_results = {}
    
    for result_file in sorted(result_files):
        try:
            df = read_result(result_file)
            solver_name = result_file.name
            all_results[solver_name] = df
            
            print(f"\n🔹 Solver: {solver_name}")
            print(f"{'─'*70}")
            
            if args.format == 'table':
                print(df.to_string(index=False))
                if 'cut' in df.columns and 'time' in df.columns:
                    print(f"\n📈 Estadísticas:")
                    print(df[['cut', 'time']].describe().to_string())
                elif 'cut' in df.columns:
                    print(f"\n📈 Estadísticas:")
                    print(df['cut'].describe().to_string())
            
            elif args.format == 'csv':
                print(df.to_csv(index=False))
            
            elif args.format == 'summary':
                print(f"  Instancias: {len(df)}")
                if 'cut' in df.columns:
                    print(f"  Cut promedio: {df['cut'].mean():.2f} ± {df['cut'].std():.2f}")
                    print(f"  Mejor cut: {df['cut'].max():.0f}")
                if 'time' in df.columns:
                    print(f"  Tiempo promedio: {df['time'].mean():.6f}s")
            
        except Exception as e:
            print(f"❌ Error leyendo {result_file.name}: {e}")
    
    # Comparación entre solvers
    if len(all_results) > 1 and args.format == 'summary':
        print(f"\n{'='*70}")
        print(f"📊 Comparación de solvers")
        print(f"{'='*70}")
        
        comparison_data = {
            'Solver': list(all_results.keys()),
            'Instances': [len(df) for df in all_results.values()]
        }
        
        # Añadir columnas según disponibilidad
        first_df = list(all_results.values())[0]
        if 'cut' in first_df.columns:
            comparison_data['Avg Cut'] = [df['cut'].mean() if 'cut' in df.columns else 0 for df in all_results.values()]
            comparison_data['Std Cut'] = [df['cut'].std() if 'cut' in df.columns else 0 for df in all_results.values()]
        
        if 'time' in first_df.columns:
            comparison_data['Avg Time (s)'] = [df['time'].mean() if 'time' in df.columns else 0 for df in all_results.values()]
        
        comparison = pd.DataFrame(comparison_data)
        
        if 'Avg Cut' in comparison.columns:
            comparison = comparison.sort_values('Avg Cut', ascending=False)
        
        print(comparison.to_string(index=False))


if __name__ == '__main__':
    main()

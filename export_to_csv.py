#!/usr/bin/env python
"""
Exportar todos los resultados a archivos CSV para análisis externo.

Uso:
    python export_to_csv.py
    python export_to_csv.py --dataset BA_800vertices_weighted
"""

import pandas as pd
from pathlib import Path
import argparse


def read_result(filepath):
    """Lee un archivo de resultado pickle"""
    try:
        with open(filepath, 'rb') as f:
            return pd.read_pickle(f)
    except:
        return None


def main():
    parser = argparse.ArgumentParser(description='Exportar resultados a CSV')
    parser.add_argument('--dataset', type=str, default=None,
                        help='Dataset específico (opcional). Si no se especifica, exporta todos.')
    parser.add_argument('--output-dir', type=str, default='results_csv',
                        help='Directorio de salida para los CSVs')
    args = parser.parse_args()
    
    results_dir = Path('results')
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    if not results_dir.exists():
        print("❌ No existe el directorio 'results'")
        return
    
    # Determinar datasets a procesar
    if args.dataset:
        datasets = [results_dir / args.dataset]
        if not datasets[0].exists():
            print(f"❌ No existe el dataset: {args.dataset}")
            return
    else:
        datasets = [d for d in results_dir.iterdir() if d.is_dir()]
    
    print(f"\n{'='*70}")
    print(f"📊 EXPORTANDO RESULTADOS A CSV")
    print(f"{'='*70}\n")
    
    total_exported = 0
    
    for dataset_dir in sorted(datasets):
        dataset_name = dataset_dir.name
        print(f"\n📁 Procesando dataset: {dataset_name}")
        
        # Crear subdirectorio para este dataset
        dataset_output_dir = output_dir / dataset_name
        dataset_output_dir.mkdir(exist_ok=True)
        
        # También crear un CSV combinado
        combined_results = []
        
        for result_file in sorted(dataset_dir.iterdir()):
            solver_name = result_file.name
            df = read_result(result_file)
            
            if df is not None:
                # Guardar CSV individual
                csv_filename = f"{solver_name.replace(' ', '_')}.csv"
                csv_path = dataset_output_dir / csv_filename
                df.to_csv(csv_path, index=False)
                print(f"  ✓ {solver_name} → {csv_path}")
                
                # Añadir al combinado
                df_copy = df.copy()
                df_copy['solver'] = solver_name
                combined_results.append(df_copy)
                
                total_exported += 1
        
        # Guardar CSV combinado
        if combined_results:
            combined_df = pd.concat(combined_results, ignore_index=True)
            combined_path = output_dir / f"{dataset_name}_combined.csv"
            combined_df.to_csv(combined_path, index=False)
            print(f"  ✓ Combinado → {combined_path}")
    
    print(f"\n{'='*70}")
    print(f"✅ Exportación completada!")
    print(f"   Total de archivos exportados: {total_exported}")
    print(f"   Directorio de salida: {output_dir.absolute()}")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()

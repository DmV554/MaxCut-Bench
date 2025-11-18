"""
Script de comparación de CLR vs otros métodos en MaxCut-Bench.

Este script:
1. Carga resultados de CLR y todos los baselines
2. Calcula approximation ratios
3. Genera tablas comparativas
4. Muestra métricas únicas de CLR (gaps certificados)
"""
import os
import pandas as pd
import numpy as np
from tabulate import tabulate
import pickle


def load_results(root_folder='results'):
    """
    Carga todos los resultados disponibles.
    
    Returns:
        dict: {dataset: {algorithm: DataFrame}}
    """
    dataset_results = {}
    
    for dataset in os.listdir(root_folder):
        dataset_path = os.path.join(root_folder, dataset)
        if not os.path.isdir(dataset_path):
            continue
        
        dataset_results[dataset] = {}
        
        # Cargar optimal values
        optimal_path = os.path.join('data/testing', dataset, 'optimal')
        if os.path.exists(optimal_path):
            try:
                opt_df = pd.read_pickle(optimal_path)
                dataset_results[dataset]['_OPT'] = opt_df['OPT'].values
            except:
                print(f"⚠️  No se pudo cargar optimal para {dataset}")
                dataset_results[dataset]['_OPT'] = None
        
        # Cargar resultados de algoritmos
        for algorithm in os.listdir(dataset_path):
            algo_path = os.path.join(dataset_path, algorithm)
            try:
                df = pd.read_pickle(algo_path)
                dataset_results[dataset][algorithm] = df
            except Exception as e:
                print(f"⚠️  Error cargando {dataset}/{algorithm}: {e}")
    
    return dataset_results


def compute_approximation_ratios(dataset_results):
    """
    Calcula approximation ratios para todos los algoritmos.
    
    Returns:
        dict: {dataset: {algorithm: mean_ratio}}
    """
    ratios = {}
    
    for dataset, algos in dataset_results.items():
        if '_OPT' not in algos or algos['_OPT'] is None:
            continue
        
        OPT = algos['_OPT']
        ratios[dataset] = {}
        
        # Actualizar OPT con mejores cortes encontrados
        for algo_name, df in algos.items():
            if algo_name == '_OPT':
                continue
            if 'cut' in df.columns:
                OPT = np.maximum(OPT, df['cut'].values)
        
        # Calcular ratios
        for algo_name, df in algos.items():
            if algo_name == '_OPT':
                continue
            if 'cut' in df.columns:
                ratio = (df['cut'].values / OPT).mean()
                ratios[dataset][algo_name] = ratio
    
    return ratios


def create_comparison_table(ratios, highlight_clr=True):
    """
    Crea tabla comparativa de approximation ratios.
    """
    # Obtener todos los algoritmos
    all_algorithms = sorted(set(
        algo for dataset_algos in ratios.values() 
        for algo in dataset_algos.keys()
    ))
    
    # Ordenar con CLR al final
    if 'CLR' in all_algorithms and highlight_clr:
        all_algorithms.remove('CLR')
        all_algorithms.append('CLR')
    
    # Crear tabla
    table_data = []
    for dataset in sorted(ratios.keys()):
        row = [dataset]
        for algo in all_algorithms:
            if algo in ratios[dataset]:
                ratio = ratios[dataset][algo]
                row.append(f"{ratio:.4f}")
            else:
                row.append("N/A")
        table_data.append(row)
    
    # Headers
    headers = ["Dataset"] + all_algorithms
    
    return table_data, headers


def analyze_clr_specific_metrics(dataset_results):
    """
    Analiza métricas únicas de CLR (gaps certificados).
    """
    clr_metrics = {}
    
    for dataset, algos in dataset_results.items():
        if 'CLR' not in algos:
            continue
        
        df = algos['CLR']
        
        # Verificar que tenga columnas de certificación
        if 'gap_relative' not in df.columns:
            continue
        
        metrics = {
            'mean_gap': df['gap_relative'].mean() * 100,
            'std_gap': df['gap_relative'].std() * 100,
            'max_gap': df['gap_relative'].max() * 100,
            'min_gap': df['gap_relative'].min() * 100,
        }
        
        # Contar por certification level
        if 'certification_level' in df.columns:
            level_counts = df['certification_level'].value_counts()
            total = len(df)
            metrics['excellent_pct'] = (level_counts.get('excellent', 0) / total) * 100
            metrics['good_pct'] = (level_counts.get('good', 0) / total) * 100
            metrics['acceptable_pct'] = (level_counts.get('acceptable', 0) / total) * 100
            metrics['poor_pct'] = (level_counts.get('poor', 0) / total) * 100
        
        # Tiempos
        if 'time_sdp' in df.columns and 'time_sampling' in df.columns:
            metrics['mean_time_sdp'] = df['time_sdp'].mean()
            metrics['mean_time_sampling'] = df['time_sampling'].mean()
            metrics['mean_time_total'] = df['time'].mean()
        
        clr_metrics[dataset] = metrics
    
    return clr_metrics


def print_clr_certification_table(clr_metrics):
    """
    Imprime tabla de certificación de CLR.
    """
    print("\n" + "="*80)
    print("CLR QUALITY CERTIFICATION (Métrica Única)")
    print("="*80)
    
    if not clr_metrics:
        print("⚠️  No hay resultados de CLR con métricas de certificación")
        return
    
    table_data = []
    for dataset in sorted(clr_metrics.keys()):
        m = clr_metrics[dataset]
        row = [
            dataset,
            f"{m['mean_gap']:.2f}%",
            f"{m['std_gap']:.2f}%",
            f"{m.get('excellent_pct', 0):.1f}%",
            f"{m.get('good_pct', 0):.1f}%",
            f"{m.get('acceptable_pct', 0):.1f}%",
        ]
        table_data.append(row)
    
    headers = ["Dataset", "Mean Gap", "Std Gap", "Excellent", "Good", "Acceptable"]
    print(tabulate(table_data, headers=headers, tablefmt="grid"))
    
    # Resumen global
    print("\nResumen Global:")
    all_mean_gaps = [m['mean_gap'] for m in clr_metrics.values()]
    all_excellent = [m.get('excellent_pct', 0) for m in clr_metrics.values()]
    
    print(f"  Gap promedio global: {np.mean(all_mean_gaps):.2f}% ± {np.std(all_mean_gaps):.2f}%")
    print(f"  % instancias con certificación EXCELLENT: {np.mean(all_excellent):.1f}%")


def print_time_comparison(dataset_results, clr_metrics):
    """
    Compara tiempos de ejecución.
    """
    print("\n" + "="*80)
    print("COMPARACIÓN DE TIEMPOS DE EJECUCIÓN")
    print("="*80)
    
    table_data = []
    
    for dataset in sorted(dataset_results.keys()):
        algos = dataset_results[dataset]
        
        row = [dataset]
        
        # Tiempos de algoritmos principales
        for algo_name in ['TS', 'EO', 'ECO-DQN', 'CLR']:
            if algo_name in algos and 'time' in algos[algo_name].columns:
                mean_time = algos[algo_name]['time'].mean()
                row.append(f"{mean_time:.3f}s")
            else:
                row.append("N/A")
        
        # Para CLR, mostrar descomposición
        if dataset in clr_metrics and 'mean_time_sdp' in clr_metrics[dataset]:
            m = clr_metrics[dataset]
            sdp_time = m['mean_time_sdp']
            samp_time = m['mean_time_sampling']
            row.append(f"{sdp_time:.3f}s")
            row.append(f"{samp_time:.3f}s")
        else:
            row.append("N/A")
            row.append("N/A")
        
        table_data.append(row)
    
    headers = ["Dataset", "Tabu Search", "EO", "ECO-DQN", "CLR Total", "CLR SDP", "CLR Sampling"]
    print(tabulate(table_data, headers=headers, tablefmt="grid"))


def compare_clr_vs_best_baseline(ratios):
    """
    Compara CLR vs el mejor baseline por dataset.
    """
    print("\n" + "="*80)
    print("CLR vs MEJOR BASELINE")
    print("="*80)
    
    table_data = []
    
    for dataset in sorted(ratios.keys()):
        if 'CLR' not in ratios[dataset]:
            continue
        
        clr_ratio = ratios[dataset]['CLR']
        
        # Encontrar mejor baseline (excluir CLR)
        baselines = {k: v for k, v in ratios[dataset].items() if k != 'CLR'}
        if not baselines:
            continue
        
        best_baseline_name = max(baselines, key=baselines.get)
        best_baseline_ratio = baselines[best_baseline_name]
        
        # Calcular diferencia
        diff = clr_ratio - best_baseline_ratio
        diff_pct = (diff / best_baseline_ratio) * 100
        
        status = "✅" if diff > 0 else ("⚠️" if diff > -0.01 else "❌")
        
        row = [
            dataset,
            f"{clr_ratio:.4f}",
            best_baseline_name,
            f"{best_baseline_ratio:.4f}",
            f"{diff:+.4f}",
            f"{diff_pct:+.2f}%",
            status
        ]
        table_data.append(row)
    
    headers = ["Dataset", "CLR", "Best Baseline", "Best Ratio", "Diff", "Diff %", "Status"]
    print(tabulate(table_data, headers=headers, tablefmt="grid"))
    
    # Resumen
    wins = sum(1 for row in table_data if row[-1] == "✅")
    ties = sum(1 for row in table_data if row[-1] == "⚠️")
    losses = sum(1 for row in table_data if row[-1] == "❌")
    total = len(table_data)
    
    print(f"\nResumen:")
    print(f"  Victorias: {wins}/{total} ({wins/total*100:.1f}%)")
    print(f"  Empates: {ties}/{total} ({ties/total*100:.1f}%)")
    print(f"  Derrotas: {losses}/{total} ({losses/total*100:.1f}%)")


def main():
    print("="*80)
    print("COMPARACIÓN CLR vs STATE-OF-THE-ART")
    print("="*80)
    print()
    
    # 1. Cargar resultados
    print("📂 Cargando resultados...")
    dataset_results = load_results()
    
    datasets_with_clr = [d for d in dataset_results if 'CLR' in dataset_results[d]]
    total_datasets = len(dataset_results)
    
    print(f"   Datasets totales: {total_datasets}")
    print(f"   Datasets con CLR: {len(datasets_with_clr)}")
    print(f"   Datasets: {', '.join(datasets_with_clr)}")
    print()
    
    # 2. Calcular approximation ratios
    print("📊 Calculando approximation ratios...")
    ratios = compute_approximation_ratios(dataset_results)
    print(f"   ✅ Ratios calculados para {len(ratios)} datasets")
    print()
    
    # 3. Tabla principal de comparación
    print("="*80)
    print("TABLA DE APPROXIMATION RATIOS")
    print("="*80)
    table_data, headers = create_comparison_table(ratios)
    print(tabulate(table_data, headers=headers, tablefmt="grid"))
    
    # 4. Análisis CLR específico
    print("\n" + "="*80)
    print("ANÁLISIS ESPECÍFICO DE CLR")
    print("="*80)
    clr_metrics = analyze_clr_specific_metrics(dataset_results)
    print(f"Datasets con métricas CLR: {len(clr_metrics)}")
    
    # 5. Tabla de certificación (única de CLR)
    print_clr_certification_table(clr_metrics)
    
    # 6. Comparación de tiempos
    print_time_comparison(dataset_results, clr_metrics)
    
    # 7. CLR vs mejor baseline
    compare_clr_vs_best_baseline(ratios)
    
    print("\n" + "="*80)
    print("✅ ANÁLISIS COMPLETADO")
    print("="*80)
    print()
    print("Notas:")
    print("- CLR es el ÚNICO método con certificados de calidad por instancia")
    print("- Los gaps reportados garantizan proximidad al óptimo")
    print("- Tiempo de SDP es dominante pero amortizable con caching")
    print()


if __name__ == '__main__':
    main()

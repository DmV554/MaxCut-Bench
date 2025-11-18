"""
Test de pGW vs GW
"""
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from utils.baseline_solvers import probabilistic_gw, goemans_williamson
from core.quality_certificate import compute_quality_certificate, print_quality_certificate


def test_pgw_improvement():
    """
    Verificar que pGW(K=50) mejora sobre GW(K=1).
    """
    print("\n" + "="*60)
    print("TEST: pGW vs GW en grafos de BA_20")
    print("="*60)
    
    from scipy.sparse import load_npz
    import glob
    
    # Obtener grafos de BA_20
    graph_files = sorted(glob.glob('data/training/BA_20/*.npz'))[:10]  # Primeros 10
    
    if len(graph_files) == 0:
        print("⚠️  No se encontraron grafos en data/training/BA_20/")
        return True
    
    print(f"\nEvaluando en {len(graph_files)} grafos...")
    
    gw_cuts = []
    pgw_cuts = []
    gaps_gw = []
    gaps_pgw = []
    
    for i, graph_file in enumerate(graph_files):
        adjacency = load_npz(graph_file).toarray()
        
        # GW (K=1)
        gw_result = goemans_williamson(adjacency)
        gw_cuts.append(gw_result['cut'])
        gaps_gw.append((gw_result['Z_SDP'] - gw_result['cut']) / gw_result['Z_SDP'])
        
        # pGW (K=50)
        pgw_result = probabilistic_gw(adjacency, K=50)
        pgw_cuts.append(pgw_result['cut'])
        gaps_pgw.append((pgw_result['Z_SDP'] - pgw_result['cut']) / pgw_result['Z_SDP'])
        
        improvement = ((pgw_result['cut'] / gw_result['cut']) - 1) * 100
        
        print(f"Grafo {i+1:2d}: GW={gw_result['cut']:6.2f}, "
              f"pGW={pgw_result['cut']:6.2f}, "
              f"mejora={improvement:5.1f}%")
    
    print("\n" + "-"*60)
    print("RESUMEN:")
    print("-"*60)
    print(f"Media GW:     {np.mean(gw_cuts):.2f}")
    print(f"Media pGW:    {np.mean(pgw_cuts):.2f}")
    print(f"Mejora:       {((np.mean(pgw_cuts)/np.mean(gw_cuts))-1)*100:.1f}%")
    print(f"\nGap medio GW:  {np.mean(gaps_gw)*100:.2f}%")
    print(f"Gap medio pGW: {np.mean(gaps_pgw)*100:.2f}%")
    
    # Verificar que pGW mejora
    assert np.mean(pgw_cuts) > np.mean(gw_cuts), "pGW debe mejorar sobre GW"
    
    print("\n✅ TEST PASADO: pGW mejora sobre GW")
    
    # Mostrar certificado de ejemplo
    print("\n" + "="*60)
    print("EJEMPLO DE CERTIFICADO DE CALIDAD")
    print("="*60)
    cert = compute_quality_certificate(
        cut_value=pgw_cuts[0],
        Z_SDP=pgw_result['Z_SDP'],
        gw_baseline=gw_cuts[0]
    )
    print_quality_certificate(cert)
    
    return True


if __name__ == '__main__':
    try:
        test_pgw_improvement()
        print("\n✅ pGW baseline validado correctamente")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

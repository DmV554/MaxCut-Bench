"""
Sistema de Certificación de Calidad para MaxCut
"""
import numpy as np
from typing import Dict, Optional


def compute_quality_certificate(cut_value: float,
                               Z_SDP: float,
                               Z_optimal: Optional[float] = None,
                               gw_baseline: Optional[float] = None) -> Dict:
    """
    Calcula métricas de certificación de calidad.
    
    Esta es la CONTRIBUCIÓN ÚNICA de CLR: certificar calidad post-hoc.
    
    Args:
        cut_value: Valor del corte encontrado
        Z_SDP: Valor óptimo de la relajación SDP (cota superior)
        Z_optimal: Valor óptimo verdadero (si es conocido)
        gw_baseline: Valor del baseline GW (para comparación)
        
    Returns:
        {
            'cut': float,
            'Z_SDP': float,
            'gap_absolute': float,          # Z_SDP - cut
            'gap_relative': float,          # (Z_SDP - cut) / Z_SDP
            'approx_ratio': float,          # cut / Z_SDP
            'certification_level': str,     # 'excellent' | 'good' | 'acceptable' | 'poor'
            'gap_to_optimal': float | None,
            'improvement_over_gw': float | None
        }
    """
    # Verificar que cut <= Z_SDP (con tolerancia numérica)
    if cut_value > Z_SDP + 1e-6:
        raise ValueError(f"Cut ({cut_value:.4f}) excede cota superior SDP ({Z_SDP:.4f}). "
                        "Hay un error en la implementación.")
    
    gap_absolute = Z_SDP - cut_value
    gap_relative = gap_absolute / Z_SDP if Z_SDP > 0 else float('inf')
    approx_ratio = cut_value / Z_SDP if Z_SDP > 0 else 0.0
    
    # Clasificar calidad del certificado
    if gap_relative <= 0.05:
        certification_level = 'excellent'  # ≤5% gap
    elif gap_relative <= 0.10:
        certification_level = 'good'       # 5-10% gap
    elif gap_relative <= 0.20:
        certification_level = 'acceptable' # 10-20% gap
    else:
        certification_level = 'poor'       # >20% gap
    
    result = {
        'cut': cut_value,
        'Z_SDP': Z_SDP,
        'gap_absolute': gap_absolute,
        'gap_relative': gap_relative,
        'approx_ratio': approx_ratio,
        'certification_level': certification_level
    }
    
    # Gap respecto al óptimo (si es conocido)
    if Z_optimal is not None:
        gap_to_optimal = Z_optimal - cut_value
        gap_to_optimal_rel = gap_to_optimal / Z_optimal if Z_optimal > 0 else float('inf')
        result['gap_to_optimal'] = gap_to_optimal
        result['gap_to_optimal_relative'] = gap_to_optimal_rel
    else:
        result['gap_to_optimal'] = None
        result['gap_to_optimal_relative'] = None
    
    # Mejora sobre GW baseline
    if gw_baseline is not None and gw_baseline > 0:
        improvement = (cut_value - gw_baseline) / gw_baseline
        result['improvement_over_gw'] = improvement
    else:
        result['improvement_over_gw'] = None
    
    return result


def print_quality_certificate(cert: Dict, verbose: bool = True):
    """Pretty print del certificado de calidad."""
    if not verbose:
        return
    
    print("=" * 60)
    print("CERTIFICADO DE CALIDAD CLR")
    print("=" * 60)
    print(f"Corte encontrado:     {cert['cut']:.4f}")
    print(f"Cota superior (SDP):  {cert['Z_SDP']:.4f}")
    print(f"Gap absoluto:         {cert['gap_absolute']:.4f}")
    print(f"Gap relativo:         {cert['gap_relative']*100:.2f}%")
    print(f"Approx ratio:         {cert['approx_ratio']:.4f}")
    print(f"Nivel de cert.:       {cert['certification_level'].upper()}")
    
    if cert['gap_to_optimal'] is not None:
        print(f"Gap al óptimo:        {cert['gap_to_optimal_relative']*100:.2f}%")
    
    if cert['improvement_over_gw'] is not None:
        print(f"Mejora sobre GW:      {cert['improvement_over_gw']*100:.2f}%")
    
    print("=" * 60)

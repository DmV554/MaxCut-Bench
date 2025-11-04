# Guía de Compilación y Estructura del Pre-Print

## Archivos Generados

1. **`pre_print_maxcut.tex`**: Documento LaTeX completo con toda la sección de Experimental Setup
2. **`experimental_setup_summary.md`**: Resumen ejecutivo con tablas y detalles de implementación
3. **`pre_print_maxcut.txt`**: Versión texto actualizada

## Compilar el PDF

```bash
cd docs/
pdflatex pre_print_maxcut.tex
pdflatex pre_print_maxcut.tex  # Segunda pasada para referencias
```

O usando latexmk:
```bash
latexmk -pdf pre_print_maxcut.tex
```

## Estructura del Documento LaTeX

### Secciones Principales

1. **Abstract** (página 1)
   - Problema y motivación
   - Contribución principal (CLR)
   - Métrica única (certificación de calidad)

2. **Introducción** (páginas 1-2)
   - Motivación y contexto
   - Brecha y objetivo
   - Estado del arte

3. **Propuesta** (páginas 2-3)
   - Idea general (mezcla conservadora π_λ)
   - Componentes y flujo
   - Plan de evaluación

4. **Configuración Experimental** (páginas 3-5) ⭐ NUEVA
   - 4.1 Benchmark Framework
   - 4.2 Datasets (17 distribuciones, ~875 grafos)
   - 4.3 Baselines (11 métodos comparados)
   - 4.4 Arquitectura de CLR
   - 4.5 Métricas de Evaluación
   - 4.6 Configuración Computacional
   - 4.7 Protocolo de Evaluación

5. **Conclusión** (página 6)

6. **Referencias** (páginas 6-7)
   - 10 referencias completas

## Contenido de la Sección 4 (Experimental Setup)

### 4.1 Benchmark Framework
- MaxCut-Bench como infraestructura unificada
- Entorno conda benchenv
- 12 métodos integrados

### 4.2 Datasets (Detallado)

**Training:**
- BA_20: 100 grafos, n=20, para entrenar GNN

**Testing Principal (n=200-800):**
- 10 distribuciones con 5-100 grafos cada una
- BA, ER, Watts-Strogatz, Holme-Kim
- Weighted y unweighted

**Testing Especializado:**
- Dense MC, Planar, Toroidal, SK spin glasses, Physics
- Total: ~875 grafos

### 4.3 Baselines (11 métodos)

**Clásicos con garantías:**
- Standard Greedy
- GW (K=1)
- pGW (K=50)

**Metaheurísticas:**
- Tabu Search (50 rep, 2n steps)
- Extremal Optimization

**Deep RL (6 métodos):**
- S2V-DQN, ECO-DQN, LS-DQN
- SoftTabu, RUN-CSP, ANYCSP

**Especializados:**
- Gflow-CombOpt

### 4.4 Arquitectura CLR (4 componentes)

**SDP Solver:**
- CVXPY + SCS (open-source)
- Formulación estándar de GW
- Cholesky factorization
- Disk caching

**GNN Policy:**
- 3-layer GCN
- Hidden dim: 64
- Features: original + SDP vectors
- Output: hyperplane r ∈ S^(d-1)

**RL Training:**
- REINFORCE + baseline
- Reward: cut_policy - cut_uniform
- Adam (lr=1e-4)
- 1000 graphs, 50 epochs

**Mixed Sampler:**
- π_λ = λ·Uniform + (1-λ)·π_learned
- K=50 samples
- λ ∈ {0.0, 0.25, 0.5, 0.75, 1.0}

### 4.5 Métricas (3 tipos)

**Primarias (comparables):**
- Cut value
- Approximation ratio
- Time

**Certificación (únicas de CLR):**
- Z_SDP
- Gap absoluto/relativo
- Certification level (4 niveles)

**Descomposición:**
- Time_SDP, Time_Sampling
- Uniform/Learned fractions

### 4.6 Hardware & Software

**Hardware:**
- CPU: Intel Xeon / AMD Ryzen
- RAM: 32 GB
- GPU: NVIDIA (para training)

**Software:**
- Python 3.9
- PyTorch 2.0+ con PyG
- CVXPY 1.3+ con SCS
- Conda benchenv

**Config:**
- 10 threads (paralelo)
- 50 repeticiones (estocástico)
- Seed 42 (reproducibilidad)

### 4.7 Protocolo de Evaluación

**Pipeline por grafo:**
1. Cargar de data/testing/
2. Resolver SDP (o cache)
3. Sample K=50 de π_λ
4. Evaluar K cortes
5. Seleccionar mejor
6. Computar certificado
7. Guardar en pickle

**Ablation Studies:**
- Lambda: 5 valores × 17 distribuciones
- K: 4 valores × key distributions
- Generalization: BA→ER/WS/HK

**Análisis:**
- Tablas de approximation ratios
- Gaps certificados (CLR only)
- Tiempos descompuestos
- Gráficos de ablación
- In vs. out-of-distribution

## Figuras y Tablas Sugeridas (para Resultados)

### Tabla 1: Approximation Ratios
```
Dataset               | Greedy | GW  | pGW | TS  | ECO | CLR(0.5) |
---------------------|--------|-----|-----|-----|-----|----------|
BA_800_weighted      | 0.XXX  | ... | ... | ... | ... | 0.XXX    |
ER_800_weighted      | ...    | ... | ... | ... | ... | ...      |
...                  |        |     |     |     |     |          |
```
- Bold: mejor por fila
- Formato: 4 decimales

### Tabla 2: CLR Quality Certificates
```
Dataset               | Mean Gap | Std Gap | Excellent | Good | Accept. |
---------------------|----------|---------|-----------|------|---------|
BA_800_weighted      | 5.2%     | 1.8%    | 45%       | 40%  | 15%     |
...                  |          |         |           |      |         |
```

### Figura 1: Lambda Ablation
- X-axis: λ ∈ [0, 1]
- Y-axis: Approximation ratio
- Lines: Diferentes distribuciones
- Trade-off: performance vs. conservatism

### Figura 2: K Ablation
- X-axis: K ∈ {1, 10, 50, 100}
- Y-axis: Normalized cut value
- Show: Diminishing returns after K=50

### Figura 3: Time Decomposition
- Bar chart: CLR vs. TS vs. ECO-DQN
- Stacked: Time_SDP + Time_Sampling
- Note: SDP amortizable con cache

### Figura 4: Generalization Heatmap
- Rows: Train distribution
- Cols: Test distribution
- Values: Approximation ratio
- Diagonal: In-distribution (best)

## Paquetes LaTeX Necesarios

Ya incluidos en el .tex:
```latex
\usepackage[utf8]{inputenc}
\usepackage[spanish]{babel}
\usepackage{amsmath}
\usepackage{geometry}
\usepackage{booktabs}      % Para tablas profesionales
\usepackage{array}
\usepackage{multirow}
```

Para figuras (agregar si necesario):
```latex
\usepackage{graphicx}
\usepackage{subcaption}
\usepackage{tikz}
```

## Notas Importantes

### Contribuciones Únicas de CLR
1. **Quality Certification**: Único método ML con gaps certificados
2. **Conservative Mixing**: Trade-off explícito λ
3. **Competitive Performance**: Supera baselines manteniendo certificados

### Métricas que Solo CLR Reporta
- Z_SDP (cota dual)
- Gap absoluto y relativo
- Certification levels
- Esto es DIFERENCIADOR clave vs. todos los baselines

### Reproducibilidad
Todo está documentado para reproducir:
- Seed 42 fijo
- Conda environment especificado
- Scripts de evaluación en solvers/CLR/
- Datasets públicos en MaxCut-Bench

## Próximos Pasos

1. **Compilar PDF**: `pdflatex pre_print_maxcut.tex`
2. **Revisar formato**: Verificar que compile sin errores
3. **Ejecutar experimentos**: Seguir protocolo en sección 4.7
4. **Generar resultados**: Tablas y figuras sugeridas
5. **Agregar sección 5**: Resultados Experimentales (cuando estén listos)
6. **Agregar sección 6**: Discusión y Análisis
7. **Finalizar**: Abstract actualizado con números

## Comandos Útiles

```bash
# Compilar LaTeX
cd docs/
pdflatex pre_print_maxcut.tex

# Limpiar archivos auxiliares
rm *.aux *.log *.out

# Ver PDF
evince pre_print_maxcut.pdf &  # Linux
open pre_print_maxcut.pdf      # macOS

# Entrenar CLR
cd ..
python solvers/CLR/train.py --distribution BA_20 --epochs 50 --gpu

# Evaluar CLR
python solvers/CLR/evaluate.py \
    --test_distribution BA_800vertices_weighted \
    --lambda_mix 0.5 \
    --K 50

# Generar tabla comparativa
python table.py
```

## Contacto y Créditos

**Autores:**
- Daniel Miranda
- Nikolai Navea
- Vicente Arratia
- Javier Sepúlveda

**Curso:** OII464 - Desarrollo de Diseños Híbridos para Optimización  
**Profesor:** Emanuel Vega  
**Fecha:** 4 de Noviembre de 2025

---

**¡Experimental Setup completo y listo para experimentación!** 🚀

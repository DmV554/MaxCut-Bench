# Experimental Setup - Resumen Ejecutivo para CLR

## 📋 Quick Reference

### Datasets de Evaluación

| Distribución | # Grafos | Tamaño (n) | Tipo | Propósito |
|-------------|----------|------------|------|-----------|
| BA_20 | 100 | 20 | Training | Entrenar GNN policy |
| BA_200vertices_weighted | 100 | 200 | Test | Evaluación principal |
| BA_800vertices_unweighted | 100 | 800 | Test | Escalabilidad |
| BA_800vertices_weighted | 100 | 800 | Test | Evaluación principal |
| ER_200vertices_weighted | 100 | 200 | Test | Evaluación principal |
| ER_800vertices_unweighted | 5 | 800 | Test | Escalabilidad |
| ER_800vertices_weighted | 5 | 800 | Test | Escalabilidad |
| WattsStrogatz_800 (2x) | 200 | 800 | Test | Generalización |
| HomleKim_800 (2x) | 200 | 800 | Test | Generalización |
| Physics | 10 | var | Test | Real-world |
| SK_spin_70 | 40 | 100 | Test | Spin glasses |
| Planar (2x) | 8 | 800 | Test | Especializado |
| **TOTAL** | **~875** | **20-800** | - | - |

### Baselines Comparados

#### Métodos Clásicos (con garantías)
- **Standard Greedy**: Construcción greedy simple
- **GW (K=1)**: Goemans-Williamson estándar
- **pGW (K=50)**: Probabilistic GW (best-of-K uniforme)

#### Metaheurísticas
- **Tabu Search**: 50 repeticiones, 2n pasos, 10 threads
- **Extremal Optimization**: 50 repeticiones, 2n pasos

#### Deep RL Methods
- **S2V-DQN**: Structure2Vec con Q-learning
- **ECO-DQN**: Equilibrium-based CO
- **LS-DQN**: Local search guiada por DQN
- **SoftTabu**: ECO + Tabu relaxation
- **RUN-CSP**: Recursive update network
- **ANYCSP**: Adaptive CSP solver
- **Gflow-CombOpt**: Generative flow networks

### Configuración de CLR

#### Componente SDP
```python
Solver: CVXPY + SCS (open-source)
Formulación: max 1/4 Σ w_ij(1 - X_ij) s.t. diag(X)=1, X⪰0
Factorización: Cholesky para vectores {x_i}
Caching: Disco (evita re-cómputos)
```

#### GNN Policy
```python
Arquitectura: 3-layer GCN
Node features: original + SDP vectors (dim ≈ n+d)
Hidden dim: 64
Pooling: Global mean
Output: Hiperplano r ∈ S^(d-1)
Dropout: 0.1
Entropy coef: 0.01
```

#### RL Training
```python
Algoritmo: REINFORCE + uniform baseline
Reward: cut_policy - cut_uniform
Optimizer: Adam (lr=1e-4)
Training data: 1000 grafos/distribución
Epochs: 50 (early stopping)
Train/Val split: 80/20
Validation: cada 5 epochs
```

#### Mixed Sampler
```python
Distribución: π_λ = λ·Uniform + (1-λ)·π_learned
K: 50 samples (best-of-K)
Lambda values: {0.0, 0.25, 0.5, 0.75, 1.0}
Default: λ=0.5 (conservador)
```

### Métricas de Evaluación

#### Métricas Estándar (comparables con baselines)
- **Cut Value**: Valor objetivo
- **Approximation Ratio**: cut/OPT
- **Time**: Tiempo total (SDP + sampling)

#### Métricas Únicas de CLR (Certificación)
- **Z_SDP**: Cota superior certificable
- **Gap Absoluto**: Z_SDP - cut
- **Gap Relativo**: (Z_SDP - cut)/Z_SDP × 100%
- **Certification Level**: 
  - EXCELLENT: <5%
  - GOOD: 5-10%
  - ACCEPTABLE: 10-15%
  - POOR: >15%

#### Métricas de Ablación
- **Time_SDP**: Tiempo de relajación SDP
- **Time_Sampling**: Tiempo de muestreo
- **Uniform_Fraction**: % muestras uniformes
- **Learned_Fraction**: % muestras aprendidas

## 🔬 Protocolo de Evaluación

### Pipeline de Evaluación (por grafo)
```
1. Cargar grafo desde data/testing/{distribution}/
2. Resolver SDP (o cargar de cache)
   → Vectores {x_i}, valor Z_SDP
3. Samplear K=50 hiperplanos de π_λ
   → Mix uniforme/aprendido según λ
4. Evaluar K cortes candidatos
5. Seleccionar mejor corte
6. Computar certificado: gap = (Z_SDP - cut)/Z_SDP
7. Guardar resultados (formato pickle)
```

### Experimentos Principales

#### Exp 1: Comparación con Baselines
```bash
# Evaluar CLR (λ=0.5) en todas las distribuciones
for dist in BA_800vertices_weighted ER_800vertices_weighted ...; do
    python solvers/CLR/evaluate.py \
        --test_distribution $dist \
        --train_distribution BA_20 \
        --lambda_mix 0.5 \
        --K 50
done
```

**Output esperado:**
- Tablas de approximation ratio vs. baselines
- CLR único con gaps certificados
- Análisis de tiempo: SDP dominante pero amortizado con cache

#### Exp 2: Lambda Ablation
```bash
# Estudiar impacto de λ ∈ {0.0, 0.25, 0.5, 0.75, 1.0}
for lambda in 0.0 0.25 0.5 0.75 1.0; do
    python solvers/CLR/evaluate.py \
        --test_distribution BA_800vertices_weighted \
        --lambda_mix $lambda \
        --K 50
done
```

**Hipótesis:**
- λ=0.0: Mejor performance, peor gap (riesgoso)
- λ=1.0: Baseline pGW, gap consistente
- λ=0.5: Balance óptimo

#### Exp 3: K Ablation
```bash
# Estudiar impacto de K ∈ {1, 10, 50, 100}
for K in 1 10 50 100; do
    python solvers/CLR/evaluate.py \
        --test_distribution BA_800vertices_weighted \
        --lambda_mix 0.5 \
        --K $K
done
```

**Hipótesis:**
- K↑ → cut↑ (diminishing returns)
- K=50: Sweet spot (balance quality/tiempo)

#### Exp 4: Generalización
```bash
# Entrenar en BA_20, evaluar en otras distribuciones
distributions=(
    "BA_800vertices_weighted"
    "ER_800vertices_weighted"
    "WattsStrogatz_800vertices_weighted"
    "HomleKim_800vertices_weighted"
)

for dist in "${distributions[@]}"; do
    python solvers/CLR/evaluate.py \
        --test_distribution $dist \
        --train_distribution BA_20 \
        --lambda_mix 0.5 \
        --K 50
done
```

**Análisis:**
- In-distribution (BA→BA): Expected best
- Out-of-distribution (BA→ER/WS): Test robustness
- Componente uniforme (λ=0.5) ayuda generalización

## 📊 Visualizaciones Esperadas

### Para el Paper

1. **Tabla 1: Approximation Ratios**
   - Filas: Distribuciones
   - Columnas: Todos los baselines + CLR(λ=0.5)
   - Bold: Mejor resultado por distribución
   - Formato: 0.XXXX (4 decimales)

2. **Tabla 2: Gaps Certificados (CLR only)**
   - Columnas: Mean Gap, Std Gap, Certification Level
   - Highlight: CLR es único con esta métrica

3. **Figura 1: Lambda Ablation**
   - X: λ ∈ [0,1]
   - Y: Approximation ratio
   - Líneas por distribución
   - Show trade-off: performance vs. conservatism

4. **Figura 2: K Ablation**
   - X: K ∈ {1,10,50,100}
   - Y: Cut value (normalized)
   - Show: Diminishing returns

5. **Figura 3: Tiempo vs. Baselines**
   - Bar chart: CLR vs. TS vs. ECO-DQN
   - Descomponer: Time_SDP + Time_Sampling
   - Note: SDP cacheable, one-time cost

6. **Figura 4: Generalización**
   - Heatmap: Train dist (rows) × Test dist (cols)
   - Values: Approximation ratio
   - Diagonal: In-distribution
   - Off-diagonal: Transferability

## 🔑 Contribuciones Clave para Destacar

### Novedad 1: Quality Certification
> "CLR es el primer método basado en ML que proporciona certificados de calidad post-hoc por instancia, usando el gap dual de la relajación SDP."

**Evidencia:**
- Gap relativo promedio: ~X% (esperado <10%)
- XX% de instancias con certificación EXCELLENT
- Ningún baseline de ML reporta esta métrica

### Novedad 2: Conservative Mixing
> "El parámetro λ permite un trade-off explícito entre performance empírico y conservatismo teórico."

**Evidencia:**
- λ=0.5 supera a baselines manteniendo gaps <Y%
- Robust a generalización out-of-distribution

### Novedad 3: Competitive Performance
> "CLR(λ=0.5) supera a Tabu Search en Z distribuciones y iguala o mejora a métodos de DRL state-of-the-art."

**Evidencia:**
- Mean approximation ratio: CLR > TS en BA, ER
- CLR ≈ ECO-DQN pero con certificados

## 🛠️ Detalles de Implementación

### Formato de Resultados
```python
# Output: results/{test_distribution}/CLR (pickle)
df_results.columns = [
    'filename',           # Graph filename
    'cut',                # Best cut value
    'time',               # Total time (SDP + sampling)
    'time_sdp',           # SDP solve time
    'time_sampling',      # Sampling time
    'Z_SDP',              # SDP dual bound
    'gap_absolute',       # Z_SDP - cut
    'gap_relative',       # (Z_SDP - cut)/Z_SDP
    'approx_ratio',       # cut/Z_SDP
    'certification_level',# EXCELLENT/GOOD/ACCEPTABLE/POOR
    'lambda_mix',         # Lambda parameter
    'K',                  # Number of samples
    'uniform_fraction',   # % uniform samples
    'learned_fraction',   # % learned samples
    'mean_cut',           # Mean of K cuts
    'std_cut',            # Std of K cuts
]
```

### Reproducibilidad
```bash
# Setup
conda activate benchenv
cd MaxCut-Bench

# Train policy
python solvers/CLR/train.py \
    --distribution BA_20 \
    --num_graphs 1000 \
    --epochs 50 \
    --gpu \
    --seed 42

# Evaluate
python solvers/CLR/evaluate.py \
    --test_distribution BA_800vertices_weighted \
    --train_distribution BA_20 \
    --lambda_mix 0.5 \
    --K 50 \
    --seed 42

# Generate tables
python table.py  # MaxCut-Bench unified table
```

## 📝 Checklist Pre-Submission

- [ ] Entrenar modelos en BA_20, ER_200, WS_800
- [ ] Evaluar en todas las 17 distribuciones
- [ ] Completar lambda ablation (5 valores × 17 distribuciones)
- [ ] Completar K ablation (4 valores × key distributions)
- [ ] Generar Tabla 1 (approximation ratios)
- [ ] Generar Tabla 2 (gaps certificados)
- [ ] Crear Figuras 1-4
- [ ] Verificar reproducibilidad con seed=42
- [ ] Comparar tiempos con/sin cache SDP
- [ ] Análisis de generalización in/out-of-distribution
- [ ] Escribir sección de análisis de resultados
- [ ] Code/data release en GitHub

## 🎯 Mensaje Clave

**CLR reconcilia el dilema performance vs. garantías:**
- Supera a baselines clásicos (GW, Tabu Search) en performance empírico
- Único método de ML con certificados de calidad por instancia
- Parámetro λ permite control explícito del trade-off
- Robusto a generalización out-of-distribution

---

**Nota**: Este documento es un resumen ejecutivo. Ver `pre_print_maxcut.txt` sección 4 para detalles completos del Experimental Setup formal.

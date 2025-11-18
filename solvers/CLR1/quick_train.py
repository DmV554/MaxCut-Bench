"""
Quick training script para validar CLR en GPU pequeña (3GB).

Este script:
- Genera solo 100 grafos pequeños (n=15)
- Usa arquitectura ligera (32 dims, 2 capas)
- Training rápido (20 epochs)
- Valida que todos los componentes funcionen

Ideal para: Testing rápido en 1060 3GB
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import time

print("="*60)
print("CLR QUICK TRAINING TEST")
print("="*60)

# Check PyTorch availability
try:
    import torch
    print(f"✅ PyTorch: {torch.__version__}")
    print(f"   CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
except ImportError:
    print("❌ PyTorch no instalado")
    print("   Instalar: pip install torch")
    sys.exit(1)

# Check PyTorch Geometric
try:
    import torch_geometric
    print(f"✅ PyTorch Geometric: {torch_geometric.__version__}")
except ImportError:
    print("❌ PyTorch Geometric no instalado")
    print("   Instalar: pip install torch-geometric")
    sys.exit(1)

# Check NetworkX
try:
    import networkx as nx
    print(f"✅ NetworkX: {nx.__version__}")
except ImportError:
    print("❌ NetworkX no instalado")
    print("   Instalar: pip install networkx")
    sys.exit(1)

print()

# Import CLR components
from models.gnn_policy import GNNPolicy, create_gnn_policy
from models.rl_trainer import RLTrainer, prepare_training_data
from core.sdp_solver import SDPSolver

print("✅ Todos los imports correctos\n")

# Configuration para GPU pequeña
CONFIG = {
    'num_graphs': 100,         # Pocos grafos
    'n': 15,                   # Grafos pequeños (n=15)
    'm': 2,                    # BA parameter
    'hidden_dim': 32,          # Red pequeña
    'num_layers': 2,           # 2 capas en vez de 3
    'epochs': 20,              # Pocas epochs
    'lr': 1e-3,                # Learning rate más alto para convergencia rápida
    'baseline_K': 5,           # Menos samples
    'policy_K': 5,             # Menos samples
    'batch_size': 1,           # No batching (grafos uno por uno)
}

print("Configuración:")
for key, value in CONFIG.items():
    print(f"  {key}: {value}")
print()

# 1. Generar grafos pequeños
print("="*60)
print("1. GENERANDO GRAFOS SINTÉTICOS")
print("="*60)

graphs = []
for i in range(CONFIG['num_graphs']):
    G = nx.barabasi_albert_graph(CONFIG['n'], CONFIG['m'], seed=i)
    adjacency = nx.to_scipy_sparse_array(G, format='csr', dtype=float)
    adjacency_dense = adjacency.toarray()
    
    graph_data = {
        'adjacency_matrix': adjacency_dense,
        'distribution': 'BA_15',
        'n': CONFIG['n'],
        'id': i
    }
    graphs.append(graph_data)

print(f"✅ Generados {len(graphs)} grafos BA({CONFIG['n']}, {CONFIG['m']})")

# Split train/val
num_train = int(len(graphs) * 0.8)
train_graphs = graphs[:num_train]
val_graphs = graphs[num_train:]

print(f"   Train: {len(train_graphs)} grafos")
print(f"   Val: {len(val_graphs)} grafos\n")

# 2. Resolver SDPs
print("="*60)
print("2. RESOLVIENDO SDPs")
print("="*60)

sdp_solver = SDPSolver(solver='SCS', cache_dir='./cache_sdp_test', verbose=False)

start_time = time.time()
train_data = prepare_training_data(train_graphs, sdp_solver)
val_data = prepare_training_data(val_graphs, sdp_solver)
sdp_time = time.time() - start_time

print(f"✅ SDPs resueltos en {sdp_time:.2f}s")
print(f"   Train: {len(train_data)} grafos")
print(f"   Val: {len(val_data)} grafos\n")

# 3. Crear GNN (pequeña)
print("="*60)
print("3. CREANDO GNN POLICY")
print("="*60)

# Inferir dimensión de vectores SDP
_, first_vectors, _ = train_data[0]
sdp_dim = first_vectors.shape[1]

print(f"   Dimensión de vectores SDP: {sdp_dim}")

gnn_policy = create_gnn_policy(
    sdp_vector_dim=sdp_dim,
    hidden_dim=CONFIG['hidden_dim'],
    num_layers=CONFIG['num_layers'],
    dropout=0.1
)

num_params = sum(p.numel() for p in gnn_policy.parameters())
print(f"✅ GNN creada: {num_params:,} parámetros")

# Check GPU memory
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    print(f"   GPU memory allocated: {torch.cuda.memory_allocated(0) / 1e6:.2f} MB\n")
else:
    print("   Running on CPU\n")

# 4. Crear trainer
print("="*60)
print("4. CREANDO RL TRAINER")
print("="*60)

trainer = RLTrainer(
    gnn_policy=gnn_policy,
    sdp_solver=sdp_solver,
    learning_rate=CONFIG['lr'],
    baseline_K=CONFIG['baseline_K'],
    policy_K=CONFIG['policy_K'],
    entropy_coef=0.01,
    use_baseline_value=True,
    device='cuda' if torch.cuda.is_available() else 'cpu'
)

print("✅ Trainer listo\n")

# 5. Training loop
print("="*60)
print("5. ENTRENAMIENTO")
print("="*60)
print()

best_val_improvement = -float('inf')
training_history = {'train': [], 'val': []}

for epoch in range(CONFIG['epochs']):
    print(f"Epoch {epoch+1}/{CONFIG['epochs']}")
    print("-" * 40)
    
    # Train
    start_epoch = time.time()
    train_stats = trainer.train_epoch(train_data, verbose=False)
    epoch_time = time.time() - start_epoch
    training_history['train'].append(train_stats)
    
    print(f"Train ({epoch_time:.2f}s):")
    print(f"  Cut policy: {train_stats['mean_cut_policy']:.2f}")
    print(f"  Cut uniform: {train_stats['mean_cut_uniform']:.2f}")
    print(f"  Improvement: {train_stats['improvement']:.2f}%")
    print(f"  Gap policy: {train_stats['gap_policy']:.2f}%")
    print(f"  Loss: {train_stats['policy_loss']:.4f}")
    
    # Validate every 5 epochs
    if (epoch + 1) % 5 == 0:
        print("  Validando...")
        val_stats = trainer.validate(val_data)
        training_history['val'].append(val_stats)
        
        print(f"Val:")
        print(f"  Cut policy: {val_stats['mean_cut_policy']:.2f}")
        print(f"  Cut uniform: {val_stats['mean_cut_uniform']:.2f}")
        print(f"  Improvement: {val_stats['improvement']:.2f}%")
        print(f"  Gap policy: {val_stats['gap_policy']:.2f}%")
        
        if val_stats['improvement'] > best_val_improvement:
            best_val_improvement = val_stats['improvement']
            print(f"  ✅ Mejor modelo! (improvement={best_val_improvement:.2f}%)")
    
    print()
    
    # Clear GPU cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# 6. Summary
print("="*60)
print("RESUMEN")
print("="*60)
print(f"Epochs completados: {CONFIG['epochs']}")
print(f"Mejor improvement (val): {best_val_improvement:.2f}%")
print()

# Plot training curves
try:
    import matplotlib.pyplot as plt
    
    train_improvements = [epoch['improvement'] for epoch in training_history['train']]
    epochs = list(range(1, len(train_improvements) + 1))
    
    plt.figure(figsize=(10, 5))
    
    # Plot 1: Improvement
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_improvements, 'b-', label='Train')
    if training_history['val']:
        val_epochs = [i*5 for i in range(1, len(training_history['val'])+1)]
        val_improvements = [epoch['improvement'] for epoch in training_history['val']]
        plt.plot(val_epochs, val_improvements, 'r-o', label='Val')
    plt.xlabel('Epoch')
    plt.ylabel('Improvement over Uniform (%)')
    plt.title('CLR Training Progress')
    plt.legend()
    plt.grid(True)
    
    # Plot 2: Gap
    plt.subplot(1, 2, 2)
    train_gaps = [epoch['gap_policy'] for epoch in training_history['train']]
    train_gaps_uniform = [epoch['gap_uniform'] for epoch in training_history['train']]
    plt.plot(epochs, train_gaps, 'b-', label='Policy')
    plt.plot(epochs, train_gaps_uniform, 'r--', label='Uniform')
    plt.xlabel('Epoch')
    plt.ylabel('Gap to Z_SDP (%)')
    plt.title('Quality Gaps')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('solvers/CLR/quick_training_results.png', dpi=150)
    print("✅ Gráficos guardados en solvers/CLR/quick_training_results.png")
    
except ImportError:
    print("⚠️  matplotlib no disponible, saltando gráficos")

print()
print("="*60)
print("✅ ENTRENAMIENTO COMPLETADO EXITOSAMENTE")
print("="*60)
print()
print("Conclusión:")
if best_val_improvement > 0:
    print("✅ CLR está aprendiendo! La policy supera al baseline uniforme.")
else:
    print("⚠️  CLR aún no supera baseline (normal en training corto)")
    print("   Solución: Más epochs, más grafos, o mejor tuning")

print()
print("Próximos pasos:")
print("1. Training real: python solvers/CLR/train.py --distribution BA_20 --num_graphs 1000 --epochs 50")
print("2. Evaluation: python solvers/CLR/evaluate.py --test_distribution BA_200vertices_weighted")
print()

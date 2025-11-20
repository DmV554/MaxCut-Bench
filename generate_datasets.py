import networkx as nx
import numpy as np
import scipy.sparse as sp
import os
from tqdm import tqdm  # Opcional: para ver barra de progreso (pip install tqdm)

# --- CONFIGURACIÓN SEGÚN EL PAPER (Apéndice A.1) ---
CONFIG = {
    'ER_200': { 
        'type': 'ER',
        'n': 200,           # Tamaño para training según paper
        'p': 0.15,          # Probabilidad de arista
        'num_graphs': 4000  # Cantidad de grafos
    },
    'BA_200': { 
        'type': 'BA',
        'n': 200,           # Tamaño para training según paper
        'm': 4,             # Aristas por nuevo nodo
        'num_graphs': 4000  # Cantidad de grafos
    }
}

OUTPUT_DIR = "./data/training_generated"

def generate_dataset(config_name, params):
    print(f"🚀 Generando {params['num_graphs']} grafos para {config_name}...")
    
    # Crear carpeta si no existe
    save_dir = os.path.join(OUTPUT_DIR, config_name)
    os.makedirs(save_dir, exist_ok=True)
    
    for i in tqdm(range(params['num_graphs']), desc=config_name):
        # 1. Generar Topología
        if params['type'] == 'ER':
            # Erdős-Rényi
            g = nx.erdos_renyi_graph(n=params['n'], p=params['p'])
        elif params['type'] == 'BA':
            # Barabási-Albert
            g = nx.barabasi_albert_graph(n=params['n'], m=params['m'])
        
        # 2. Asignar Pesos (Weighted: {-1, 1})
        # El paper menciona w ∈ {0, ±1}. Como la matriz es dispersa, el 0 es implícito.
        # Asignamos aleatoriamente -1 o 1 a cada arista existente.
        for (u, v) in g.edges():
            weight = np.random.choice([-1, 1])
            g[u][v]['weight'] = weight
            
        # 3. Convertir a CSR Matrix (Formato del repositorio)
        # nx.to_scipy_sparse_array devuelve una matriz simétrica para grafos no dirigidos,
        # que es exactamente lo que queremos.
        adj_matrix = nx.to_scipy_sparse_array(g, format='csr', weight='weight')
        
        # Nota: Aseguramos que sea tipo float o int según prefieras. 
        # El ejemplo que pasaste tenía enteros, pero float es más seguro para ML.
        adj_matrix = adj_matrix.astype(np.float32) 

        # 4. Guardar en formato .npz comprimido
        # Esto crea automáticamente los archivos indices.npy, indptr.npy, data.npy, etc. dentro del zip
        file_path = os.path.join(save_dir, f"graph_{i}.npz")
        sp.save_npz(file_path, adj_matrix)

    print(f"✅ Completado: {save_dir}\n")

if __name__ == "__main__":
    # Generar ambos datasets
    generate_dataset('ER_200', CONFIG['ER_200'])
    generate_dataset('BA_200', CONFIG['BA_200'])
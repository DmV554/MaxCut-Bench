"""
Utility script para listar modelos preentrenados disponibles.

Usage:
    python list_pretrained_models.py
"""
import os
import pickle
import torch
from datetime import datetime

def list_pretrained_models(pretrained_dir='solvers/CLR/pretrained'):
    """Lista todos los modelos preentrenados disponibles."""
    
    if not os.path.exists(pretrained_dir):
        print(f"❌ Carpeta de modelos no encontrada: {pretrained_dir}")
        return
    
    distributions = [d for d in os.listdir(pretrained_dir) 
                     if os.path.isdir(os.path.join(pretrained_dir, d))]
    
    if not distributions:
        print(f"⚠️  No hay modelos preentrenados disponibles en {pretrained_dir}")
        print("\nPara entrenar un modelo, ejecuta:")
        print("  python solvers/CLR/train.py --distribution BA_20 --num_graphs 1000 --epochs 50")
        return
    
    print("="*80)
    print("MODELOS PREENTRENADOS DISPONIBLES")
    print("="*80 + "\n")
    
    for dist in sorted(distributions):
        dist_path = os.path.join(pretrained_dir, dist)
        best_model_path = os.path.join(dist_path, 'policy_best.pth')
        final_model_path = os.path.join(dist_path, 'policy_final.pth')
        history_path = os.path.join(dist_path, 'training_history.pkl')
        
        print(f"📦 {dist}")
        print("-" * 80)
        
        # Best model
        if os.path.exists(best_model_path):
            try:
                checkpoint = torch.load(best_model_path, map_location='cpu', weights_only=False)
                epoch = checkpoint.get('epoch', 'N/A')
                stats = checkpoint.get('stats', {})
                improvement = stats.get('improvement', 'N/A')
                
                # Get file modification time
                mod_time = os.path.getmtime(best_model_path)
                mod_date = datetime.fromtimestamp(mod_time).strftime('%Y-%m-%d %H:%M:%S')
                
                print(f"  ✅ policy_best.pth")
                print(f"     Epoch: {epoch}")
                print(f"     Improvement: {improvement:.2f}%" if isinstance(improvement, (int, float)) else f"     Improvement: {improvement}")
                print(f"     Fecha: {mod_date}")
            except Exception as e:
                print(f"  ⚠️  policy_best.pth (error al cargar: {e})")
        else:
            print(f"  ❌ policy_best.pth no encontrado")
        
        # Final model
        if os.path.exists(final_model_path):
            try:
                checkpoint = torch.load(final_model_path, map_location='cpu', weights_only=False)
                epoch = checkpoint.get('epoch', 'N/A')
                mod_time = os.path.getmtime(final_model_path)
                mod_date = datetime.fromtimestamp(mod_time).strftime('%Y-%m-%d %H:%M:%S')
                
                print(f"  ✅ policy_final.pth")
                print(f"     Epoch: {epoch}")
                print(f"     Fecha: {mod_date}")
            except Exception as e:
                print(f"  ⚠️  policy_final.pth (error al cargar: {e})")
        
        # Training history
        if os.path.exists(history_path):
            try:
                with open(history_path, 'rb') as f:
                    history = pickle.load(f)
                
                n_epochs = len(history.get('train', []))
                n_val = len(history.get('val', []))
                
                print(f"  ✅ training_history.pkl")
                print(f"     Epochs registrados: {n_epochs} train, {n_val} val")
            except Exception as e:
                print(f"  ⚠️  training_history.pkl (error al cargar: {e})")
        
        print()
    
    print("="*80)
    print(f"Total: {len(distributions)} distribuciones con modelos preentrenados")
    print("="*80)
    print("\nPara evaluar un modelo:")
    print("  python solvers/CLR/evaluate.py --test_distribution <TEST> --train_distribution <DIST>")
    print("\nPara entrenar un nuevo modelo:")
    print("  python solvers/CLR/train.py --distribution <DIST> --num_graphs 1000 --epochs 50")
    print("="*80)


if __name__ == '__main__':
    list_pretrained_models()

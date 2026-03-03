import os
import json
from evaluate import evaluate_model

# Dossier racine des modèles
MODELS_ROOT = "models_trained"
# Dataset à comparer
DATASET = "dataset_multi_mw"
# Liste des modèles à comparer
MODEL_NAMES = ["ODMR_CNN", "ODMR_CNN_Compact", "ODMR_CNN_Deep", "FrequencyAttention", "MWConfig_CNN", "HybridODMRPredictor"]

# Résultats
results = {}

for model_name in MODEL_NAMES:
    model_dir = os.path.join(MODELS_ROOT, DATASET, model_name.lower())
    if not os.path.exists(model_dir):
        print(f"[WARN] Model dir not found: {model_dir}")
        continue
    metrics = evaluate_model(model_name, model_dir=model_dir, dataset_dir=DATASET, show=False)
    results[model_name] = metrics

# Affichage MAE
print("\n====================== COMPARAISON DES MAE (A) =====================")
header = ["Model", "MAE_Ax", "MAE_Ay", "MAE_Az", "MAE_Mean"]
print("| {:<20} | {:>8} | {:>8} | {:>8} | {:>8} |".format(*header))
print("|" + "-"*22 + "|" + "-"*10 + "|" + "-"*10 + "|" + "-"*10 + "|" + "-"*10 + "|")
mae_table = []
for model_name, metrics in results.items():
    try:
        mae_ax = metrics['Ax']['MAE']
        mae_ay = metrics['Ay']['MAE']
        mae_az = metrics['Az']['MAE']
        mae_mean = round((mae_ax + mae_ay + mae_az) / 3, 4)
        mae_table.append((model_name, mae_ax, mae_ay, mae_az, mae_mean))
    except Exception as e:
        print(f"[WARN] Could not extract MAE for {model_name}: {e}")
        continue
# Tri par MAE moyen croissant
mae_table.sort(key=lambda x: x[4])
for row in mae_table:
    print("| {:<20} | {:>8.3f} | {:>8.3f} | {:>8.3f} | {:>8.4f} |".format(*row))

# Sauvegarde des résultats bruts
output_path = os.path.join(MODELS_ROOT, DATASET, "comparative_evaluation.json")
with open(output_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nComparative evaluation saved as {output_path}")

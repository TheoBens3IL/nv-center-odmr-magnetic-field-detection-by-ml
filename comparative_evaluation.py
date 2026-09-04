import os
import json
import torch
from evaluate import load_model, get_test_loader, evaluate_cnn, evaluate_regressor, evaluate_two_stage, compute_metrics

MODELS_ROOT = "models_trained"
DATASET = "dataset_multi_mw_2"
MODEL_NAMES = ["ODMR_CNN", "ODMR_CNN_Compact", "ODMR_CNN_Deep", "FrequencyAttention", "MWConfig_CNN", "AxisSplitRegressor"]

results = {}

device = 'cuda' if torch.cuda.is_available() else 'cpu'

for model_name in MODEL_NAMES:
    model_dir = os.path.join(MODELS_ROOT, DATASET, model_name.lower())
    if not os.path.exists(model_dir):
        print(f"[WARN] Model dir not found: {model_dir}")
        continue
    dataset_dir = os.path.join("datasets_pytorch", DATASET)
    test_loader, labels_mean, labels_std = get_test_loader(dataset_dir, model_name, batch_size=64)
    model = load_model(model_name, model_dir, dataset_dir, device=device)
    if model_name.lower() == 'zoneawareregressor':
        y_pred, y_true, _ = evaluate_regressor(model, test_loader, device=device)
    elif model_name.lower() == 'zoneawaretwostage' or model_name.lower() == 'zoneawaretwostage_joint':
        y_pred, y_true, _ = evaluate_two_stage(model, test_loader, device=device)
    else:
        y_pred, y_true = evaluate_cnn(model, test_loader, device=device)
    metrics = compute_metrics(y_pred, y_true, labels_mean, labels_std)
    results[model_name] = metrics

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

mae_table.sort(key=lambda x: x[4])
for row in mae_table:
    print("| {:<20} | {:>8.3f} | {:>8.3f} | {:>8.3f} | {:>8.3f} |".format(*row))

output_path = os.path.join(MODELS_ROOT, DATASET, "comparative_evaluation.json")
with open(output_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nComparative evaluation saved as {output_path}")

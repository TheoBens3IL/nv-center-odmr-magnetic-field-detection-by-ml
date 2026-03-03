"""
Compare Ridge regression performance across different MW configurations.
Tests each of the 10 MW configs individually and compares with all-config model.
"""

import subprocess
import re
import pandas as pd
from pathlib import Path

DATASET_DIR = "dataset_multi_mw"

def run_ridge_config(mw_config=None, dataset=DATASET_DIR):
    """Run Ridge training for a specific config and extract R² scores."""
    
    # Build command
    cmd = ['python', 'train_ridge.py', '--dataset_dir', dataset]
    if mw_config is not None:
        cmd.extend(['--mw_config', str(mw_config)])
    
    if mw_config is not None:
        print(f"Testing MW Config {mw_config}...")
    else:
        print(f"Testing ALL configs as one...")
    
    # Run command
    result = subprocess.run(cmd, capture_output=True, text=True)
    output = result.stdout
    
    # Extract R² scores from output
    r2_pattern = r'(Ax|Ay|Az|Ar|theta|phi):\s*([-\d.]+)'
    matches = re.findall(r2_pattern, output)
    
    # Create dictionary of R² scores
    r2_dict = {name: float(score) for name, score in matches}
    
    # Extract absolute MAE scores (real data)
    abs_mae_pattern = r'MAE \(absolute\): (\w+)=([\d\.eE+-]+)\w*, (\w+)=([\d\.eE+-]+)\w*, (\w+)=([\d\.eE+-]+)\w*'
    abs_mae_matches = re.findall(abs_mae_pattern, output)
    if abs_mae_matches:
        # Get test set MAE (last occurrence)
        test_abs_mae = abs_mae_matches[-1]
        # test_abs_mae is a tuple: (label1, val1, label2, val2, label3, val3)
        r2_dict['MAE_' + test_abs_mae[0]] = float(test_abs_mae[1])
        r2_dict['MAE_' + test_abs_mae[2]] = float(test_abs_mae[3])
        r2_dict['MAE_' + test_abs_mae[4]] = float(test_abs_mae[5])
    return r2_dict


def main():
    results = []
    
    # Test each individual config
    for i in range(10):
        r2_scores = run_ridge_config(mw_config=i)
        if r2_scores:
            r2_scores['Config'] = f'Config_{i}'
            results.append(r2_scores)
    
    # Test all configs combined
    r2_scores = run_ridge_config(mw_config=None)
    if r2_scores:
        r2_scores['Config'] = 'All_Configs'
        results.append(r2_scores)
    
    # Create DataFrame
    df = pd.DataFrame(results)
    df = df.set_index('Config')
    
    # Prepare output text
    output_file = 'models_ridge/ridge_configs_diagnostic.txt'
    output_lines = []
    
    output_lines.append("="*60)
    output_lines.append(f"COMPARISON RESULTS FOR THE DATASET: {DATASET_DIR}")
    output_lines.append("="*60)
    output_lines.append("")
    output_lines.append(df.to_string())
    output_lines.append("")
    output_lines.append("")
    
    # Find best config for each component
    output_lines.append("="*60)
    output_lines.append("BEST CONFIGURATION PER COMPONENT (R2)")
    output_lines.append("="*60)
    
    r2_cols = [col for col in df.columns if not col.startswith('NMAE_')]
    for col in r2_cols:
        best_config = df[col].idxmax()
        best_value = df[col].max()
        output_lines.append(f"{col:10s}: {best_config:15s} (R2 = {best_value:.4f})")
    
    output_lines.append("")
    output_lines.append("="*60)
    output_lines.append("BEST CONFIGURATION PER COMPONENT (MAE)")
    output_lines.append("="*60)
    
    mae_cols = [col for col in df.columns if col.startswith('MAE_')]
    for col in mae_cols:
        best_config = df[col].idxmin()  # Lower is better for MAE
        best_value = df[col].min()
        component = col.replace('MAE_', '')
        output_lines.append(f"{component:10s}: {best_config:15s} (MAE = {best_value:.4f})")
    
    # Save to file
    with open(output_file, 'w') as f:
        f.write('\n'.join(output_lines))
    
    print(f"\nRidge configuration comparison results saved to: {output_file}")
    print()


if __name__ == "__main__":
    try: 
        main()
    except Exception as e:
        print(e)
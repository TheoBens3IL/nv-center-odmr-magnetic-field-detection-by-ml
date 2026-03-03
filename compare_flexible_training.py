"""
Automated comparison of different component/config combinations.
Tests all relevant scenarios and saves summary.
"""

import subprocess
import re
import pandas as pd
from pathlib import Path
import time


def run_experiment(target_components, num_mw_configs):
    """Run training experiment and extract metrics."""
    
    components_str = " ".join(target_components)
    
    print(f"\n{'='*60}")
    print(f"Testing: {target_components} with {num_mw_configs} MW configs")
    print(f"{'='*60}")
    
    # Build command
    cmd = [
        'python', 'train_flexible.py',
        '--target_components', *target_components,
        '--num_mw_configs', str(num_mw_configs),
        '--patience', '20',
        '--epochs', '200'
    ]
    
    # Run
    start_time = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    duration = time.time() - start_time
    
    output = result.stdout
    
    # Extract metrics
    results = {
        'targets': '+'.join(target_components),
        'num_mw_configs': num_mw_configs,
        'duration_sec': round(duration, 1)
    }
    
    # Parse R² and MAE for each component
    for comp in ['Ax', 'Ay', 'Az']:
        if comp in target_components:
            # Extract MAE
            mae_pattern = rf'{comp}:.*?MAE:\s+([\d.]+)'
            mae_match = re.search(mae_pattern, output, re.DOTALL)
            if mae_match:
                results[f'{comp}_MAE'] = float(mae_match.group(1))
            # Extract R²
            r2_pattern = rf'{comp}:.*?R²:\s+([\-\d.]+)'
            r2_match = re.search(r2_pattern, output, re.DOTALL)
            if r2_match:
                results[f'{comp}_R2'] = float(r2_match.group(1))
        else:
            results[f'{comp}_MAE'] = None
            results[f'{comp}_R2'] = None
    
    # Extract final epoch
    epoch_pattern = r'Early stopping at epoch (\d+)'
    epoch_match = re.search(epoch_pattern, output)
    if epoch_match:
        results['final_epoch'] = int(epoch_match.group(1))
    else:
        results['final_epoch'] = 200
    
    return results


def main():
    print("="*60)
    print("COMPREHENSIVE COMPONENT/CONFIG COMPARISON")
    print("="*60)
    print()
    
    experiments = []
    
    # 1. Single component predictions
    print("\n### PART 1: Single Component Predictions ###")
    for comp in ['Ax', 'Ay', 'Az']:
        for num_configs in [10, 5, 3, 1]:
            result = run_experiment([comp], num_configs)
            experiments.append(result)
    
    # 2. Ay + Az together
    print("\n### PART 2: Ay+Az Joint Prediction ###")
    for num_configs in [10, 5, 3]:
        result = run_experiment(['Ay', 'Az'], num_configs)
        experiments.append(result)
    
    # 3. All components (baseline)
    print("\n### PART 3: Full Prediction (Baseline) ###")
    for num_configs in [10, 5]:
        result = run_experiment(['Ax', 'Ay', 'Az'], num_configs)
        experiments.append(result)
    
    # Create DataFrame
    df = pd.DataFrame(experiments)
    
    # Reorder columns
    cols = ['targets', 'num_mw_configs', 'final_epoch', 'duration_sec',
        'Ax_R2', 'Ax_MAE', 'Ay_R2', 'Ay_MAE', 'Az_R2', 'Az_MAE']
    df = df[cols]
    
    # Save results
    output_file = 'models_flexible/flexible_training_comparison.csv'
    df.to_csv(output_file, index=False)
    
    print(f"\n{'='*60}")
    print("SUMMARY RESULTS")
    print(f"{'='*60}\n")
    print(df.to_string(index=False))
    print(f"\n\nResults saved to: {output_file}")
    
    # Key findings
    print(f"\n{'='*60}")
    print("KEY FINDINGS")
    print(f"{'='*60}")
    
    # Best for each component
    for comp in ['Ax', 'Ay', 'Az']:
        df_comp = df[df[f'{comp}_MAE'].notna()]
        if len(df_comp) > 0:
            best_idx = df_comp[f'{comp}_MAE'].idxmin()
            best = df.loc[best_idx]
            print(f"\nBest {comp} prediction:")
            print(f"  Config: {best['targets']} with {best['num_mw_configs']} MW configs")
            print(f"  MAE: {best[f'{comp}_MAE']:.4f}")
            print(f"  R²: {best[f'{comp}_R2']:.4f}")
    
    # Compare MW config impact
    print(f"\n{'='*60}")
    print("MW CONFIG IMPACT (Ay prediction)")
    print(f"{'='*60}")
    ay_only = df[df['targets'] == 'Ay'].sort_values('num_mw_configs', ascending=False)
    if len(ay_only) > 0:
        for _, row in ay_only.iterrows():
            print(f"  {row['num_mw_configs']} configs: MAE={row['Ay_MAE']:.4f}, R²={row['Ay_R2']:.4f}")


if __name__ == "__main__":
    main()

import os
import sys
import subprocess
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import time

# Number of iterations
NUM_ITERATIONS = 10
TICKER = "AAPL"

def run_ensemble_analysis(iteration):
    """Run the normalized ensemble analysis script"""
    print(f"Running iteration {iteration+1}/{NUM_ITERATIONS}...")
    
    # Run the script
    cmd = f"python scripts/normalized_ensemble.py --ticker {TICKER}"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Error in iteration {iteration+1}: {result.stderr}")
        return None
    
    # Extract the last created CSV file from the output directory
    all_files = [os.path.join("outputs/normalized_ensemble", f) for f in os.listdir("outputs/normalized_ensemble") 
                if f.startswith(f"{TICKER}_normalized_ensemble") and f.endswith(".csv")]
    
    if not all_files:
        print(f"No output files found for iteration {iteration+1}")
        return None
    
    # Get the most recent file
    latest_file = max(all_files, key=os.path.getctime)
    
    # Parse the output to extract key metrics
    final_price = None
    predicted_change = None
    up_votes = 0
    down_votes = 0
    neutral_votes = 0
    
    for line in result.stdout.split('\n'):
        if "Final predicted price:" in line:
            parts = line.split('$')
            if len(parts) > 1:
                final_price = float(parts[1].split(' ')[0])
        elif "Predicted change:" in line:
            parts = line.split(':')
            if len(parts) > 1:
                predicted_change = float(parts[1].strip().replace('%', ''))
        elif "UP:" in line:
            parts = line.split('(')
            if len(parts) > 1:
                up_votes = float(parts[1].split('%')[0])
        elif "DOWN:" in line:
            parts = line.split('(')
            if len(parts) > 1:
                down_votes = float(parts[1].split('%')[0])
        elif "NEUTRAL:" in line:
            parts = line.split('(')
            if len(parts) > 1:
                neutral_votes = float(parts[1].split('%')[0])
    
    return {
        'iteration': iteration + 1,
        'file': latest_file,
        'final_price': final_price,
        'predicted_change': predicted_change,
        'up_votes': up_votes,
        'down_votes': down_votes,
        'neutral_votes': neutral_votes
    }

def analyze_results(results):
    """Analyze the results of multiple runs"""
    # Extract key metrics
    final_prices = [r['final_price'] for r in results if r is not None and r['final_price'] is not None]
    predicted_changes = [r['predicted_change'] for r in results if r is not None and r['predicted_change'] is not None]
    up_votes = [r['up_votes'] for r in results if r is not None and r['up_votes'] is not None]
    down_votes = [r['down_votes'] for r in results if r is not None and r['down_votes'] is not None]
    neutral_votes = [r['neutral_votes'] for r in results if r is not None and r['neutral_votes'] is not None]
    
    # Calculate statistics
    stats = {
        'final_price': {
            'mean': np.mean(final_prices) if final_prices else None,
            'median': np.median(final_prices) if final_prices else None,
            'std': np.std(final_prices) if final_prices else None,
            'min': min(final_prices) if final_prices else None,
            'max': max(final_prices) if final_prices else None
        },
        'predicted_change': {
            'mean': np.mean(predicted_changes) if predicted_changes else None,
            'median': np.median(predicted_changes) if predicted_changes else None,
            'std': np.std(predicted_changes) if predicted_changes else None,
            'min': min(predicted_changes) if predicted_changes else None,
            'max': max(predicted_changes) if predicted_changes else None
        },
        'votes': {
            'up_mean': np.mean(up_votes) if up_votes else None,
            'down_mean': np.mean(down_votes) if down_votes else None,
            'neutral_mean': np.mean(neutral_votes) if neutral_votes else None
        }
    }
    
    # Create plots
    create_plots(results, stats)
    
    return stats

def create_plots(results, stats):
    """Create visualizations of the results"""
    # Create output directory
    os.makedirs('outputs/multiple_runs', exist_ok=True)
    
    # Filter out None results
    valid_results = [r for r in results if r is not None and r['final_price'] is not None]
    
    if not valid_results:
        print("No valid results to plot")
        return
    
    # Timestamp for filenames
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Plot 1: Final prices across iterations
    plt.figure(figsize=(12, 6))
    iterations = [r['iteration'] for r in valid_results]
    final_prices = [r['final_price'] for r in valid_results]
    
    plt.plot(iterations, final_prices, 'o-', color='blue')
    plt.axhline(y=stats['final_price']['mean'], color='red', linestyle='--', label=f"Mean: ${stats['final_price']['mean']:.2f}")
    
    plt.title(f"{TICKER} Final Predicted Prices Across {NUM_ITERATIONS} Iterations", fontsize=16)
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Final Predicted Price ($)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f"outputs/multiple_runs/{TICKER}_final_prices_{timestamp}.png")
    plt.close()
    
    # Plot 2: Predicted changes across iterations
    plt.figure(figsize=(12, 6))
    predicted_changes = [r['predicted_change'] for r in valid_results]
    
    plt.plot(iterations, predicted_changes, 'o-', color='green')
    plt.axhline(y=stats['predicted_change']['mean'], color='red', linestyle='--', 
               label=f"Mean: {stats['predicted_change']['mean']:.2f}%")
    
    plt.title(f"{TICKER} Predicted Changes Across {NUM_ITERATIONS} Iterations", fontsize=16)
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Predicted Change (%)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f"outputs/multiple_runs/{TICKER}_predicted_changes_{timestamp}.png")
    plt.close()
    
    # Plot 3: Vote distribution
    plt.figure(figsize=(10, 6))
    labels = ['UP', 'DOWN', 'NEUTRAL']
    means = [stats['votes']['up_mean'], stats['votes']['down_mean'], stats['votes']['neutral_mean']]
    
    colors = ['green', 'red', 'gray']
    plt.bar(labels, means, color=colors)
    
    plt.title(f"{TICKER} Average Direction Votes Across {NUM_ITERATIONS} Iterations", fontsize=16)
    plt.ylabel('Percentage', fontsize=12)
    plt.grid(True, axis='y', alpha=0.3)
    
    # Add value labels
    for i, v in enumerate(means):
        plt.text(i, v + 0.5, f"{v:.1f}%", ha='center')
    
    plt.tight_layout()
    plt.savefig(f"outputs/multiple_runs/{TICKER}_vote_distribution_{timestamp}.png")
    plt.close()
    
    # Save results to CSV
    results_df = pd.DataFrame(valid_results)
    results_df.to_csv(f"outputs/multiple_runs/{TICKER}_multiple_runs_{timestamp}.csv", index=False)
    
    print(f"Plots and results saved to outputs/multiple_runs/ with timestamp {timestamp}")

def print_summary(stats):
    """Print a summary of the results"""
    print("\n" + "=" * 60)
    print(f"SUMMARY OF {NUM_ITERATIONS} ENSEMBLE RUNS FOR {TICKER}")
    print("=" * 60)
    
    print("\nFinal Predicted Price:")
    print(f"  Mean: ${stats['final_price']['mean']:.2f}")
    print(f"  Median: ${stats['final_price']['median']:.2f}")
    print(f"  Standard Deviation: ${stats['final_price']['std']:.2f}")
    print(f"  Range: ${stats['final_price']['min']:.2f} - ${stats['final_price']['max']:.2f}")
    
    print("\nPredicted Change:")
    print(f"  Mean: {stats['predicted_change']['mean']:.2f}%")
    print(f"  Median: {stats['predicted_change']['median']:.2f}%")
    print(f"  Standard Deviation: {stats['predicted_change']['std']:.2f}%")
    print(f"  Range: {stats['predicted_change']['min']:.2f}% - {stats['predicted_change']['max']:.2f}%")
    
    print("\nAverage Direction Distribution:")
    print(f"  UP: {stats['votes']['up_mean']:.1f}%")
    print(f"  DOWN: {stats['votes']['down_mean']:.1f}%")
    print(f"  NEUTRAL: {stats['votes']['neutral_mean']:.1f}%")
    
    # Determine overall signal
    max_vote = max(stats['votes']['up_mean'], stats['votes']['down_mean'], stats['votes']['neutral_mean'])
    signal = "BUY" if max_vote == stats['votes']['up_mean'] else "SELL" if max_vote == stats['votes']['down_mean'] else "HOLD"
    confidence = max_vote
    
    print("\nOverall Trading Signal:")
    print(f"  {signal} with {confidence:.1f}% confidence")
    print("=" * 60)

def main():
    print(f"Starting multiple ensemble analysis for {TICKER}...")
    print(f"Will run {NUM_ITERATIONS} iterations...")
    
    start_time = time.time()
    results = []
    
    for i in range(NUM_ITERATIONS):
        result = run_ensemble_analysis(i)
        if result is not None:
            results.append(result)
    
    print(f"\nCompleted {len(results)} successful iterations out of {NUM_ITERATIONS}")
    
    if results:
        stats = analyze_results(results)
        print_summary(stats)
    else:
        print("No valid results to analyze")
    
    end_time = time.time()
    print(f"\nTotal execution time: {(end_time - start_time) / 60:.2f} minutes")

if __name__ == "__main__":
    main() 
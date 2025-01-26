import json
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def plot_training_loss(json_file):
    # Read the JSON file
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # Extract the log history
    log_history = data['log_history']
    
    # Convert to DataFrame for easier plotting
    df = pd.DataFrame(log_history)
    
    # Create regular epoch intervals of 0.5
    epoch_intervals = np.arange(0, df['epoch'].max() + 0.5, 0.5)
    
    # Find the closest points to our desired intervals
    plot_points = []
    for interval in epoch_intervals:
        closest_idx = (df['epoch'] - interval).abs().idxmin()
        plot_points.append({
            'epoch': df.loc[closest_idx, 'epoch'],
            'loss': df.loc[closest_idx, 'loss']
        })
    
    # Convert plot points to DataFrame
    plot_df = pd.DataFrame(plot_points)
    
    # Create the plot
    plt.figure(figsize=(12, 6))
    
    # Plot full data as a light line
    plt.plot(df['epoch'], df['loss'], 'lightblue', alpha=0.3, linewidth=1, label='All points')
    
    # Plot interval points with markers
    plt.plot(plot_df['epoch'], plot_df['loss'], 'b-o', linewidth=2, markersize=6, label='0.5 epoch intervals')
    
    # Customize the plot
    plt.title('Training Loss Over Time (0.5 Epoch Intervals)', fontsize=14, pad=15)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add legend
    plt.legend()
    
    # Add margin to y-axis
    plt.margins(y=0.1)
    
    # Format y-axis to show fewer decimal places
    plt.gca().yaxis.set_major_formatter(plt.FormatStrFormatter('%.4f'))
    
    # Set x-axis ticks to show 0.5 intervals
    plt.xticks(epoch_intervals, rotation=45)
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('sop_training_loss.png')
    plt.close()
    
    # Print some basic statistics
    print("\nStatistics at 0.5 epoch intervals:")
    for _, row in plot_df.iterrows():
        print(f"Epoch {row['epoch']:.1f}: Loss = {row['loss']:.4f}")
    
    print(f"\nOverall Statistics:")
    print(f"Initial loss: {df['loss'].iloc[0]:.4f}")
    print(f"Final loss: {df['loss'].iloc[-1]:.4f}")
    print(f"Minimum loss: {df['loss'].min():.4f}")
    print(f"Total epochs: {df['epoch'].max():.2f}")

# Usage
if __name__ == "__main__":
    plot_training_loss('./BERT-SOP/trainer_state.json')
"""
Visualize bout length distributions across receivers to diagnose short bout issues.
Compare actual bout lengths to expected half-hour to hour-long bouts.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Set up plotting style
plt.rcParams['figure.figsize'] = (16, 10)

def analyze_bout_lengths(project_dir):
    """
    Analyze and visualize bout length distributions from presence data.
    
    Parameters
    ----------
    project_dir : str
        Path to project directory containing project.h5
    """
    
    # Load presence data
    h5_path = Path(project_dir) / 'project.h5'
    print(f"Loading presence data from: {h5_path}")
    
    presence = pd.read_hdf(h5_path, key='/presence')
    print(f"Loaded {len(presence):,} presence records")
    print(f"Columns: {list(presence.columns)}")
    
    # Calculate bout duration in seconds and hours
    presence['bout_duration_sec'] = presence['last_epoch'] - presence['first_epoch']
    presence['bout_duration_hrs'] = presence['bout_duration_sec'] / 3600
    
    # Get bout statistics
    print("\n" + "="*80)
    print("BOUT LENGTH STATISTICS")
    print("="*80)
    
    # Overall statistics
    print(f"\nOverall bout duration statistics (hours):")
    print(presence['bout_duration_hrs'].describe())
    
    # Count very short bouts (< 5 minutes)
    short_bouts = presence[presence['bout_duration_sec'] < 300]
    print(f"\nBouts < 5 minutes: {len(short_bouts):,} ({100*len(short_bouts)/len(presence):.1f}%)")
    
    # Count medium bouts (5 min - 30 min)
    medium_bouts = presence[(presence['bout_duration_sec'] >= 300) & 
                            (presence['bout_duration_sec'] < 1800)]
    print(f"Bouts 5-30 minutes: {len(medium_bouts):,} ({100*len(medium_bouts)/len(presence):.1f}%)")
    
    # Count expected length bouts (30 min - 2 hours)
    expected_bouts = presence[(presence['bout_duration_sec'] >= 1800) & 
                              (presence['bout_duration_sec'] < 7200)]
    print(f"Bouts 30min-2hrs: {len(expected_bouts):,} ({100*len(expected_bouts)/len(presence):.1f}%)")
    
    # Count long bouts (> 2 hours)
    long_bouts = presence[presence['bout_duration_sec'] >= 7200]
    print(f"Bouts > 2 hours: {len(long_bouts):,} ({100*len(long_bouts)/len(presence):.1f}%)")
    
    # Per-receiver statistics
    print("\n" + "="*80)
    print("BOUT LENGTHS BY RECEIVER")
    print("="*80)
    
    receiver_stats = presence.groupby('rec_id').agg({
        'bout_duration_hrs': ['count', 'min', 'median', 'mean', 'max'],
        'num_detections': ['median', 'mean']
    }).round(2)
    
    print("\n", receiver_stats)
    
    # Count short bouts per receiver
    short_by_rec = presence[presence['bout_duration_sec'] < 300].groupby('rec_id').size()
    total_by_rec = presence.groupby('rec_id').size()
    pct_short = (100 * short_by_rec / total_by_rec).round(1)
    
    print("\n% of bouts < 5 minutes by receiver:")
    print(pct_short.sort_values(ascending=False))
    
    # Create visualizations
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    
    # 1. Overall distribution (log scale)
    ax = axes[0, 0]
    presence['bout_duration_hrs'].hist(bins=100, ax=ax, edgecolor='black', alpha=0.7)
    ax.axvline(0.5, color='red', linestyle='--', linewidth=2, label='30 min')
    ax.axvline(1.0, color='orange', linestyle='--', linewidth=2, label='1 hour')
    ax.set_xlabel('Bout Duration (hours)')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Bout Lengths (All Receivers)')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Zoomed in on short bouts (0-2 hours)
    ax = axes[0, 1]
    short_data = presence[presence['bout_duration_hrs'] <= 2]['bout_duration_hrs']
    short_data.hist(bins=50, ax=ax, edgecolor='black', alpha=0.7)
    ax.axvline(0.5, color='red', linestyle='--', linewidth=2, label='30 min')
    ax.axvline(1.0, color='orange', linestyle='--', linewidth=2, label='1 hour')
    ax.set_xlabel('Bout Duration (hours)')
    ax.set_ylabel('Frequency')
    ax.set_title('Bout Lengths 0-2 Hours (Zoomed)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Box plot by receiver
    ax = axes[1, 0]
    receiver_order = presence.groupby('rec_id')['bout_duration_hrs'].median().sort_values().index
    presence.boxplot(column='bout_duration_hrs', by='rec_id', ax=ax, 
                     positions=range(len(receiver_order)))
    ax.set_xticklabels(receiver_order, rotation=45, ha='right')
    ax.axhline(0.5, color='red', linestyle='--', linewidth=1.5, label='30 min')
    ax.axhline(1.0, color='orange', linestyle='--', linewidth=1.5, label='1 hour')
    ax.set_ylabel('Bout Duration (hours)')
    ax.set_title('Bout Length Distribution by Receiver')
    ax.set_yscale('log')
    ax.legend()
    plt.suptitle('')  # Remove auto-generated title
    
    # 4. Detections per bout vs bout duration
    ax = axes[1, 1]
    sample = presence.sample(min(10000, len(presence)))  # Sample for performance
    ax.scatter(sample['num_detections'], sample['bout_duration_hrs'], 
               alpha=0.3, s=10)
    ax.set_xlabel('Number of Detections per Bout')
    ax.set_ylabel('Bout Duration (hours)')
    ax.set_title('Detections vs Bout Duration')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.axhline(0.5, color='red', linestyle='--', alpha=0.5, label='30 min')
    ax.axhline(1.0, color='orange', linestyle='--', alpha=0.5, label='1 hour')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 5. Cumulative distribution
    ax = axes[2, 0]
    sorted_durations = np.sort(presence['bout_duration_hrs'])
    cumulative = np.arange(1, len(sorted_durations) + 1) / len(sorted_durations) * 100
    ax.plot(sorted_durations, cumulative, linewidth=2)
    ax.axvline(0.5, color='red', linestyle='--', linewidth=2, label='30 min')
    ax.axvline(1.0, color='orange', linestyle='--', linewidth=2, label='1 hour')
    ax.set_xlabel('Bout Duration (hours)')
    ax.set_ylabel('Cumulative % of Bouts')
    ax.set_title('Cumulative Distribution of Bout Lengths')
    ax.set_xscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 6. Bout duration by number of detections (binned)
    ax = axes[2, 1]
    # Bin by detection count
    presence['det_bin'] = pd.cut(presence['num_detections'], 
                                 bins=[0, 3, 10, 50, 100, 500, 10000],
                                 labels=['1-3', '4-10', '11-50', '51-100', '101-500', '500+'])
    
    presence.boxplot(column='bout_duration_hrs', by='det_bin', ax=ax)
    ax.axhline(0.5, color='red', linestyle='--', linewidth=1.5, label='30 min')
    ax.axhline(1.0, color='orange', linestyle='--', linewidth=1.5, label='1 hour')
    ax.set_xlabel('Detections per Bout')
    ax.set_ylabel('Bout Duration (hours)')
    ax.set_title('Bout Duration by Detection Count')
    ax.set_yscale('log')
    ax.legend()
    plt.suptitle('')
    
    plt.tight_layout()
    
    # Save figure
    output_path = Path(project_dir) / 'bout_length_analysis.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved visualization to: {output_path}")
    
    # Show problematic short bouts
    print("\n" + "="*80)
    print("SAMPLE OF VERY SHORT BOUTS (< 5 minutes)")
    print("="*80)
    
    very_short = presence[presence['bout_duration_sec'] < 300].copy()
    very_short = very_short.sort_values('num_detections')
    
    print(f"\nShowing 20 shortest bouts:")
    print(very_short[['rec_id', 'freq_code', 'num_detections', 
                      'bout_duration_sec', 'bout_duration_hrs']].head(20).to_string())
    
    # Show receivers with highest proportion of short bouts
    print("\n" + "="*80)
    print("RECEIVERS WITH MOST SHORT BOUTS (< 5 min)")
    print("="*80)
    
    short_bout_summary = pd.DataFrame({
        'total_bouts': total_by_rec,
        'short_bouts': short_by_rec,
        'pct_short': pct_short
    }).sort_values('pct_short', ascending=False)
    
    print("\n", short_bout_summary)
    
    plt.show()
    
    return presence

if __name__ == '__main__':
    # Update this path to your project directory
    project_dir = r'c:\Users\Kevin.Nebiolo\OneDrive - Kleinschmidt Associates\Software\mast'
    
    presence_df = analyze_bout_lengths(project_dir)
    
    print("\n" + "="*80)
    print("Analysis complete!")
    print("="*80)
    print("\nNext steps:")
    print("1. If most bouts are < 30 minutes, increase eps_multiplier in DBSCAN")
    print("2. If you see many 1-3 detection bouts, these are likely spurious")
    print("3. Compare bout lengths to your expected 0.5-1 hour residence times")
    print("4. Consider filtering bouts by duration threshold if needed")

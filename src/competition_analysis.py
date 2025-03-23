#!/usr/bin/env python
"""
Competition Analysis for IMA Case Competition

This script analyzes the raw competition data to:
1. Identify Mavericks' weaknesses through detailed statistical analysis
2. Develop data-driven weightings for player evaluation
3. Generate insights for the competition report
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats

def load_competition_data():
    """Load and prepare the raw competition data."""
    project_root = Path(__file__).parent.parent
    data_dir = project_root / "data"
    
    # Load main data files
    player_data = pd.read_excel(data_dir / "2025_scc_5a_nba_player_data_2023.xlsx")
    data_dict = pd.read_excel(data_dir / "2025_scc_5b_nba_player_data_dictionary.xlsx")
    team_locations = pd.read_excel(data_dir / "2025_scc_5c_nba_team_locations.xlsx")
    
    return player_data, data_dict, team_locations

def analyze_mavericks_weaknesses(player_data):
    """
    Perform detailed analysis of Mavericks' weaknesses.
    Returns statistical measures of team performance gaps.
    """
    mavs_data = player_data[player_data['Team'] == 'DAL']
    league_data = player_data[player_data['Team'] != 'DAL']
    
    # Define key performance metrics to analyze
    performance_metrics = [
        'Total_Points', 'Total_Rebounds', 'Assists', 'Steals', 'Blocks',
        'Field_Goals_Made', 'Three_Pointers_Made', 'Free_Throws_Made',
        'Turnovers', 'Personal_Fouls'
    ]
    
    # Calculate per-game averages for both teams
    mavs_per_game = mavs_data[performance_metrics].mean()
    league_per_game = league_data[performance_metrics].mean()
    
    # Calculate statistical significance of differences
    significance = {}
    for metric in performance_metrics:
        t_stat, p_value = stats.ttest_ind(
            mavs_data[metric].dropna(),
            league_data[metric].dropna()
        )
        significance[metric] = {
            't_statistic': t_stat,
            'p_value': p_value,
            'significant': p_value < 0.05
        }
    
    # Create comprehensive comparison
    comparison = pd.DataFrame({
        'Mavericks': mavs_per_game,
        'League_Average': league_per_game,
        'Difference': mavs_per_game - league_per_game,
        'P_Value': [significance[m]['p_value'] for m in performance_metrics],
        'Significant': [significance[m]['significant'] for m in performance_metrics]
    })
    
    return comparison

def calculate_metric_weights(weakness_analysis):
    """
    Calculate weights for different metrics based on team weaknesses.
    Weights are determined by:
    1. Magnitude of the performance gap
    2. Statistical significance of the gap
    3. Relative importance of the metric
    """
    # Start with absolute differences
    weights = weakness_analysis['Difference'].abs()
    
    # Adjust for statistical significance
    weights = weights * weakness_analysis['Significant'].astype(float)
    
    # Handle any NaN values
    weights = weights.fillna(0)
    
    # If all weights are zero, set them equal
    if weights.sum() == 0:
        weights = pd.Series(1.0 / len(weights), index=weights.index)
    else:
        # Normalize weights
        weights = weights / weights.sum()
    
    return weights

def visualize_weaknesses(weakness_analysis, weights):
    """Create visualizations of team weaknesses and metric weights."""
    # Set up the plotting style
    plt.style.use('default')  # Using default style instead of seaborn
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Performance Gaps
    gaps = weakness_analysis['Difference'].sort_values(ascending=True)
    gaps.plot(kind='barh', ax=ax1)
    ax1.set_title('Mavericks Performance Gaps vs League Average')
    ax1.set_xlabel('Difference (Mavs - League)')
    
    # Plot 2: Metric Weights
    weights.plot(kind='barh', ax=ax2)
    ax2.set_title('Metric Weights Based on Team Weaknesses')
    ax2.set_xlabel('Weight')
    
    plt.tight_layout()
    return fig

def generate_insights(weakness_analysis, weights):
    """Generate key insights from the analysis."""
    insights = {
        'critical_weaknesses': [],
        'strengths': [],
        'weighting_justification': [],
        'recommendations': []
    }
    
    # Identify critical weaknesses (significant negative gaps)
    critical_weaknesses = weakness_analysis[
        (weakness_analysis['Difference'] < 0) & 
        (weakness_analysis['Significant'])
    ]
    
    for metric, row in critical_weaknesses.iterrows():
        insights['critical_weaknesses'].append(
            f"{metric}: Mavericks are {abs(row['Difference']):.2f} below league average "
            f"(p={row['P_Value']:.3f})"
        )
    
    # Identify strengths (significant positive gaps)
    strengths = weakness_analysis[
        (weakness_analysis['Difference'] > 0) & 
        (weakness_analysis['Significant'])
    ]
    
    for metric, row in strengths.iterrows():
        insights['strengths'].append(
            f"{metric}: Mavericks are {row['Difference']:.2f} above league average "
            f"(p={row['P_Value']:.3f})"
        )
    
    # Justify weightings
    top_weights = weights.nlargest(3)
    for metric, weight in top_weights.items():
        insights['weighting_justification'].append(
            f"{metric}: Weight of {weight:.3f} based on significant gap of "
            f"{abs(weakness_analysis.loc[metric, 'Difference']):.2f}"
        )
    
    return insights

def main():
    # Load data
    print("Loading competition data...")
    player_data, data_dict, team_locations = load_competition_data()
    
    # Analyze weaknesses
    print("\nAnalyzing Mavericks' weaknesses...")
    weakness_analysis = analyze_mavericks_weaknesses(player_data)
    
    # Calculate weights
    print("\nCalculating metric weights...")
    weights = calculate_metric_weights(weakness_analysis)
    
    # Generate insights
    print("\nGenerating insights...")
    insights = generate_insights(weakness_analysis, weights)
    
    # Create visualizations
    print("\nCreating visualizations...")
    fig = visualize_weaknesses(weakness_analysis, weights)
    
    # Save results
    output_dir = Path(__file__).parent.parent / "output"
    output_dir.mkdir(exist_ok=True)
    
    # Save analysis results
    weakness_analysis.to_excel(output_dir / "weakness_analysis.xlsx")
    weights.to_excel(output_dir / "metric_weights.xlsx")
    fig.savefig(output_dir / "weakness_visualization.png")
    
    # Print insights
    print("\nKey Insights:")
    print("\nCritical Weaknesses:")
    for weakness in insights['critical_weaknesses']:
        print(f"- {weakness}")
    
    print("\nTeam Strengths:")
    for strength in insights['strengths']:
        print(f"- {strength}")
    
    print("\nWeighting Justification:")
    for justification in insights['weighting_justification']:
        print(f"- {justification}")
    
    print(f"\nResults saved to: {output_dir}")

if __name__ == "__main__":
    main() 
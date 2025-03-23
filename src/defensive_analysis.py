#!/usr/bin/env python
"""
Mavericks Defensive Analysis

This script analyzes the Mavericks' defensive weaknesses and recommends
appropriate weightings for defensive metrics.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats

def load_player_data():
    """Load the NBA player data."""
    project_root = Path(__file__).parent.parent
    data_dir = project_root / "data"
    
    # Load player data
    player_data = pd.read_excel(data_dir / "2025_scc_5a_nba_player_data_2023.xlsx")
    return player_data

def analyze_defensive_metrics(player_data):
    """Analyze defensive metrics for Mavericks vs league average."""
    # Identify Mavericks players
    mavs_data = player_data[player_data['Team'] == 'DAL']
    league_data = player_data[player_data['Team'] != 'DAL']
    
    # Create derived defensive metrics
    for df in [player_data, mavs_data, league_data]:
        # Per game stats
        df['Blocks_Per_Game'] = df['Blocks'] / df['Games_Played']
        df['Steals_Per_Game'] = df['Steals'] / df['Games_Played']
        df['Defensive_Rebounds_Per_Game'] = df['Defensive_Rebounds'] / df['Games_Played']
        df['Total_Rebounds_Per_Game'] = df['Total_Rebounds'] / df['Games_Played']
        
        # Opponent metrics (estimated)
        # For team-level metrics, we'll use available player stats to approximate
        df['Opponent_FG_Pct'] = 1.0 - df['FG_percentage']
        df['Opponent_3PT_Pct'] = 1.0 - df['Three_Pointers_Percentage']
        
        # Calculate defensive efficiency
        if 'Defensive_Rating' not in df.columns:
            # Approximate defensive rating (points allowed per 100 possessions)
            # This is a simplified version since we don't have direct opponent data
            df['Defensive_Rating'] = 100 * (df['Personal_Fouls'] * 0.44) / (df['Steals'] + df['Blocks'])
    
    # Define key defensive metrics to analyze
    defensive_metrics = [
        'Blocks_Per_Game',
        'Steals_Per_Game',
        'Defensive_Rebounds_Per_Game',
        'Total_Rebounds_Per_Game',
        'Opponent_FG_Pct',         # Lower is better for defense
        'Opponent_3PT_Pct',        # Lower is better for defense
        'Personal_Fouls',          # Lower is better for defense
        'Defensive_Rating'         # Lower is better
    ]
    
    # Make sure these columns exist in our dataset
    available_metrics = [m for m in defensive_metrics if m in player_data.columns]
    
    # Calculate team-level statistics
    mavs_stats = {}
    league_stats = {}
    
    for metric in available_metrics:
        # For per-game stats and percentages, we want means
        if 'Per_Game' in metric or 'Pct' in metric or 'Rating' in metric:
            mavs_stats[metric] = mavs_data[metric].mean()
            league_stats[metric] = league_data[metric].mean()
        else:
            # For cumulative stats, we need to sum and normalize by games
            mavs_stats[metric] = mavs_data[metric].sum() / mavs_data['Games_Played'].sum()
            league_stats[metric] = league_data[metric].sum() / league_data['Games_Played'].sum()
    
    # Convert to Series
    mavs_stats = pd.Series(mavs_stats)
    league_stats = pd.Series(league_stats)
    
    # Calculate percentage difference
    pct_diff = (mavs_stats - league_stats) / league_stats * 100
    
    # For "lower is better" metrics, reverse the interpretation
    for metric in available_metrics:
        if 'Opponent' in metric or metric == 'Personal_Fouls' or metric == 'Defensive_Rating':
            pct_diff[metric] = -pct_diff[metric]
    
    # Statistical significance testing
    p_values = {}
    significance = {}
    for metric in available_metrics:
        if len(mavs_data[metric].dropna()) > 1 and len(league_data[metric].dropna()) > 1:
            t_stat, p_val = stats.ttest_ind(
                mavs_data[metric].dropna(),
                league_data[metric].dropna(),
                equal_var=False
            )
            p_values[metric] = p_val
            significance[metric] = p_val < 0.05
    
    # Compile results
    results = pd.DataFrame({
        'Mavericks': mavs_stats,
        'League_Average': league_stats,
        'Difference': mavs_stats - league_stats,
        'Pct_Difference': pct_diff,
        'P_Value': pd.Series(p_values),
        'Significant': pd.Series(significance)
    })
    
    return results

def calculate_defensive_weightings(defense_analysis):
    """
    Calculate recommended weightings for defensive metrics based on team weaknesses.
    Combines statistical analysis with basketball domain knowledge.
    """
    # Define baseline importance of each defensive metric (domain knowledge)
    baseline_importance = {
        'Blocks_Per_Game': 15,
        'Steals_Per_Game': 15,
        'Defensive_Rebounds_Per_Game': 20,
        'Total_Rebounds_Per_Game': 15,
        'Opponent_FG_Pct': 15,
        'Opponent_3PT_Pct': 10,
        'Personal_Fouls': 5,
        'Defensive_Rating': 5
    }
    
    # Get metrics with negative percentage difference (team is worse than league average)
    weaknesses = defense_analysis[defense_analysis['Pct_Difference'] < 0]
    
    # Start with baseline weights
    all_metrics = defense_analysis.index
    weights = pd.Series({m: baseline_importance.get(m, 10) for m in all_metrics})
    
    # If we have identified weaknesses, adjust weights based on weakness magnitude
    if not weaknesses.empty:
        # Calculate weakness magnitudes
        weakness_magnitudes = weaknesses['Pct_Difference'].abs()
        
        # Adjust for statistical significance
        if 'Significant' in weaknesses.columns:
            # Double the weight of statistically significant weaknesses
            significance_factor = weaknesses['Significant'].astype(float) * 1.0 + 1.0
            weakness_magnitudes = weakness_magnitudes * significance_factor
        
        # Handle any NaN values
        weakness_magnitudes = weakness_magnitudes.fillna(0)
        
        # Increase the weight for weak areas (the weaker the area, the more weight it gets)
        for metric in weaknesses.index:
            # Calculate a boost factor based on weakness magnitude
            # This will give more weight to metrics with larger weaknesses
            boost_factor = 1.0 + (weakness_magnitudes[metric] / 100)
            weights[metric] = weights[metric] * boost_factor
    
    # Normalize to 100%
    weights = weights / weights.sum() * 100
    
    return weights

def main():
    # Load player data
    print("Loading NBA player data...")
    player_data = load_player_data()
    
    # Display column names to see what's available
    print("\nAvailable columns in dataset:")
    print(player_data.columns.tolist())
    
    # Analyze defensive metrics
    print("\nAnalyzing Mavericks defensive metrics...")
    defense_analysis = analyze_defensive_metrics(player_data)
    
    # Calculate recommended weightings
    print("\nCalculating recommended defensive weightings...")
    weightings = calculate_defensive_weightings(defense_analysis)
    
    # Output results
    print("\n" + "="*50)
    print("           MAVERICKS DEFENSIVE ANALYSIS")
    print("="*50)
    
    print("\nA. Dallas Mavericks Defensive Weaknesses:")
    print("-"*45)
    
    # Print weaknesses (negative percentage differences)
    weaknesses = defense_analysis[defense_analysis['Pct_Difference'] < 0].sort_values('Pct_Difference')
    
    if not weaknesses.empty:
        for metric, row in weaknesses.iterrows():
            sig_marker = "**" if row.get('Significant', False) else ""
            print(f"- {metric}: {row['Mavericks']:.2f} vs. League {row['League_Average']:.2f} "
                f"({abs(row['Pct_Difference']):.1f}% worse){sig_marker}")
            
            # Add contextual explanation for each weakness
            if 'Blocks' in metric:
                print("  Impact: Poor shot blocking allows opponents easier scoring opportunities at the rim")
            elif 'Steals' in metric:
                print("  Impact: Low steal rate indicates passive defense and fewer transition opportunities")
            elif 'Rebounds' in metric:
                print("  Impact: Rebounding weakness gives opponents second-chance opportunities")
            elif 'Opponent_FG_Pct' in metric:
                print("  Impact: High opponent shooting efficiency indicates poor defensive pressure")
            elif 'Opponent_3PT_Pct' in metric:
                print("  Impact: Poor perimeter defense allows high-value scoring opportunities")
            elif 'Personal_Fouls' in metric:
                print("  Impact: Excessive fouling leads to free points and player availability issues")
            elif 'Defensive_Rating' in metric:
                print("  Impact: Overall defensive inefficiency requires comprehensive defensive improvement")
    else:
        print("No significant defensive weaknesses identified in the available data.")
    
    print("\nB. Recommended Defensive Stat Weightings:")
    print("-"*45)
    
    # Sort weightings by importance
    sorted_weights = weightings.sort_values(ascending=False)
    
    for metric, weight in sorted_weights.items():
        print(f"- {metric}: {weight:.1f}%")
        # Add explanation for each weighting
        if 'Blocks' in metric:
            print("  Rationale: Shot blocking directly prevents scoring and intimidates opponents")
        elif 'Steals' in metric:
            print("  Rationale: Steals create transition opportunities and disrupt opponent offense")
        elif 'Total_Rebounds' in metric:
            print("  Rationale: Ending opponent possessions is fundamental to good defense")
        elif 'Defensive_Rebounds' in metric:
            print("  Rationale: Limiting second-chance points is critical for defensive efficiency")
        elif 'Opponent_FG_Pct' in metric:
            print("  Rationale: Forcing missed shots is the primary objective of defense")
        elif 'Opponent_3PT_Pct' in metric:
            print("  Rationale: Three-point defense is especially valuable in modern NBA")
        elif 'Personal_Fouls' in metric:
            print("  Rationale: Avoiding fouls prevents free points and keeps players on the floor")
        elif 'Defensive_Rating' in metric:
            print("  Rationale: Overall defensive efficiency is the ultimate defensive goal")
    
    # Note about statistical significance
    if "**" in ''.join([f"{row.get('Significant', False)}" for _, row in weaknesses.iterrows()]):
        print("\nNote: ** indicates statistically significant difference (p < 0.05)")
    
    # Save results to file
    output_dir = Path(__file__).parent.parent / "output"
    output_dir.mkdir(exist_ok=True)
    defense_analysis.to_excel(output_dir / "mavericks_defensive_analysis.xlsx")
    weightings.to_excel(output_dir / "recommended_defensive_weightings.xlsx")
    
    print(f"\nFull analysis results saved to: {output_dir}")

if __name__ == "__main__":
    main() 
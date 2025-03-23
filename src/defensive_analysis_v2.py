#!/usr/bin/env python
"""
Mavericks Gap Analysis

This program loads the NBA player data, cleans and preprocesses it,
and then focuses on the Dallas Mavericks ("DAL") by computing per-game averages
for key defensive metrics. It then calculates league-wide per-game averages,
compares the Mavericks to the league, and identifies the top three metrics
(where the Mavericks fall most behind the league average). Finally, it outputs
a report and displays a bar chart of the gaps.

Key metrics used here (which you can modify or extend) include:
  - Blocks per Game (BPG)
  - Total Rebounds per Game (RPG)
  - Steals per Game (SPG)
  
Adjust file paths and metric choices as needed.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------
# 1. DATA LOADING & PREPROCESSING
# ---------------------------

def load_player_data(player_file):
    """
    Loads the NBA player data from an Excel file.
    
    Parameters:
      player_file (str): Path to the NBA player data Excel file.
    
    Returns:
      DataFrame: Raw player data.
    """
    df = pd.read_excel(player_file)
    return df

def preprocess_player_data(df):
    """
    Clean and preprocess the player data.
      - Remove currency symbols and commas from salary columns.
      - Convert percentage fields to decimals.
      - Fill missing numeric values with 0.
      - Standardize column names.
    
    Parameters:
      df (DataFrame): Raw player data.
    
    Returns:
      DataFrame: Cleaned player data.
    """
    # List of salary columns based on data dictionary
    salary_cols = ["Sal22 23", "Sal23 24", "Sal24 25", "Sal25 26", "Sal26 27", "Sal28 29", "Total Guaranteed Salary"]
    for col in salary_cols:
        if col in df.columns:
            df[col] = df[col].replace({'\$': '', ',': ''}, regex=True)
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Identify percentage columns by keyword (assuming columns contain "percentage")
    percent_cols = [col for col in df.columns if 'percentage' in col.lower()]
    for col in percent_cols:
        df[col] = df[col].astype(str).replace({'%': '', ',': ''}, regex=True)
        df[col] = pd.to_numeric(df[col], errors='coerce')
        if df[col].max() > 1:
            df[col] = df[col] / 100.0

    # Fill missing numeric values with 0
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(0)
    
    # Standardize column names: trim spaces, replace spaces with underscores, and lower-case them.
    df.columns = df.columns.str.strip().str.replace(' ', '_').str.lower()
    
    return df

# ---------------------------
# 2. COMPUTE PER-GAME METRICS AND COMPARE TO LEAGUE
# ---------------------------

def compute_per_game_stats(df):
    """
    Computes per-game statistics for key metrics.
    
    Assumes that the dataset includes:
      - games_played, blocks, total_rebounds, steals.
      
    Adds new columns:
      - blocks_per_game, rebounds_per_game, steals_per_game.
    
    Parameters:
      df (DataFrame): Cleaned player data.
    
    Returns:
      DataFrame: The same DataFrame with additional per-game metric columns.
    """
    # Avoid division by zero: if games_played == 0, set per-game values to 0.
    df['blocks_per_game'] = df.apply(lambda row: row['blocks'] / row['games_played'] if row['games_played'] > 0 else 0, axis=1)
    df['rebounds_per_game'] = df.apply(lambda row: row['total_rebounds'] / row['games_played'] if row['games_played'] > 0 else 0, axis=1)
    df['steals_per_game'] = df.apply(lambda row: row['steals'] / row['games_played'] if row['games_played'] > 0 else 0, axis=1)
    return df

def compute_team_and_league_averages(df, metrics):
    """
    Computes league averages for given metrics, and the averages for the Mavericks.
    
    Parameters:
      df (DataFrame): The player dataset with per-game metrics.
      metrics (list): List of metric column names (e.g., ['blocks_per_game', 'rebounds_per_game', 'steals_per_game']).
    
    Returns:
      tuple: (league_avgs, mavericks_avgs, differences)
             Each is a dictionary mapping metric names to average values.
    """
    league_avgs = {}
    mavericks_avgs = {}
    differences = {}
    
    # Filter data for Mavericks; assume team column is 'tm' and Mavericks use "DAL"
    mavericks_df = df[df['tm'].str.upper() == "DAL"]
    
    for metric in metrics:
        if metric in df.columns:
            league_avgs[metric] = df[metric].mean()
            mavericks_avgs[metric] = mavericks_df[metric].mean()
            differences[metric] = mavericks_avgs[metric] - league_avgs[metric]
    
    return league_avgs, mavericks_avgs, differences

def identify_top_weaknesses(differences, top_n=3):
    """
    Identifies the top 'n' metrics where the Mavericks' performance is below league average.
    
    Parameters:
      differences (dict): Dictionary of {metric: (mavericks_avg - league_avg)}.
      top_n (int): Number of weakest metrics to identify.
    
    Returns:
      list: List of tuples (metric, difference) sorted by difference ascending.
            (More negative difference indicates a larger gap.)
    """
    # Sort metrics by difference (most negative first)
    sorted_metrics = sorted(differences.items(), key=lambda x: x[1])
    return sorted_metrics[:top_n]

def visualize_gap_report(league_avgs, mavericks_avgs, differences):
    """
    Creates a bar chart showing league average, Mavericks average, and the difference for each metric.
    
    Parameters:
      league_avgs (dict): League averages.
      mavericks_avgs (dict): Mavericks averages.
      differences (dict): Difference (Mavericks - League).
    """
    # Set seaborn style for better visualization
    sns.set_style("whitegrid")
    
    metrics = list(league_avgs.keys())
    league_vals = [league_avgs[m] for m in metrics]
    mavericks_vals = [mavericks_avgs[m] for m in metrics]
    diff_vals = [differences[m] for m in metrics]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    plt.figure(figsize=(10,6))
    plt.bar(x - width/2, league_vals, width, label='League Average', color='lightgray')
    plt.bar(x + width/2, mavericks_vals, width, label='Mavericks Average', color='steelblue')
    plt.xticks(x, [m.replace('_', ' ').title() for m in metrics])
    plt.ylabel("Per-Game Average")
    plt.title("Mavericks vs. League Averages for Key Metrics")
    plt.legend()
    
    # Display difference as text above the bars for Mavericks
    for i, diff in enumerate(diff_vals):
        plt.text(x[i] + width/2, mavericks_vals[i] + 0.05, f"{diff:.2f}", ha='center', va='bottom', color='black')
    
    plt.tight_layout()
    plt.show()

# ---------------------------
# 3. MAIN FUNCTION: MAVERICKS GAP ANALYSIS
# ---------------------------

def main():
    # Set the file path for the player data Excel file.
    player_file = "../data/2025_scc_5a_nba_player_data_2023.xlsx"
    
    # 1. Load the player data.
    print("Loading player data...")
    player_df = load_player_data(player_file)
    
    # 2. Preprocess the player data.
    print("Preprocessing data...")
    player_df_clean = preprocess_player_data(player_df)
    
    # 3. Compute per-game statistics (blocks, rebounds, steals).
    print("Computing per-game statistics...")
    player_df_pg = compute_per_game_stats(player_df_clean)
    
    # 4. For our gap analysis, choose key defensive metrics.
    key_metrics = ['blocks_per_game', 'rebounds_per_game', 'steals_per_game']
    
    # 5. Compute league-wide and Mavericks (DAL) averages.
    league_avgs, mavericks_avgs, differences = compute_team_and_league_averages(player_df_pg, key_metrics)
    
    # 6. Identify the top 3 weakest links (metrics with largest negative difference).
    top_weaknesses = identify_top_weaknesses(differences, top_n=3)
    print("\nTop 3 Weakest Links for the Mavericks (Metric: Mavericks Average - League Average):")
    for metric, diff in top_weaknesses:
        print(f"  {metric.replace('_', ' ').title()}: {diff:.2f}")
    
    # 7. Visualize the gap report across all key metrics.
    visualize_gap_report(league_avgs, mavericks_avgs, differences)
    
    # Optionally, output a summary table.
    summary_df = pd.DataFrame({
        "Metric": [m.replace('_', ' ').title() for m in key_metrics],
        "League Average": [league_avgs[m] for m in key_metrics],
        "Mavericks Average": [mavericks_avgs[m] for m in key_metrics],
        "Difference": [differences[m] for m in key_metrics]
    })
    print("\nSummary Report:")
    print(summary_df)
    
    print("\nMavericks gap analysis completed.")

if __name__ == '__main__':
    main()

#!/usr/bin/env python
"""
Mavericks Player Acquisition Analysis for the 2025 IMA Case Competition

This script:
  1. Loads NBA player data and (optionally) supporting datasets.
  2. Preprocesses and cleans the data (salary fields, percentages, missing values, etc.).
  3. Creates derived metrics such as effective FG%, assist-to-turnover ratio, and salary efficiency.
  4. Performs a gap analysis specifically for the Dallas Mavericks by comparing their averages
     to league averages on key metrics.
  5. Filters and ranks players based on a composite score designed to address the Mavericks' needs:
     - Rim protection (blocks)
     - Rebounding (total rebounds)
     - Secondary scoring (effective FG%)
     - Cost-effectiveness (salary efficiency)
  6. Generates visualizations for team comparisons and salary vs. performance.
  
Adjust file paths and weight parameters as needed.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------
# 1. DATA LOADING & PREPROCESSING
# ---------------------------

def load_data(player_file, team_stats_file=None, data_dict_file=None, team_locations_file=None):
    """
    Load datasets from Excel files.
    
    Parameters:
        player_file (str): Path to the NBA player data Excel file.
        team_stats_file (str): (Optional) Path to team stats Excel file.
        data_dict_file (str): (Optional) Path to the data dictionary Excel file.
        team_locations_file (str): (Optional) Path to the team locations Excel file.
    
    Returns:
        tuple: (player_df, team_stats_df, data_dict_df, team_locations_df)
               If optional files are not provided, their DataFrames are set to None.
    """
    player_df = pd.read_excel(player_file)
    
    team_stats_df = pd.read_excel(team_stats_file) if team_stats_file else None
    data_dict_df = pd.read_excel(data_dict_file) if data_dict_file else None
    team_locations_df = pd.read_excel(team_locations_file) if team_locations_file else None
    
    return player_df, team_stats_df, data_dict_df, team_locations_df

def preprocess_player_data(df):
    """
    Preprocess player data by cleaning salary and percentage columns,
    filling missing numeric values, and standardizing column names.
    
    Parameters:
        df (DataFrame): Raw player data.
    
    Returns:
        DataFrame: Cleaned player data.
    """
    # List of salary columns from the data dictionary
    salary_cols = ["Sal22 23", "Sal23 24", "Sal24 25", "Sal25 26", "Sal26 27", "Sal28 29", "Total Guaranteed Salary"]
    for col in salary_cols:
        if col in df.columns:
            df[col] = df[col].replace({'\$': '', ',': ''}, regex=True)
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Identify percentage columns (by looking for 'percentage' in the column name, case-insensitive)
    percent_cols = [col for col in df.columns if 'percentage' in col.lower()]
    for col in percent_cols:
        df[col] = df[col].astype(str).replace({'%': '', ',': ''}, regex=True)
        df[col] = pd.to_numeric(df[col], errors='coerce')
        # If values appear on a 0-100 scale, convert to decimals (0-1)
        if df[col].max() > 1:
            df[col] = df[col] / 100.0

    # Fill missing numeric values with 0 (adjust as needed)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(0)
    
    # Standardize column names: strip, replace spaces with underscores, and lower-case all names.
    df.columns = df.columns.str.strip().str.replace(' ', '_').str.lower()
    
    return df

def merge_datasets(player_df, team_stats_df=None, team_locations_df=None):
    """
    Merge player data with team stats and team locations if provided.
    
    Parameters:
        player_df (DataFrame): Cleaned player data.
        team_stats_df (DataFrame): (Optional) Team stats data.
        team_locations_df (DataFrame): (Optional) Team locations data.
    
    Returns:
        DataFrame: Merged dataset.
    """
    merged_df = player_df.copy()
    # If team_stats data is provided, merge on the team identifier.
    if team_stats_df is not None:
        # Standardize team column names: assume player_df has 'tm' and team_stats_df has 'team'
        merged_df['tm'] = merged_df['tm'].str.strip().str.upper()
        team_stats_df['team'] = team_stats_df['team'].str.strip().str.upper()
        merged_df = pd.merge(merged_df, team_stats_df, how='left', left_on='tm', right_on='team')
    
    # Similarly, merge team_locations if provided.
    if team_locations_df is not None:
        team_locations_df['team'] = team_locations_df['team'].str.strip().str.upper()
        merged_df = pd.merge(merged_df, team_locations_df, how='left', on='team')
    
    return merged_df

def feature_engineering(df):
    """
    Create derived metrics:
      - Effective Field Goal Percentage (eFG%):
            eFG% = (field_goals_made + 0.5 * three_pointers_made) / field_goals_attempted
      - Assist-to-Turnover Ratio:
            ast_to_to_ratio = assists / turnovers (with safe division)
      - Salary Efficiency:
            salary_efficiency = total_guaranteed_salary / total_points (lower is better)
    
    Parameters:
        df (DataFrame): Merged dataset.
    
    Returns:
        DataFrame: Dataset with new feature columns.
    """
    # Calculate eFG% if columns exist
    req_cols_efg = ['field_goals_made', 'three_pointers_made', 'field_goals_attempted']
    if all(col in df.columns for col in req_cols_efg):
        df['efg_percentage'] = (df['field_goals_made'] + 0.5 * df['three_pointers_made']) / df['field_goals_attempted']
    
    # Calculate Assist-to-Turnover Ratio
    req_cols_ast_to = ['assists', 'turnovers']
    if all(col in df.columns for col in req_cols_ast_to):
        df['ast_to_to_ratio'] = df['assists'] / df['turnovers'].replace(0, np.nan)
    
    # Calculate Salary Efficiency (Total Guaranteed Salary per Total Points)
    if 'total_guaranteed_salary' in df.columns and 'total_points' in df.columns:
        # Replace zero total_points with nan to avoid division by zero
        df['salary_efficiency'] = df['total_guaranteed_salary'] / df['total_points'].replace(0, np.nan)
    
    return df

# ---------------------------
# 2. MAVERICKS TEAM GAP ANALYSIS
# ---------------------------

def analyze_team_performance(df, team_abbr):
    """
    Compare the selected team's averages for key metrics against league averages.
    
    Parameters:
        df (DataFrame): Dataset with player data.
        team_abbr (str): Team abbreviation to analyze (e.g., "DAL" for Mavericks).
    
    Returns:
        DataFrame: A summary table with league averages, team averages, and differences.
    """
    # Define key metrics of interest based on Mavericks' weaknesses:
    # We focus on: efg_percentage (secondary scoring), blocks (rim protection), 
    # total_rebounds (rebounding), and salary_efficiency (cost-effectiveness).
    metrics = ['efg_percentage', 'blocks', 'total_rebounds', 'salary_efficiency']
    
    # Compute league averages (using all players)
    league_avgs = {metric: df[metric].mean() for metric in metrics if metric in df.columns}
    league_df = pd.DataFrame(league_avgs, index=[0]).T.rename(columns={0: 'League_Average'})
    
    # Compute team averages (filter by team abbreviation, assumed in 'tm')
    team_data = df[df['tm'] == team_abbr]
    team_avgs = {metric: team_data[metric].mean() for metric in metrics if metric in team_data.columns}
    team_df = pd.DataFrame(team_avgs, index=[0]).T.rename(columns={0: f'{team_abbr}_Average'})
    
    # Combine and compute difference (team average - league average)
    comp_df = league_df.join(team_df)
    comp_df['Difference'] = comp_df[f'{team_abbr}_Average'] - comp_df['League_Average']
    
    return comp_df

# ---------------------------
# 3. PLAYER FILTERING & RANKING FOR MAVERICKS
# ---------------------------

def rank_players(df, position_filter=None, performance_metrics=None, salary_cap=None, weights=None, percentile_threshold=80):
    """
    Filter and rank players based on customized criteria.
    
    For the Mavericks, we focus on players who can help with:
      - Rim Protection & Interior Defense (blocks)
      - Rebounding (total_rebounds)
      - Secondary Scoring (efg_percentage)
      - Cost-Effectiveness (salary_efficiency; lower is better, so we invert)
    
    Parameters:
        df (DataFrame): Dataset with player performance and salary.
        position_filter (list): List of positions to include (e.g., ["c", "pf", "f"]). Uses substring matching.
        performance_metrics (list): List of metric names to include in scoring.
        salary_cap (float): Maximum allowed salary.
        weights (dict): Dictionary mapping metric names to weights.
        percentile_threshold (float): Keep players in the top X percentile of composite score.
    
    Returns:
        DataFrame: Filtered and ranked players with a computed composite score.
    """
    df_filtered = df.copy()
    
    # Filter by position if specified (assume column 'position' is lowercase)
    if position_filter and 'position' in df_filtered.columns:
        regex_pattern = '|'.join(position_filter)
        df_filtered = df_filtered[df_filtered['position'].str.contains(regex_pattern, case=False, na=False)]
    
    # Filter by salary cap (using total_guaranteed_salary)
    if salary_cap and 'total_guaranteed_salary' in df_filtered.columns:
        df_filtered = df_filtered[df_filtered['total_guaranteed_salary'] <= salary_cap]
    
    # If performance_metrics and weights are provided, normalize each and calculate composite score.
    if performance_metrics and weights:
        for metric in performance_metrics:
            if metric in df_filtered.columns:
                min_val = df_filtered[metric].min()
                max_val = df_filtered[metric].max()
                # Normalize the metric to [0,1]
                df_filtered[f'{metric}_norm'] = (df_filtered[metric] - min_val) / (max_val - min_val + 1e-6)
        
        # For cost metrics where lower is better, invert the normalized score.
        # Here, we assume 'salary_efficiency' is a cost metric.
        if 'salary_efficiency' in performance_metrics:
            df_filtered['salary_efficiency_norm'] = 1 - df_filtered['salary_efficiency_norm']
        
        # Calculate the weighted composite score.
        composite = np.zeros(len(df_filtered))
        for metric, weight in weights.items():
            norm_col = f'{metric}_norm'
            if norm_col in df_filtered.columns:
                composite += weight * df_filtered[norm_col]
        df_filtered['composite_score'] = composite
    
    # Apply percentile filter on the composite score, if available.
    if 'composite_score' in df_filtered.columns:
        threshold_value = np.percentile(df_filtered['composite_score'], percentile_threshold)
        df_filtered = df_filtered[df_filtered['composite_score'] >= threshold_value]
    
    # Rank players by composite score (descending order).
    df_filtered = df_filtered.sort_values(by='composite_score', ascending=False)
    
    # Select key columns for output (adjust based on available columns)
    output_cols = ['player_name', 'tm', 'position', 'age', 'total_points', 'efg_percentage', 
                   'blocks', 'total_rebounds', 'total_guaranteed_salary', 'salary_efficiency', 'composite_score']
    out_cols = [col for col in output_cols if col in df_filtered.columns]
    
    return df_filtered[out_cols]

# ---------------------------
# 4. VISUALIZATION & REPORTING
# ---------------------------

def visualize_team_comparison(comp_df, team_abbr):
    """
    Plot a bar chart comparing league averages and Mavericks' averages.
    
    Parameters:
        comp_df (DataFrame): Comparison DataFrame from analyze_team_performance.
        team_abbr (str): Team abbreviation (e.g., "DAL").
    """
    comp_df = comp_df.reset_index().rename(columns={'index': 'Metric'})
    plt.figure(figsize=(10,6))
    sns.barplot(data=comp_df, x='Metric', y='League_Average', color='lightgray', label='League Average')
    sns.barplot(data=comp_df, x='Metric', y=f'{team_abbr}_Average', color='steelblue', label=f'{team_abbr} Average')
    plt.xticks(rotation=45)
    plt.ylabel("Average Value")
    plt.title(f"{team_abbr} vs. League Averages on Key Metrics")
    plt.legend()
    plt.tight_layout()
    plt.show()

def visualize_salary_vs_metric(df, metric, salary_col='total_guaranteed_salary'):
    """
    Create a scatter plot of salary versus a chosen performance metric.
    
    Parameters:
        df (DataFrame): Player dataset.
        metric (str): Performance metric column to plot.
        salary_col (str): Salary column to use.
    """
    plt.figure(figsize=(10,6))
    sns.scatterplot(data=df, x=salary_col, y=metric, hue='position', palette='viridis')
    plt.title(f"Salary vs. {metric.replace('_', ' ').title()}")
    plt.xlabel("Total Guaranteed Salary")
    plt.ylabel(metric.replace('_', ' ').title())
    plt.tight_layout()
    plt.show()

# ---------------------------
# 5. MAIN WORKFLOW
# ---------------------------

def main():
    # --- Set file paths (update these paths as needed) ---
    player_file = "../data/2025_scc_5a_nba_player_data_2023.xlsx"
    data_dict_file = "../data/2025_scc_5b_nba_player_data_dictionary.xlsx"
    # Optionally, if you have team stats or team locations files, specify them.
    team_stats_file = None
    team_locations_file = "../data/2025_scc_5c_nba_team_locations.xlsx"

    # 1. Load Data
    player_df, team_stats_df, data_dict_df, team_locations_df = load_data(
        player_file, team_stats_file, data_dict_file, team_locations_file
    )
    print("Player data loaded. (Data Dictionary loaded if provided.)")
    
    # 2. Preprocess Player Data
    player_df_clean = preprocess_player_data(player_df)
    print("Player data preprocessed.")
    
    # 3. Merge Datasets (if additional datasets available; otherwise use player data only)
    merged_df = merge_datasets(player_df_clean, team_stats_df, team_locations_df)
    print("Datasets merged. Shape:", merged_df.shape)
    
    # 4. Feature Engineering: Derived metrics
    final_df = feature_engineering(merged_df)
    print("Feature engineering complete. Derived metrics added.")
    
    # 5. Mavericks Team Gap Analysis
    mavericks_abbr = "DAL"  # Dallas Mavericks
    team_comparison = analyze_team_performance(final_df, mavericks_abbr)
    print("\nMavericks Gap Analysis (Team vs. League Averages):")
    print(team_comparison)
    
    # Visualize the gap analysis
    visualize_team_comparison(team_comparison, mavericks_abbr)
    
    # 6. Player Filtering & Ranking for the Mavericks
    # Define criteria tailored to Mavericks' needs:
    # We focus on: effective FG% (secondary scoring), blocks (rim protection), total rebounds (rebounding),
    # and salary efficiency (cost-effectiveness). We choose positions that can help fix interior and wing issues.
    position_filter = ["c", "pf", "f"]  # centers, power forwards, and forwards
    performance_metrics = ['efg_percentage', 'blocks', 'total_rebounds', 'salary_efficiency']
    # Set weights (note: salary_efficiency will be inverted in feature engineering)
    weights = {
        'efg_percentage': 0.25,
        'blocks': 0.30,
        'total_rebounds': 0.30,
        'salary_efficiency': 0.15
    }
    salary_cap = 20000000  # e.g., $20 million; adjust if needed
    
    ranked_players = rank_players(final_df, position_filter=position_filter,
                                  performance_metrics=performance_metrics, salary_cap=salary_cap,
                                  weights=weights, percentile_threshold=80)
    
    print("\nTop Recommended Players for the Mavericks (Based on Composite Score):")
    print(ranked_players.head(10))
    
    # 7. Visualization: Plot salary vs. effective FG%
    visualize_salary_vs_metric(final_df, 'efg_percentage')
    
    # Additional visualizations and advanced algorithms (e.g., clustering, regression) can be added here.
    print("\nMavericks Player Acquisition Analysis pipeline completed.")

if __name__ == '__main__':
    main()

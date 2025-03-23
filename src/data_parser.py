#!/usr/bin/env python
"""
Data Parser for IMA Competition

This script helps parse and analyze the data from various Excel files
to build context for the Dallas Mavericks analysis.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def load_all_data():
    """
    Load all relevant data files for analysis.
    """
    # Get the project root directory (one level up from src)
    project_root = Path(__file__).parent.parent
    
    # Define paths relative to project root
    data_dir = project_root / "data"
    docs_dir = project_root / "docs" / "IMA"
    
    print(f"Loading data from: {data_dir}")
    print(f"Loading docs from: {docs_dir}")
    
    # Load main data files
    player_data = pd.read_excel(data_dir / "2025_scc_5a_nba_player_data_2023.xlsx")
    print("\nPlayer data columns:")
    print(player_data.columns.tolist())
    
    data_dict = pd.read_excel(data_dir / "2025_scc_5b_nba_player_data_dictionary.xlsx")
    print("\nData dictionary columns:")
    print(data_dict.columns.tolist())
    
    team_locations = pd.read_excel(data_dir / "2025_scc_5c_nba_team_locations.xlsx")
    print("\nTeam locations columns:")
    print(team_locations.columns.tolist())
    
    # Load additional analysis files
    top_centers = pd.read_excel(docs_dir / "25_ Top Centers.xlsx")
    print("\nTop centers columns:")
    print(top_centers.columns.tolist())
    
    filtering_data = pd.read_excel(docs_dir / "Filtering.xlsx")
    print("\nFiltering data columns:")
    print(filtering_data.columns.tolist())
    
    return {
        'player_data': player_data,
        'data_dict': data_dict,
        'team_locations': team_locations,
        'top_centers': top_centers,
        'filtering_data': filtering_data
    }

def analyze_mavericks_metrics(data):
    """
    Analyze key metrics for the Dallas Mavericks.
    """
    player_data = data['player_data']
    mavs_data = player_data[player_data['Team'] == 'DAL']
    
    # Calculate team averages for numeric columns only
    numeric_cols = player_data.select_dtypes(include=[np.number]).columns
    team_avgs = mavs_data[numeric_cols].mean()
    league_avgs = player_data[numeric_cols].mean()
    
    # Compare Mavs to league
    comparison = pd.DataFrame({
        'Mavericks': team_avgs,
        'League': league_avgs,
        'Difference': team_avgs - league_avgs
    })
    
    return comparison

def analyze_potential_players(data):
    """
    Analyze potential player acquisitions based on team needs.
    """
    player_data = data['player_data']
    top_centers = data['top_centers']
    
    # Filter for available players (not on Mavs)
    available_players = player_data[player_data['Team'] != 'DAL']
    
    # Calculate key metrics for available players
    available_players['Salary_Efficiency'] = available_players['Total_Points'] / available_players['Sal22_23']
    available_players['Defensive_Score'] = (
        available_players['Blocks'] + 
        available_players['Total_Rebounds'] + 
        available_players['Steals']
    )
    
    return available_players

def main():
    # Load all data
    print("Loading data files...")
    data = load_all_data()
    
    # Analyze Mavericks metrics
    print("\nAnalyzing Mavericks metrics...")
    mavs_comparison = analyze_mavericks_metrics(data)
    print("\nTop 5 metrics where Mavericks differ from league average:")
    print(mavs_comparison.sort_values('Difference', ascending=False).head())
    
    # Analyze potential players
    print("\nAnalyzing potential player acquisitions...")
    available_players = analyze_potential_players(data)
    print("\nTop 5 available players by defensive score:")
    print(available_players.nlargest(5, 'Defensive_Score')[['Player_Name', 'Team', 'Defensive_Score', 'Salary_Efficiency']])
    
    # Save analysis results
    output_dir = Path(__file__).parent.parent / "output"
    output_dir.mkdir(exist_ok=True)
    
    mavs_comparison.to_excel(output_dir / "mavericks_comparison.xlsx")
    available_players.to_excel(output_dir / "available_players.xlsx")
    print(f"\nAnalysis results saved to: {output_dir}")

if __name__ == "__main__":
    main() 
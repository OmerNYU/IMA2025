#!/usr/bin/env python
"""
Mavericks Player Acquisition Recommendations Generator

This script analyzes defensive metrics and available players to generate
formatted recommendations that address the Mavericks' defensive weaknesses.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

def load_analysis_data():
    """Load the defensive analysis and available players data."""
    project_root = Path(__file__).parent.parent
    output_dir = project_root / "output"
    
    # Load the defensive analysis results
    defense_analysis = pd.read_excel(output_dir / "mavericks_defensive_analysis.xlsx")
    
    # Load the available players data
    available_players = pd.read_excel(output_dir / "available_players.xlsx")
    
    # Load recommended weightings
    try:
        weightings = pd.read_excel(output_dir / "recommended_defensive_weightings.xlsx")
    except:
        # Create a default weightings series if file doesn't exist
        weightings = pd.Series({
            'Blocks_Per_Game': 16.3,
            'Steals_Per_Game': 16.5,
            'Defensive_Rebounds_Per_Game': 20.1,
            'Total_Rebounds_Per_Game': 15.8,
            'Opponent_FG_Pct': 13.3,
            'Opponent_3PT_Pct': 8.9,
            'Personal_Fouls': 4.6,
            'Defensive_Rating': 4.4
        })
    
    return defense_analysis, available_players, weightings

def analyze_player_fit(available_players, defense_analysis):
    """Calculate per-game stats and evaluate player fit for Mavs defensive needs."""
    # Calculate per-game metrics
    players = available_players.copy()
    
    # Calculate per-game statistics
    players['Blocks_Per_Game'] = players['Blocks'] / players['Games_Played']
    players['Steals_Per_Game'] = players['Steals'] / players['Games_Played']
    players['Defensive_Rebounds_Per_Game'] = players['Defensive_Rebounds'] / players['Games_Played']
    players['Total_Rebounds_Per_Game'] = players['Total_Rebounds'] / players['Games_Played']
    
    # Calculate salary efficiency
    players['Salary_Efficiency'] = players['Defensive_Score'] / players['Sal22_23']
    
    # Calculate the weighted defensive score based on Mavs needs
    # Get the Mavs defensive weaknesses (where they are below average)
    weaknesses = defense_analysis[defense_analysis['Pct_Difference'] < 0]
    
    # Calculate a custom fit score that emphasizes Mavs' weaknesses
    fit_scores = []
    for _, player in players.iterrows():
        # Calculate fit score based on how player addresses team weaknesses
        fit_score = 0
        
        # Weight by blocks contribution (addressing rim protection)
        if 'Blocks_Per_Game' in player and player['Blocks_Per_Game'] > 0:
            mavs_blocks = defense_analysis.loc[defense_analysis.index[0], 'Mavericks']
            league_blocks = defense_analysis.loc[defense_analysis.index[0], 'League_Average']
            if mavs_blocks < league_blocks:
                # Higher score for players who exceed league average in blocks
                if player['Blocks_Per_Game'] > league_blocks:
                    fit_score += 25 * (player['Blocks_Per_Game'] / league_blocks)
        
        # Weight by steals contribution (addressing perimeter defense)
        if 'Steals_Per_Game' in player and player['Steals_Per_Game'] > 0:
            mavs_steals = defense_analysis.loc[defense_analysis.index[1], 'Mavericks']
            league_steals = defense_analysis.loc[defense_analysis.index[1], 'League_Average']
            if mavs_steals < league_steals:
                # Higher score for players who exceed league average in steals
                if player['Steals_Per_Game'] > league_steals:
                    fit_score += 25 * (player['Steals_Per_Game'] / league_steals)
        
        # Weight by rebounding contribution
        if 'Total_Rebounds_Per_Game' in player and player['Total_Rebounds_Per_Game'] > 0:
            mavs_rebounds = defense_analysis.loc[defense_analysis.index[3], 'Mavericks']
            league_rebounds = defense_analysis.loc[defense_analysis.index[3], 'League_Average']
            if mavs_rebounds < league_rebounds:
                # Higher score for players who exceed league average in rebounds
                if player['Total_Rebounds_Per_Game'] > league_rebounds:
                    fit_score += 20 * (player['Total_Rebounds_Per_Game'] / league_rebounds)
        
        # Combine with salary efficiency 
        salary_component = 30 * (player['Defensive_Score'] / player['Sal22_23']) / 1e6
        
        # Combine all factors into final fit score
        final_score = fit_score + salary_component
        fit_scores.append(final_score)
    
    players['Fit_Score'] = fit_scores
    
    return players

def identify_top_targets(players, num_primary=5, num_backup=3):
    """Identify top targets and backup options based on fit score and position."""
    # Filter for players with at least 20 games played
    qualified_players = players[players['Games_Played'] >= 20]
    
    # Sort by fit score
    top_players = qualified_players.sort_values('Fit_Score', ascending=False)
    
    # Get top overall targets
    primary_targets = top_players.head(num_primary)
    
    # Get backup options (next best players excluding primary targets)
    backup_options = top_players.iloc[num_primary:num_primary+num_backup]
    
    # Get underrated gems (high fit score but lower salary)
    underrated = qualified_players.sort_values('Salary_Efficiency', ascending=False).head(5)
    
    # Remove duplicates between primary and underrated
    underrated = underrated[~underrated['Player_Name'].isin(primary_targets['Player_Name'])]
    
    return primary_targets, backup_options, underrated

def format_salary(salary):
    """Format salary in millions with 2 decimal places."""
    if salary >= 1e6:
        return f"${salary/1e6:.2f}M"
    else:
        return f"${salary/1000:.1f}K"

def get_acquisition_method(player):
    """Determine the recommended acquisition method based on player salary and age."""
    salary = player['Sal22_23']
    age = player['Age'] if 'Age' in player else 25
    
    if salary > 20e6:
        return "Trade (significant assets)"
    elif salary > 10e6:
        return "Trade (moderate assets)"
    elif salary > 5e6:
        return "Trade (minor assets)"
    elif age < 25:
        return "Targeted free agent signing"
    else:
        return "Free agent signing"

def generate_recommendations_report(primary_targets, backup_options, underrated, defense_analysis):
    """Generate a formatted recommendation report as text."""
    project_root = Path(__file__).parent.parent
    output_dir = project_root / "output"
    output_file = output_dir / "mavericks_acquisition_recommendations.txt"
    
    # Ensure output directory exists
    output_dir.mkdir(exist_ok=True)
    
    # Extract Mavericks weaknesses for reference
    weaknesses = defense_analysis[defense_analysis['Pct_Difference'] < 0].sort_values('Pct_Difference')
    
    with open(output_file, 'w') as f:
        # Title
        f.write("=" * 80 + "\n")
        f.write("DALLAS MAVERICKS DEFENSIVE PLAYER ACQUISITION RECOMMENDATIONS\n")
        f.write("=" * 80 + "\n\n")
        
        # Summary of defensive weaknesses
        f.write("SUMMARY OF IDENTIFIED DEFENSIVE WEAKNESSES:\n")
        f.write("-" * 50 + "\n")
        
        for idx, row in weaknesses.iterrows():
            metric = idx
            mavs_val = row['Mavericks']
            league_val = row['League_Average']
            pct_diff = abs(row['Pct_Difference'])
            
            f.write(f"- {metric}: {mavs_val:.2f} vs. League {league_val:.2f} ({pct_diff:.1f}% worse)\n")
        
        f.write("\n")
        
        # Section A: Recommended Players
        f.write("SECTION A: RECOMMENDED PLAYER TARGETS\n")
        f.write("-" * 50 + "\n")
        f.write("\nPRIMARY TARGETS:\n")
        
        # Create a formatted table for primary targets
        f.write(f"{'Player':<20} {'Team':<5} {'Position':<5} {'Salary':<10} {'Acquisition Method':<25}\n")
        f.write("-" * 70 + "\n")
        
        for _, player in primary_targets.iterrows():
            name = player['Player_Name']
            team = player['Team']
            position = player['Position']
            salary = format_salary(player['Sal22_23'])
            acquisition = get_acquisition_method(player)
            
            f.write(f"{name:<20} {team:<5} {position:<5} {salary:<10} {acquisition:<25}\n")
        
        # Backup options
        f.write("\nBACKUP OPTIONS:\n")
        f.write(f"{'Player':<20} {'Team':<5} {'Position':<5} {'Salary':<10} {'Acquisition Method':<25}\n")
        f.write("-" * 70 + "\n")
        
        for _, player in backup_options.iterrows():
            name = player['Player_Name']
            team = player['Team']
            position = player['Position']
            salary = format_salary(player['Sal22_23'])
            acquisition = get_acquisition_method(player)
            
            f.write(f"{name:<20} {team:<5} {position:<5} {salary:<10} {acquisition:<25}\n")
        
        # Underrated gems
        f.write("\nUNDERRATED GEMS:\n")
        f.write(f"{'Player':<20} {'Team':<5} {'Position':<5} {'Salary':<10} {'Acquisition Method':<25}\n")
        f.write("-" * 70 + "\n")
        
        for _, player in underrated.iterrows():
            name = player['Player_Name']
            team = player['Team']
            position = player['Position']
            salary = format_salary(player['Sal22_23'])
            acquisition = get_acquisition_method(player)
            
            f.write(f"{name:<20} {team:<5} {position:<5} {salary:<10} {acquisition:<25}\n")
        
        # Section B: Reasoning & Projections
        f.write("\n\nSECTION B: REASONING & PROJECTIONS\n")
        f.write("-" * 50 + "\n")
        
        # Detailed analysis for each primary target
        for _, player in primary_targets.iterrows():
            name = player['Player_Name']
            team = player['Team']
            position = player['Position']
            blocks = player['Blocks_Per_Game']
            steals = player['Steals_Per_Game']
            def_rebounds = player['Defensive_Rebounds_Per_Game']
            tot_rebounds = player['Total_Rebounds_Per_Game']
            salary = format_salary(player['Sal22_23'])
            
            f.write(f"\n{name.upper()} ({team}, {position}) - {salary}\n")
            
            # Defensive impact
            f.write("Defensive Impact:\n")
            
            # Compare to Mavs and league averages
            blocks_impact = blocks - defense_analysis.loc[0, 'Mavericks']
            blocks_vs_league = blocks - defense_analysis.loc[0, 'League_Average']
            
            steals_impact = steals - defense_analysis.loc[1, 'Mavericks']
            steals_vs_league = steals - defense_analysis.loc[1, 'League_Average']
            
            rebounds_impact = tot_rebounds - defense_analysis.loc[3, 'Mavericks']
            rebounds_vs_league = tot_rebounds - defense_analysis.loc[3, 'League_Average']
            
            # Write evaluation
            f.write(f"• Blocks: {blocks:.2f} per game ")
            if blocks_vs_league > 0:
                f.write(f"(+{blocks_vs_league:.2f} vs. league avg, +{blocks_impact:.2f} vs. Mavs)\n")
            else:
                f.write(f"({blocks_vs_league:.2f} vs. league avg, {blocks_impact:.2f} vs. Mavs)\n")
                
            f.write(f"• Steals: {steals:.2f} per game ")
            if steals_vs_league > 0:
                f.write(f"(+{steals_vs_league:.2f} vs. league avg, +{steals_impact:.2f} vs. Mavs)\n")
            else:
                f.write(f"({steals_vs_league:.2f} vs. league avg, {steals_impact:.2f} vs. Mavs)\n")
                
            f.write(f"• Rebounds: {tot_rebounds:.2f} per game ")
            if rebounds_vs_league > 0:
                f.write(f"(+{rebounds_vs_league:.2f} vs. league avg, +{rebounds_impact:.2f} vs. Mavs)\n")
            else:
                f.write(f"({rebounds_vs_league:.2f} vs. league avg, {rebounds_impact:.2f} vs. Mavs)\n")
            
            # Cost analysis
            f.write("\nCost Analysis:\n")
            salary_value = player['Sal22_23']
            if salary_value > 25e6:
                f.write(f"• High-cost acquisition ({salary}) - would require significant trade assets\n")
                f.write("• Consider cap impact and potential multi-team trade scenarios\n")
            elif salary_value > 15e6:
                f.write(f"• Moderate-cost acquisition ({salary}) - would require solid trade package\n")
                f.write("• Reasonable value given defensive impact\n")
            else:
                f.write(f"• Cost-effective acquisition ({salary}) - excellent value for defensive impact\n")
                f.write("• Low financial risk relative to potential defensive improvement\n")
            
            # Team fit
            f.write("\nTeam Fit & Chemistry:\n")
            if position in ['C', 'PF']:
                f.write("• Would significantly bolster frontcourt defense\n")
                f.write("• Addresses rim protection and rebounding weaknesses\n")
            else:
                f.write("• Would improve perimeter defense and disruptive potential\n")
                f.write("• Addresses steals and transition defense weaknesses\n")
            
            # Risk assessment
            f.write("\nUpside & Risk Assessment:\n")
            f.write("• Upside: Immediate improvement to Mavericks' defensive metrics in key weakness areas\n")
            
            if position in ['C', 'PF'] and blocks > 1.0:
                f.write("• Would provide much-needed rim protection\n")
            if steals > 1.0:
                f.write("• Elite steal rate would transform Mavericks' perimeter defense\n")
            if tot_rebounds > 7.0:
                f.write("• Would significantly improve Mavericks' rebounding weakness\n")
            
            if salary_value > 20e6:
                f.write("• Risk: High salary commitment could limit future roster flexibility\n")
            else:
                f.write("• Risk: Minimal financial risk given defensive impact potential\n")
            
            # Recommendation summary
            f.write("\nRecommendation Summary:\n")
            f.write(f"• {name} directly addresses {position}-position defensive needs\n")
            
            primary_strength = ""
            if blocks > 1.0:
                primary_strength = "shot-blocking"
            elif steals > 1.0:
                primary_strength = "perimeter defense"
            elif tot_rebounds > 8.0:
                primary_strength = "rebounding"
            else:
                primary_strength = "overall defense"
            
            f.write(f"• Primary contribution would be {primary_strength}\n")
            
            # Backup alternatives
            if _ == primary_targets.index[0]:  # Only for the top target
                matching_backups = backup_options[backup_options['Position'] == position]
                if not matching_backups.empty:
                    backup_name = matching_backups.iloc[0]['Player_Name']
                    backup_team = matching_backups.iloc[0]['Team'] 
                    backup_salary = format_salary(matching_backups.iloc[0]['Sal22_23'])
                    f.write(f"• If acquisition not feasible, consider {backup_name} ({backup_team}, {backup_salary}) as alternative\n")
            
            f.write("\n" + "-" * 50 + "\n")
        
        # Summary of benefits
        f.write("\nPROJECTED TEAM BENEFIT SUMMARY:\n")
        f.write("-" * 50 + "\n")
        
        # Calculate average improvement in key metrics if top target acquired
        if not primary_targets.empty:
            top_player = primary_targets.iloc[0]
            blocks_improvement = top_player['Blocks_Per_Game'] - defense_analysis.loc[0, 'Mavericks']
            steals_improvement = top_player['Steals_Per_Game'] - defense_analysis.loc[1, 'Mavericks']
            rebounds_improvement = top_player['Total_Rebounds_Per_Game'] - defense_analysis.loc[3, 'Mavericks']
            
            f.write("Expected defensive improvement with acquisition of top target:\n")
            f.write(f"• Blocks: +{blocks_improvement:.2f} per game ({(blocks_improvement/defense_analysis.loc[0, 'Mavericks']*100):.1f}% improvement)\n")
            f.write(f"• Steals: +{steals_improvement:.2f} per game ({(steals_improvement/defense_analysis.loc[1, 'Mavericks']*100):.1f}% improvement)\n")
            f.write(f"• Rebounds: +{rebounds_improvement:.2f} per game ({(rebounds_improvement/defense_analysis.loc[3, 'Mavericks']*100):.1f}% improvement)\n")
        
        f.write("\nBased on the identified defensive weaknesses and recommended player acquisitions,\n")
        f.write("the Mavericks could substantially improve their defensive metrics and address\n")
        f.write("the key weaknesses identified in the analysis. Primary focus should be on\n")
        f.write("improving rim protection, perimeter defense disruption, and rebounding.\n")
    
    return output_file

def main():
    # Load analysis data
    print("Loading analysis data...")
    defense_analysis, available_players, weightings = load_analysis_data()
    
    # Analyze player fit
    print("Analyzing player fit for Mavericks' defensive needs...")
    players_with_fit = analyze_player_fit(available_players, defense_analysis)
    
    # Identify top targets
    print("Identifying top acquisition targets...")
    primary_targets, backup_options, underrated = identify_top_targets(players_with_fit)
    
    # Generate recommendations report
    print("Generating formatted recommendations report...")
    output_file = generate_recommendations_report(primary_targets, backup_options, underrated, defense_analysis)
    
    print(f"\nRecommendations report generated successfully: {output_file}")
    
    # Preview top players
    print("\nTop 5 Recommended Player Targets:")
    for idx, (_, player) in enumerate(primary_targets.iterrows(), 1):
        name = player['Player_Name']
        team = player['Team']
        position = player['Position']
        salary = format_salary(player['Sal22_23'])
        
        print(f"{idx}. {name} ({team}, {position}) - {salary}")

if __name__ == "__main__":
    main() 
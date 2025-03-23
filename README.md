# NBA Defensive Analysis Tool

This project provides a data-driven analysis of NBA team defensive performance, specifically focusing on the Dallas Mavericks' defensive metrics and player acquisition recommendations. The tool analyzes available player statistics to identify defensive weaknesses and recommend appropriate weightings for evaluating potential player acquisitions.

## Project Overview

The analysis focuses on:
1. Identifying defensive weaknesses by comparing team metrics to league averages
2. Calculating statistical significance of performance gaps
3. Recommending weightings for defensive metrics based on team needs
4. Providing player acquisition recommendations based on identified weaknesses

## Data Sources

The analysis uses the following data:
- Player statistics from the 2022-2023 NBA season
- Metrics include blocks, steals, rebounds, and personal fouls
- All data is contained within the provided Excel file (`2025_scc_5a_nba_player_data_2023.xlsx`)

## Methodology

### Defensive Metrics Analysis
- Calculates per-game statistics for key defensive metrics
- Compares team performance against league averages
- Identifies statistically significant performance gaps
- Uses Welch's t-test for statistical significance testing

### Weighting System
- Starts with baseline importance weights based on basketball domain knowledge
- Adjusts weights based on:
  - Magnitude of team weaknesses
  - Statistical significance of performance gaps
- Normalizes final weights to total 100%

## Project Structure

```
├── data/
│   └── 2025_scc_5a_nba_player_data_2023.xlsx
├── src/
│   ├── defensive_analysis.py
│   └── player_recommendations.py
├── output/
│   ├── mavericks_defensive_analysis.xlsx
│   ├── recommended_defensive_weightings.xlsx
│   └── mavericks_acquisition_recommendations.txt
└── requirements.txt
```

## Installation

1. Clone the repository:
```bash
git clone [repository-url]
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the defensive analysis:
```bash
python src/defensive_analysis.py
```

2. Generate player recommendations:
```bash
python src/player_recommendations.py
```

## Output Files

1. `mavericks_defensive_analysis.xlsx`
   - Detailed comparison of Mavericks' defensive metrics vs. league averages
   - Statistical significance testing results

2. `recommended_defensive_weightings.xlsx`
   - Recommended weightings for defensive metrics
   - Rationale for each weighting

3. `mavericks_acquisition_recommendations.txt`
   - List of recommended player targets
   - Detailed analysis of each recommendation
   - Projected impact on team performance

## Dependencies

- pandas >= 1.3.0
- numpy >= 1.20.0
- matplotlib >= 3.4.0
- seaborn >= 0.11.0
- scikit-learn >= 0.24.0
- openpyxl >= 3.0.0

## Limitations

- Analysis is based on 2022-2023 season data only
- Does not include opponent-specific statistics
- Player recommendations are based on statistical analysis only and do not consider:
  - Team chemistry
  - Contract negotiations
  - Player availability
  - Team salary cap situation

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 
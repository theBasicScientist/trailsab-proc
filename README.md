# Trails A/B Analysis System

A comprehensive Python pipeline for analyzing digitized Trails A/B cognitive task data, providing detailed behavioral metrics and visualizations for neuropsychological research.

## Overview

The Trails Making Test (TMT) is a widely used neuropsychological assessment that measures visual attention and task switching. This system processes digitized pen-trace data from Trails A (sequential numbers) and Trails B (alternating numbers/letters) tasks, extracting both basic and advanced behavioral metrics.

## Features

### Core Analysis
- **Basic Metrics**: Completion times, error counts, accuracy rates
- **Advanced Metrics**: 
  - Think time vs ink time separation
  - Drawing speed and smoothness analysis
  - Path optimality ratios
  - Hesitation detection and counting
  - 60Hz temporal resampling for uniform analysis

### Data Processing
- **Multi-core parallel processing** with auto-detection of available CPUs
- Robust error handling with graceful file skipping
- Data quality validation (24-path paradigm requirement)
- Support for both preprocessed and raw data formats
- Consecutive segments approach for accurate think time calculation
- Real-time progress tracking with processing rate and ETA

### Visualizations
- Individual participant path traces with speed color-coding
- Think time visualization via circle shading
- **Spatial density heatmaps** comparing Trail A vs Trail B (side-by-side)
- **Population-level velocity heatmaps** showing movement patterns
- Group-level statistical plots with confidence bands
- Summary dashboards and analysis reports

### Output Formats
- Separate TrailsA and TrailsB summary CSV files
- Item-by-item detailed CSV files per participant
- Comprehensive text analysis reports
- Publication-ready visualization plots

## Installation

### Requirements
- Python 3.12+
- Required packages: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scipy`, `scikit-learn`, `statsmodels`, `pingouin`

### Setup
```bash
git clone https://github.com/SympatiCog/trailsab-proc.git
cd trailsab-proc
pip install -r requirements.txt
# Or using uv:
uv pip install -e .
```

## Usage

### Quick Start
```bash
# Test on a small subset
python test_analysis.py

# Full analysis with visualizations
python run_analysis.py --visualize
```

### Command Line Options
```bash
python run_analysis.py [OPTIONS]

Options:
  --scaled-points PATH        Path to trails_points_scaled.json (default: trails_points_scaled.json)
  --data-dir PATH            Directory with raw trail CSV files (default: trails_responses)
  --preprocessed-dir PATH    Directory with preprocessed CSV files (default: preprocessed)
  --output-dir PATH          Output directory for results (default: output)
  --visualize                Generate visualizations (requires matplotlib)
  --visualize-existing PATH  Generate visualizations from existing output (skips analysis)
  --plots-dest PATH          Destination for plots when using --visualize-existing
  --processes N              Number of parallel processes (default: auto-detect, max 8)
  --no-parallel              Disable parallel processing (use serial mode)
```

### Examples
```bash
# Full analysis with all features (parallel processing + visualizations)
python run_analysis.py --visualize

# Custom parallel processing with 4 workers
python run_analysis.py --visualize --processes 4

# Serial processing (no parallelization)
python run_analysis.py --visualize --no-parallel

# Generate visualizations from existing results
python run_analysis.py --visualize-existing output/ --plots-dest my_plots/
```

### Directory Structure
```
trailsab-proc/
├── trails_analyzer.py      # Core analysis engine
├── trails_visualizer.py    # Visualization components
├── run_analysis.py         # Main pipeline script
├── test_analysis.py        # Testing script
├── trails_points_scaled.csv # Circle positions/radii
├── trails_responses/       # Raw data files (*.csv)
├── preprocessed/          # Preprocessed data files
└── output/               # Analysis results
    ├── summary_statistics_trailsA.csv
    ├── summary_statistics_trailsB.csv
    ├── item_by_item/      # Per-participant detailed files
    ├── visualizations/
    │   ├── individual/    # Per-participant plots
    │   └── group/         # Population-level analyses
    │       ├── spatial_density_heatmap.png  # Trail A vs B side-by-side
    │       ├── velocity_heatmap_*.png       # Movement velocity patterns
    │       └── [statistical plots]
    └── analysis_report.txt
```

## Data Format

### Input Files
- **Raw Data**: CSV files with columns: `x`, `y`, `timestamp`, `correct_path`, etc.
- **Circle Positions**: `trails_points_scaled.csv` with target locations and radii
- **Naming Convention**: Files ending in `trail2.csv` (Trails A) or `trail4.csv` (Trails B)

### Data Quality Requirements
- Exactly 24 unique paths per file (validates paradigm correctness)
- ~60Hz sampling rate (automatically resampled if different)
- Coordinate system: 0-100% screen coordinates

## Key Metrics

### Basic Measures
- **Total Time**: Overall task completion time
- **Error Count**: Number of incorrect connections
- **Accuracy**: Percentage of correct paths

### Advanced Measures
- **Think Time**: Time spent planning within target circles
- **Ink Time**: Time spent drawing between targets
- **Drawing Speed**: Instantaneous and mean drawing velocities
- **Path Optimality**: Actual vs optimal distance ratios
- **Smoothness**: Path curvature variation analysis
- **Hesitations**: Velocity-based pause detection

## Technical Details

### Algorithm Features
- **Consecutive Segments**: Proper think/ink time separation using segment boundaries
- **60Hz Resampling**: Uniform temporal resolution across participants
- **Coordinate Alignment**: Pre-scaled coordinate system (no transformation needed)
- **Error Handling**: Graceful skipping of invalid/incomplete files

### Performance
- **Multi-core parallel processing** for optimal throughput
- Processes hundreds of files in minutes with auto-scaling
- Memory-efficient streaming processing
- Cross-platform compatibility (Windows/macOS/Linux)
- Real-time progress monitoring with processing rates
- Thread-safe design with independent analyzer instances per worker
- Robust to data quality variations

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature-name`)
3. Commit changes (`git commit -am 'Add feature'`)
4. Push to branch (`git push origin feature-name`)
5. Create Pull Request

## License

This project is licensed under the MIT License - see LICENSE file for details.

## Citation

If you use this system in your research, please cite:

```bibtex
[Add appropriate citation when published]
```

## Support

For questions, issues, or feature requests:
- Open an issue on GitHub
- Contact: [Add contact information]

## Acknowledgments

- Built for neuropsychological research applications
- Developed with cognitive assessment best practices
- Optimized for large-scale behavioral data analysis
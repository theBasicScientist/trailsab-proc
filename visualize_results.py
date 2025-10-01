#!/usr/bin/env python3
"""
Standalone visualization script for existing Trails A/B analysis results
Generates visualizations from pre-computed CSV output files
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import argparse
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# Import our existing visualizer (we'll need to modify it)
from trails_analyzer import CircleTarget
from trails_visualizer import TrailsVisualizer


@dataclass
class ReconstructedResult:
    """Reconstructed result object from CSV data for visualization compatibility"""
    participant_id: str
    trail: str
    total_time: float
    total_errors: int
    segments: List
    
    
class CSVVisualizationLoader:
    """Loads and reconstructs data from CSV output files for visualization"""
    
    def __init__(self, output_dir: str, scaled_points_path: str = 'trails_points_scaled.json'):
        self.output_dir = Path(output_dir)
        self.scaled_points_path = scaled_points_path
        self.circles = self._load_circles()
        
    def _load_circles(self) -> Dict[str, Dict[str, CircleTarget]]:
        """Load circle positions from scaled points CSV"""
        circles = {'trail2': {}, 'trail4': {}}
        
        try:
            df = pd.read_csv(self.scaled_points_path)
            
            for trail_screen in ['trail2', 'trail4']:
                trail_circles = df[df['trail_screen'] == trail_screen]
                for _, row in trail_circles.iterrows():
                    label = str(row['label'])
                    circle = CircleTarget(
                        order=len(circles[trail_screen]),
                        cx=row['x'],
                        cy=row['y'], 
                        label=label,
                        radius=row['radius']
                    )
                    circles[trail_screen][label] = circle
                    
        except Exception as e:
            print(f"Error loading circles: {e}")
            
        return circles
    
    def load_summary_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load summary statistics for group visualizations"""
        trails_a_path = self.output_dir / 'summary_statistics_trailsA.csv'
        trails_b_path = self.output_dir / 'summary_statistics_trailsB.csv'
        
        trails_a_df = pd.DataFrame()
        trails_b_df = pd.DataFrame()
        
        if trails_a_path.exists():
            trails_a_df = pd.read_csv(trails_a_path)
            print(f"Loaded {len(trails_a_df)} TrailsA participants")
        else:
            print(f"Warning: {trails_a_path} not found")
            
        if trails_b_path.exists():
            trails_b_df = pd.read_csv(trails_b_path)
            print(f"Loaded {len(trails_b_df)} TrailsB participants")
        else:
            print(f"Warning: {trails_b_path} not found")
            
        return trails_a_df, trails_b_df
    
    def load_participant_data(self, participant_id: str, trail: str) -> Optional[ReconstructedResult]:
        """Load item-by-item data for a specific participant and reconstruct result object"""
        item_file = self.output_dir / 'item_by_item' / f'{participant_id}_{trail}_items.csv'
        
        if not item_file.exists():
            print(f"Warning: {item_file} not found")
            return None
            
        try:
            df = pd.read_csv(item_file)
            if df.empty:
                return None
                
            # Reconstruct segments from CSV data
            segments = []
            for _, row in df.iterrows():
                # Create a simple segment object with the data we need
                segment = type('Segment', (), {
                    'is_error': row['is_error'],
                    'think_time': row['think_time'], 
                    'ink_time': row['ink_time'],
                    'mean_speed': row['mean_speed'],
                    'path_optimality': row['path_optimality'],
                    'smoothness': row['smoothness'],
                    'hesitation_count': row['hesitation_count'],
                    'path': row['path'],
                    'think_circle': str(row['think_circle']),
                    # Add empty coordinate arrays for compatibility
                    'x_coords': np.array([]),
                    'y_coords': np.array([]),
                    'timestamps': np.array([]),
                    'velocities': np.array([])
                })()
                segments.append(segment)
                
            # Calculate total metrics
            total_time = df['think_time'].sum() + df['ink_time'].sum()
            total_errors = int(df['is_error'].sum())
            
            return ReconstructedResult(
                participant_id=participant_id,
                trail=trail,
                total_time=total_time,
                total_errors=total_errors,
                segments=segments
            )
            
        except Exception as e:
            print(f"Error loading {item_file}: {e}")
            return None
    
    def get_available_participants(self) -> List[Tuple[str, str]]:
        """Get list of available participants and trails from item-by-item files"""
        item_dir = self.output_dir / 'item_by_item'
        if not item_dir.exists():
            return []
            
        participants = []
        for file in item_dir.glob('*_items.csv'):
            # Parse filename: participant_trail_items.csv
            name_parts = file.stem.split('_')
            if len(name_parts) >= 3 and name_parts[-1] == 'items':
                trail = name_parts[-2]
                participant = '_'.join(name_parts[:-2])
                participants.append((participant, trail))
                
        return sorted(participants)


class StandaloneVisualizer:
    """Standalone visualizer that works with CSV output data"""
    
    def __init__(self, output_dir: str, scaled_points_path: str = 'trails_points_scaled.json', plots_dest: str = 'plots'):
        self.loader = CSVVisualizationLoader(output_dir, scaled_points_path)
        self.output_dir = Path(output_dir)
        
        # Create visualization output directories
        self.viz_dir = Path(plots_dest)
        self.individual_dir = self.viz_dir / 'individual'
        self.group_dir = self.viz_dir / 'group'
        
        self.viz_dir.mkdir(exist_ok=True)
        self.individual_dir.mkdir(exist_ok=True)
        self.group_dir.mkdir(exist_ok=True)
    
    def plot_group_statistics(self):
        """Generate group-level statistical plots"""
        print("Generating group statistics visualizations...")
        
        trails_a_df, trails_b_df = self.loader.load_summary_data()
        
        if trails_a_df.empty and trails_b_df.empty:
            print("No summary data found. Cannot generate group statistics.")
            return
            
        # Set up the plot style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Create comparison plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Trails A/B Group Statistics Comparison', fontsize=16, fontweight='bold')
        
        # 1. Completion Times
        ax1 = axes[0, 0]
        if not trails_a_df.empty and not trails_b_df.empty:
            times_data = [trails_a_df['total_time'].dropna(), trails_b_df['total_time'].dropna()]
            bp = ax1.boxplot(times_data, labels=['Trails A', 'Trails B'], patch_artist=True)
            bp['boxes'][0].set_facecolor('lightblue')
            bp['boxes'][1].set_facecolor('lightcoral')
        elif not trails_a_df.empty:
            ax1.boxplot([trails_a_df['total_time'].dropna()], labels=['Trails A'], patch_artist=True)
        elif not trails_b_df.empty:
            ax1.boxplot([trails_b_df['total_time'].dropna()], labels=['Trails B'], patch_artist=True)
        
        ax1.set_ylabel('Completion Time (seconds)')
        ax1.set_title('Completion Times')
        ax1.grid(True, alpha=0.3)
        
        # 2. Error Counts  
        ax2 = axes[0, 1]
        if not trails_a_df.empty and not trails_b_df.empty:
            error_data = [trails_a_df['error_count'].dropna(), trails_b_df['error_count'].dropna()]
            bp = ax2.boxplot(error_data, labels=['Trails A', 'Trails B'], patch_artist=True)
            bp['boxes'][0].set_facecolor('lightblue')
            bp['boxes'][1].set_facecolor('lightcoral')
        elif not trails_a_df.empty:
            ax2.boxplot([trails_a_df['error_count'].dropna()], labels=['Trails A'], patch_artist=True)
        elif not trails_b_df.empty:
            ax2.boxplot([trails_b_df['error_count'].dropna()], labels=['Trails B'], patch_artist=True)
        
        ax2.set_ylabel('Error Count')
        ax2.set_title('Error Counts')
        ax2.grid(True, alpha=0.3)
        
        # 3. Think Time vs Ink Time
        ax3 = axes[1, 0]
        if not trails_a_df.empty:
            ax3.scatter(trails_a_df['mean_think_time'], trails_a_df['mean_ink_time'], 
                       alpha=0.6, label='Trails A', color='blue')
        if not trails_b_df.empty:
            ax3.scatter(trails_b_df['mean_think_time'], trails_b_df['mean_ink_time'], 
                       alpha=0.6, label='Trails B', color='red')
        
        ax3.set_xlabel('Mean Think Time (seconds)')
        ax3.set_ylabel('Mean Ink Time (seconds)')
        ax3.set_title('Think Time vs Ink Time')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Speed and Path Optimality
        ax4 = axes[1, 1]
        if not trails_a_df.empty:
            ax4.scatter(trails_a_df['mean_speed'], trails_a_df['mean_path_optimality'], 
                       alpha=0.6, label='Trails A', color='blue')
        if not trails_b_df.empty:
            ax4.scatter(trails_b_df['mean_speed'], trails_b_df['mean_path_optimality'], 
                       alpha=0.6, label='Trails B', color='red')
        
        ax4.set_xlabel('Mean Speed (pixels/second)')
        ax4.set_ylabel('Mean Path Optimality')
        ax4.set_title('Speed vs Path Optimality')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        output_path = self.group_dir / 'group_statistics_comparison.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved group statistics to {output_path}")
        
    def plot_individual_participant(self, participant_id: str, trail: str):
        """Generate individual participant visualization"""
        result = self.loader.load_participant_data(participant_id, trail)
        if not result:
            print(f"Could not load data for {participant_id} {trail}")
            return
            
        # Get circle positions for this trail
        circles = self.loader.circles.get(trail, {})
        if not circles:
            print(f"No circle data found for {trail}")
            return
            
        print(f"Generating individual plot for {participant_id} {trail}...")
        
        # Create figure
        fig, (ax_main, ax_metrics) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Main path plot
        ax_main.set_xlim(0, 100)
        ax_main.set_ylim(0, 100)
        ax_main.set_xlabel('X Position (%)')
        ax_main.set_ylabel('Y Position (%)')
        ax_main.set_title(f'{participant_id} - {trail.upper()} Path Trace')
        ax_main.grid(True, alpha=0.3)
        ax_main.set_aspect('equal')
        
        # Draw circles with think time shading
        for label, circle in circles.items():
            # Find think time for this circle
            think_times = [s.think_time for s in result.segments if s.think_circle == label]
            think_time = think_times[0] if think_times else 0
            
            # Color intensity based on think time
            intensity = min(think_time / 2.0, 1.0)  # Cap at 2 seconds for color scaling
            color = plt.cm.Blues(0.3 + 0.7 * intensity)
            
            circle_plot = plt.Circle((circle.cx, circle.cy), circle.radius, 
                                   color=color, alpha=0.7, ec='black', linewidth=1)
            ax_main.add_patch(circle_plot)
            
            # Add label
            ax_main.text(circle.cx, circle.cy, label, ha='center', va='center', 
                        fontweight='bold', fontsize=10)
        
        # Draw path connections with speed color-coding
        for i, segment in enumerate(result.segments):
            if segment.is_error:
                continue
                
            # Parse path to get start/end circles
            try:
                parts = segment.path.split(' ~ ')
                if len(parts) == 2:
                    start_label, end_label = parts[0].strip(), parts[1].strip()
                    
                    if start_label in circles and end_label in circles:
                        start_circle = circles[start_label]
                        end_circle = circles[end_label]
                        
                        # Color based on speed
                        speed = segment.mean_speed
                        speed_color = plt.cm.plasma(min(speed / 100.0, 1.0))  # Cap at 100 px/s
                        
                        ax_main.plot([start_circle.cx, end_circle.cx], 
                                   [start_circle.cy, end_circle.cy],
                                   color=speed_color, linewidth=3, alpha=0.8)
            except:
                continue
        
        # Add color bars
        sm_think = plt.cm.ScalarMappable(cmap='Blues', norm=plt.Normalize(0, 2))
        sm_think.set_array([])
        cbar_think = fig.colorbar(sm_think, ax=ax_main, shrink=0.6, pad=0.02)
        cbar_think.set_label('Think Time (seconds)', rotation=270, labelpad=15)
        
        # Metrics subplot
        metrics_data = {
            'Total Time': f"{result.total_time:.2f}s",
            'Errors': str(result.total_errors),
            'Avg Think Time': f"{np.mean([s.think_time for s in result.segments]):.3f}s",
            'Avg Ink Time': f"{np.mean([s.ink_time for s in result.segments]):.3f}s",
            'Avg Speed': f"{np.mean([s.mean_speed for s in result.segments]):.1f} px/s",
            'Avg Optimality': f"{np.mean([s.path_optimality for s in result.segments]):.2f}",
            'Avg Smoothness': f"{np.mean([s.smoothness for s in result.segments]):.2f}°",
            'Total Hesitations': str(sum([s.hesitation_count for s in result.segments]))
        }
        
        ax_metrics.axis('off')
        ax_metrics.text(0.1, 0.95, 'Performance Metrics', fontsize=14, fontweight='bold',
                       transform=ax_metrics.transAxes)
        
        y_pos = 0.85
        for metric, value in metrics_data.items():
            ax_metrics.text(0.1, y_pos, f'{metric}:', fontweight='bold',
                           transform=ax_metrics.transAxes)
            ax_metrics.text(0.6, y_pos, value, transform=ax_metrics.transAxes)
            y_pos -= 0.08
        
        plt.tight_layout()
        
        # Save plot
        output_path = self.individual_dir / f'{participant_id}_{trail}_visualization.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved individual plot to {output_path}")
    
    def generate_all_individual_plots(self, max_plots: int = None):
        """Generate plots for all available participants"""
        participants = self.loader.get_available_participants()
        
        if not participants:
            print("No participant data found in item_by_item directory")
            return
            
        print(f"Found {len(participants)} participant trails")
        
        if max_plots:
            participants = participants[:max_plots]
            print(f"Limiting to first {max_plots} participants")
        
        for i, (participant_id, trail) in enumerate(participants, 1):
            print(f"Processing {i}/{len(participants)}: {participant_id} {trail}")
            try:
                self.plot_individual_participant(participant_id, trail)
            except Exception as e:
                print(f"Error processing {participant_id} {trail}: {e}")
                continue
    
    def generate_summary_report(self):
        """Generate a text summary of the visualization outputs"""
        trails_a_df, trails_b_df = self.loader.load_summary_data()
        participants = self.loader.get_available_participants()
        
        report_path = self.viz_dir / 'visualization_summary.txt'
        
        with open(report_path, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("TRAILS A/B VISUALIZATION SUMMARY\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 60 + "\n\n")
            
            f.write("DATA SUMMARY\n")
            f.write("-" * 30 + "\n")
            f.write(f"Trails A participants: {len(trails_a_df)}\n")
            f.write(f"Trails B participants: {len(trails_b_df)}\n")
            f.write(f"Individual trails available: {len(participants)}\n\n")
            
            f.write("GENERATED VISUALIZATIONS\n")
            f.write("-" * 30 + "\n")
            
            if (self.group_dir / 'group_statistics_comparison.png').exists():
                f.write("✓ Group statistics comparison plot\n")
            
            individual_plots = list(self.individual_dir.glob('*_visualization.png'))
            f.write(f"✓ {len(individual_plots)} individual participant plots\n\n")
            
            f.write("OUTPUT LOCATIONS\n")
            f.write("-" * 30 + "\n")
            f.write(f"Group plots: {self.group_dir}\n")
            f.write(f"Individual plots: {self.individual_dir}\n")
            f.write(f"This report: {report_path}\n")
        
        print(f"Summary report saved to {report_path}")


def main():
    parser = argparse.ArgumentParser(description='Generate visualizations from existing Trails A/B analysis results')
    
    parser.add_argument('output_dir', type=str,
                       help='Directory containing analysis output files (with summary CSVs and item_by_item folder)')
    
    parser.add_argument('--scaled-points', type=str, default='trails_points_scaled.json',
                       help='Path to trails_points_scaled.json with circle locations')
    
    parser.add_argument('--group-only', action='store_true',
                       help='Generate only group-level statistics (skip individual plots)')
    
    parser.add_argument('--individual-only', action='store_true',
                       help='Generate only individual plots (skip group statistics)')
    
    parser.add_argument('--participant', type=str,
                       help='Generate plot for specific participant only (format: participant_id:trail)')
    
    parser.add_argument('--max-individual', type=int,
                       help='Maximum number of individual plots to generate')
    
    parser.add_argument('--plots-dest', type=str, default='plots',
                       help='Directory where plots should be saved (default: plots)')
    
    args = parser.parse_args()
    
    # Validate output directory
    output_path = Path(args.output_dir)
    if not output_path.exists():
        print(f"Error: Output directory {output_path} does not exist")
        return 1
    
    print("=" * 60)
    print("TRAILS A/B RESULTS VISUALIZATION")
    print("=" * 60)
    print()
    
    # Initialize visualizer
    visualizer = StandaloneVisualizer(args.output_dir, args.scaled_points, args.plots_dest)
    
    # Generate visualizations based on arguments
    if args.participant:
        # Single participant
        try:
            participant_id, trail = args.participant.split(':')
            visualizer.plot_individual_participant(participant_id, trail)
        except ValueError:
            print("Error: --participant format should be 'participant_id:trail' (e.g., 'M10965682:trail2')")
            return 1
            
    elif args.group_only:
        # Group statistics only
        visualizer.plot_group_statistics()
        
    elif args.individual_only:
        # Individual plots only
        visualizer.generate_all_individual_plots(args.max_individual)
        
    else:
        # Generate both group and individual
        visualizer.plot_group_statistics()
        visualizer.generate_all_individual_plots(args.max_individual)
    
    # Generate summary report
    visualizer.generate_summary_report()
    
    print()
    print("=" * 60)
    print("VISUALIZATION COMPLETE!")
    print("=" * 60)


if __name__ == "__main__":
    sys.exit(main())
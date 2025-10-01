#!/usr/bin/env python3
"""
Main pipeline script for Trails A/B Analysis
Processes all participant data and generates outputs
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
import argparse
from multiprocessing import Pool, cpu_count
from functools import partial
import time

# Import our modules
from trails_analyzer import TrailsAnalyzer
from trails_visualizer import TrailsVisualizer, visualize_all


def process_single_participant(filepath_and_config):
    """
    Worker function to process a single participant file
    
    Args:
        filepath_and_config: Tuple of (filepath, analyzer_config_dict)
        
    Returns:
        Tuple of (result_dict, filepath) or (None, filepath) if failed
    """
    filepath, config = filepath_and_config
    
    try:
        # Create analyzer instance for this worker (thread-safe)
        analyzer = TrailsAnalyzer(
            scaled_points_path=config['scaled_points_path'],
            data_dir=config['data_dir'],
            preprocessed_dir=config['preprocessed_dir'],
            output_dir=config['output_dir']
        )
        
        result = analyzer.process_participant(str(filepath))
        return result, str(filepath)
        
    except Exception as e:
        # Return error info for debugging
        return None, str(filepath)


def process_participants_parallel(all_files, analyzer, scaled_points_path, n_processes=None):
    """
    Process participant files in parallel
    
    Args:
        all_files: List of file paths to process
        analyzer: TrailsAnalyzer instance (for configuration)
        n_processes: Number of parallel processes (None = auto-detect)
        
    Returns:
        Tuple of (all_results, participant_results, skipped_count)
    """
    if n_processes is None:
        n_processes = min(cpu_count(), len(all_files), 8)  # Cap at 8 to avoid overwhelming
    
    print(f"Processing {len(all_files)} files using {n_processes} parallel processes...")
    
    # Prepare configuration for workers
    config = {
        'scaled_points_path': scaled_points_path,
        'data_dir': str(analyzer.data_dir),
        'preprocessed_dir': str(analyzer.preprocessed_dir),
        'output_dir': str(analyzer.output_dir)
    }
    
    # Create input tuples for workers
    worker_inputs = [(filepath, config) for filepath in all_files]
    
    all_results = []
    participant_results = {}
    skipped_count = 0
    processed_count = 0
    
    start_time = time.time()
    
    # Process in parallel with progress reporting
    with Pool(processes=n_processes) as pool:
        # Use imap for progress tracking
        for result, filepath in pool.imap(process_single_participant, worker_inputs):
            processed_count += 1
            
            if result:
                all_results.append(result)
                
                # Group results by participant
                pid = result['participant_id']
                if pid not in participant_results:
                    participant_results[pid] = {}
                participant_results[pid][result['trail']] = result
            else:
                skipped_count += 1
            
            # Progress reporting
            if processed_count % 25 == 0 or processed_count == len(all_files):
                elapsed = time.time() - start_time
                rate = processed_count / elapsed if elapsed > 0 else 0
                remaining = (len(all_files) - processed_count) / rate if rate > 0 else 0
                
                print(f"  Processed {processed_count}/{len(all_files)} files "
                      f"({rate:.1f} files/sec, ~{remaining:.0f}s remaining)")
    
    elapsed_total = time.time() - start_time
    print(f"Parallel processing completed in {elapsed_total:.1f} seconds "
          f"({len(all_files)/elapsed_total:.1f} files/sec average)")
    
    return all_results, participant_results, skipped_count


def create_analysis_report(analyzer, all_results, participant_results):
    """
    Create a comprehensive analysis report
    
    Args:
        analyzer: TrailsAnalyzer instance
        all_results: List of all results
        participant_results: Dictionary grouped by participant
    """
    report_path = analyzer.output_dir / 'analysis_report.txt'
    
    with open(report_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("TRAILS A/B TASK ANALYSIS REPORT\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n\n")
        
        # Overall statistics
        f.write("OVERALL STATISTICS\n")
        f.write("-" * 40 + "\n")
        
        trail2_results = [r for r in all_results if r and r['trail'] == 'trail2']
        trail4_results = [r for r in all_results if r and r['trail'] == 'trail4']
        
        f.write(f"Total participants processed: {len(participant_results)}\n")
        f.write(f"Trail A (Sequential) sessions: {len(trail2_results)}\n")
        f.write(f"Trail B (Alternating) sessions: {len(trail4_results)}\n\n")
        
        # Trail A statistics
        if trail2_results:
            f.write("TRAIL A (Sequential Numbers) - Summary\n")
            f.write("-" * 40 + "\n")
            
            times = [r['total_time'] for r in trail2_results]
            errors = [r['total_errors'] for r in trail2_results]
            
            f.write(f"Completion Time: {np.mean(times):.2f} ± {np.std(times):.2f} seconds\n")
            f.write(f"Error Count: {np.mean(errors):.2f} ± {np.std(errors):.2f}\n")
            f.write(f"Participants with errors: {sum(1 for e in errors if e > 0)}/{len(errors)}\n\n")
            
            # Advanced metrics
            think_times = []
            ink_times = []
            speeds = []
            smoothness_vals = []
            hesitations = []
            
            for result in trail2_results:
                correct_segs = [s for s in result['segments'] if not s.is_error]
                if correct_segs:
                    think_times.extend([s.think_time for s in correct_segs])
                    ink_times.extend([s.ink_time for s in correct_segs])
                    speeds.extend([s.mean_speed for s in correct_segs if s.mean_speed > 0])
                    smoothness_vals.extend([s.smoothness for s in correct_segs])
                    hesitations.extend([s.hesitation_count for s in correct_segs])
            
            if think_times:
                f.write(f"Think Time per circle: {np.mean(think_times):.3f} ± {np.std(think_times):.3f} seconds\n")
            if ink_times:
                f.write(f"Ink Time per line: {np.mean(ink_times):.3f} ± {np.std(ink_times):.3f} seconds\n")
            if speeds:
                f.write(f"Mean drawing speed: {np.mean(speeds):.1f} ± {np.std(speeds):.1f} pixels/second\n")
            if smoothness_vals:
                f.write(f"Path smoothness: {np.mean(smoothness_vals):.2f} ± {np.std(smoothness_vals):.2f} degrees\n")
            if hesitations:
                f.write(f"Hesitations per line: {np.mean(hesitations):.2f} ± {np.std(hesitations):.2f}\n")
            f.write("\n")
        
        # Trail B statistics
        if trail4_results:
            f.write("TRAIL B (Alternating Numbers/Letters) - Summary\n")
            f.write("-" * 40 + "\n")
            
            times = [r['total_time'] for r in trail4_results]
            errors = [r['total_errors'] for r in trail4_results]
            
            f.write(f"Completion Time: {np.mean(times):.2f} ± {np.std(times):.2f} seconds\n")
            f.write(f"Error Count: {np.mean(errors):.2f} ± {np.std(errors):.2f}\n")
            f.write(f"Participants with errors: {sum(1 for e in errors if e > 0)}/{len(errors)}\n\n")
            
            # Advanced metrics
            think_times = []
            ink_times = []
            speeds = []
            smoothness_vals = []
            hesitations = []
            
            for result in trail4_results:
                correct_segs = [s for s in result['segments'] if not s.is_error]
                if correct_segs:
                    think_times.extend([s.think_time for s in correct_segs])
                    ink_times.extend([s.ink_time for s in correct_segs])
                    speeds.extend([s.mean_speed for s in correct_segs if s.mean_speed > 0])
                    smoothness_vals.extend([s.smoothness for s in correct_segs])
                    hesitations.extend([s.hesitation_count for s in correct_segs])
            
            if think_times:
                f.write(f"Think Time per circle: {np.mean(think_times):.3f} ± {np.std(think_times):.3f} seconds\n")
            if ink_times:
                f.write(f"Ink Time per line: {np.mean(ink_times):.3f} ± {np.std(ink_times):.3f} seconds\n")
            if speeds:
                f.write(f"Mean drawing speed: {np.mean(speeds):.1f} ± {np.std(speeds):.1f} pixels/second\n")
            if smoothness_vals:
                f.write(f"Path smoothness: {np.mean(smoothness_vals):.2f} ± {np.std(smoothness_vals):.2f} degrees\n")
            if hesitations:
                f.write(f"Hesitations per line: {np.mean(hesitations):.2f} ± {np.std(hesitations):.2f}\n")
            f.write("\n")
        
        # Comparison between conditions
        if trail2_results and trail4_results:
            f.write("CONDITION COMPARISON (Trail B - Trail A)\n")
            f.write("-" * 40 + "\n")
            
            a_times = [r['total_time'] for r in trail2_results]
            b_times = [r['total_time'] for r in trail4_results]
            
            f.write(f"Time difference: {np.mean(b_times) - np.mean(a_times):.2f} seconds\n")
            f.write(f"Time ratio (B/A): {np.mean(b_times) / np.mean(a_times):.2f}\n")
            
            a_errors = [r['total_errors'] for r in trail2_results]
            b_errors = [r['total_errors'] for r in trail4_results]
            
            f.write(f"Error difference: {np.mean(b_errors) - np.mean(a_errors):.2f}\n")
            f.write(f"Error rate Trail A: {100 * sum(1 for e in a_errors if e > 0) / len(a_errors):.1f}%\n")
            f.write(f"Error rate Trail B: {100 * sum(1 for e in b_errors if e > 0) / len(b_errors):.1f}%\n")
            f.write("\n")
        
        # Output files generated
        f.write("OUTPUT FILES GENERATED\n")
        f.write("-" * 40 + "\n")
        f.write(f"TrailsA summary: {analyzer.output_dir / 'summary_statistics_trailsA.csv'}\n")
        f.write(f"TrailsB summary: {analyzer.output_dir / 'summary_statistics_trailsB.csv'}\n")
        f.write(f"Item-by-item CSVs: {analyzer.output_dir / 'item_by_item'}/*.csv\n")
        f.write(f"Individual visualizations: {analyzer.output_dir / 'visualizations/individual'}/*.png\n")
        f.write(f"Group visualizations: {analyzer.output_dir / 'visualizations/group'}/*.png\n")
        f.write(f"This report: {report_path}\n")
        
    print(f"Analysis report saved to {report_path}")


def main(args):
    """Main execution function"""
    
    print("=" * 60)
    print("TRAILS A/B TASK ANALYSIS PIPELINE")
    print("=" * 60)
    print()
    
    # Initialize analyzer
    print("Initializing analyzer...")
    analyzer = TrailsAnalyzer(
        scaled_points_path=args.scaled_points,
        data_dir=args.data_dir,
        preprocessed_dir=args.preprocessed_dir,
        output_dir=args.output_dir
    )
    
    # Find all participant files
    all_files = []
    
    # Check both directories and combine unique files
    processed_ids = set()
    
    # Check preprocessed directory first (priority)
    if analyzer.preprocessed_dir.exists():
        preprocessed_files = list(analyzer.preprocessed_dir.glob('*trail2.csv'))
        preprocessed_files.extend(analyzer.preprocessed_dir.glob('*trail4.csv'))
        for f in preprocessed_files:
            all_files.append(f)
            # Track processed IDs to avoid duplicates
            processed_ids.add(f.name)
        if preprocessed_files:
            print(f"Found {len(preprocessed_files)} files in preprocessed directory")
    
    # Also check trails_responses directory
    if analyzer.data_dir.exists():
        data_files = list(analyzer.data_dir.glob('*trail2.csv'))
        data_files.extend(analyzer.data_dir.glob('*trail4.csv'))
        # Add files not already processed
        added = 0
        for f in data_files:
            if f.name not in processed_ids:
                all_files.append(f)
                added += 1
        if added > 0:
            print(f"Found {added} additional files in data directory")
    
    if not all_files:
        print("ERROR: No trail files found to process!")
        return
    
    print(f"Total files to process: {len(all_files)}")
    print()
    
    # Process each participant (with parallelization)
    print("Processing participant data...")
    
    if args.no_parallel or len(all_files) < 4:
        # Use serial processing for small datasets or if explicitly disabled
        print("Using serial processing...")
        all_results = []
        participant_results = {}
        skipped_count = 0
        
        for i, filepath in enumerate(all_files, 1):
            if i % 10 == 0:
                print(f"  Processed {i}/{len(all_files)} files...")
            
            result = analyzer.process_participant(str(filepath))
            
            if result:
                all_results.append(result)
                
                # Group results by participant
                pid = result['participant_id']
                if pid not in participant_results:
                    participant_results[pid] = {}
                participant_results[pid][result['trail']] = result
            else:
                skipped_count += 1
    else:
        # Use parallel processing
        all_results, participant_results, skipped_count = process_participants_parallel(
            all_files, analyzer, args.scaled_points, args.processes
        )
    
    print(f"Successfully processed {len(all_results)} trail sessions")
    print(f"From {len(participant_results)} unique participants")
    print()
    
    # Generate summary statistics
    print("Generating summary statistics...")
    trails_a_df, trails_b_df = analyzer.generate_summary_statistics(all_results)
    
    # Save TrailsA summary
    if not trails_a_df.empty:
        trails_a_output = analyzer.output_dir / 'summary_statistics_trailsA.csv'
        trails_a_df.to_csv(trails_a_output, index=False)
        print(f"  TrailsA saved to {trails_a_output}")
    
    # Save TrailsB summary
    if not trails_b_df.empty:
        trails_b_output = analyzer.output_dir / 'summary_statistics_trailsB.csv'
        trails_b_df.to_csv(trails_b_output, index=False)
        print(f"  TrailsB saved to {trails_b_output}")
    
    # Generate item-by-item CSVs
    print("Generating item-by-item CSVs...")
    item_dir = analyzer.output_dir / 'item_by_item'
    item_dir.mkdir(exist_ok=True)
    
    for pid, trails in participant_results.items():
        for trail_id, result in trails.items():
            output_path = item_dir / f"{pid}_{trail_id}_items.csv"
            analyzer.generate_item_by_item_csv(result, str(output_path))
    
    print(f"  Saved {len(participant_results)} participant files to {item_dir}")
    print()
    
    # Generate visualizations if requested
    if args.visualize:
        print("Generating visualizations...")
        # Use same parallel settings as analysis for visualizations
        parallel_viz = not args.no_parallel
        n_processes_viz = args.processes if not args.no_parallel else None
        visualize_all(analyzer, all_results, participant_results, 
                     parallel=parallel_viz, n_processes=n_processes_viz)
        print()
    
    # Create analysis report
    print("Creating analysis report...")
    create_analysis_report(analyzer, all_results, participant_results)
    
    print()
    print("=" * 60)
    print("ANALYSIS COMPLETE!")
    print("=" * 60)
    
    return analyzer, all_results, participant_results


if __name__ == "__main__":
    # Ensure multiprocessing works correctly on all platforms
    import multiprocessing
    multiprocessing.set_start_method('spawn', force=True)
    
    parser = argparse.ArgumentParser(description='Trails A/B Task Analysis Pipeline')
    
    parser.add_argument('--scaled-points', type=str, default='trails_points_scaled.json',
                       help='Path to trails_points_scaled.json with circle locations')
    
    parser.add_argument('--data-dir', type=str, default='trails_responses',
                       help='Directory containing raw trail response CSV files')
    
    parser.add_argument('--preprocessed-dir', type=str, default='preprocessed',
                       help='Directory containing preprocessed CSV files')
    
    parser.add_argument('--output-dir', type=str, default='output',
                       help='Directory for analysis outputs')
    
    parser.add_argument('--visualize', action='store_true',
                       help='Generate visualizations (requires matplotlib)')
    
    parser.add_argument('--visualize-existing', type=str, metavar='OUTPUT_DIR',
                       help='Generate visualizations from existing analysis output directory (skips analysis)')
    
    parser.add_argument('--plots-dest', type=str, default='plots',
                       help='Directory where plots should be saved when using --visualize-existing (default: plots)')
    
    parser.add_argument('--processes', type=int, default=None,
                       help='Number of parallel processes to use (default: auto-detect, max 8)')
    
    parser.add_argument('--no-parallel', action='store_true',
                       help='Disable parallel processing (use serial processing)')
    
    args = parser.parse_args()
    
    # Handle visualization-only mode
    if args.visualize_existing:
        print("=== VISUALIZATION MODE ===")
        print(f"Generating visualizations from: {args.visualize_existing}")
        
        # Import and run standalone visualizer
        from visualize_results import StandaloneVisualizer
        visualizer = StandaloneVisualizer(args.visualize_existing, args.scaled_points, args.plots_dest)
        visualizer.plot_group_statistics()
        visualizer.generate_all_individual_plots()
        visualizer.generate_summary_report()
        
        print("Visualization complete!")
        sys.exit(0)
    
    # Run analysis
    analyzer, all_results, participant_results = main(args)
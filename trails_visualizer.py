"""
Visualization module for Trails A/B Task Analysis
Creates individual and group-level visualizations with speed/timing information
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import LineCollection
import matplotlib.colors as mcolors
import seaborn as sns
from scipy.ndimage import gaussian_filter
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import json
from trails_analyzer import TrailsAnalyzer, LineSegment, CircleTarget
from multiprocessing import Pool, cpu_count
import multiprocessing
import time


class TrailsVisualizer:
    """Visualization class for Trails data"""
    
    def __init__(self, analyzer: TrailsAnalyzer):
        """
        Initialize visualizer with analyzer instance
        
        Args:
            analyzer: TrailsAnalyzer instance with loaded configuration
        """
        self.analyzer = analyzer
        self.config = analyzer.config  # Compatibility config created from circles
        self.circles = analyzer.circles
        
        # Set style
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
        
        # Create output directories
        self.vis_dir = Path('output/visualizations')
        self.vis_dir.mkdir(parents=True, exist_ok=True)
        
        self.individual_dir = self.vis_dir / 'individual'
        self.individual_dir.mkdir(exist_ok=True)
        
        self.group_dir = self.vis_dir / 'group'
        self.group_dir.mkdir(exist_ok=True)
    
    def plot_individual_participant(self, result: Dict, save_path: Optional[str] = None):
        """
        Create visualization for individual participant data
        
        Args:
            result: Participant result dictionary from analyzer
            save_path: Optional path to save figure
        """
        if result is None or 'segments' not in result:
            return
        
        trail_id = result['trail']
        circles = self.circles[trail_id]
        config = self.config[trail_id]
        
        # Create figure with subplots for velocity profile
        fig = plt.figure(figsize=(16, 10))
        
        # Main trail plot
        ax_main = plt.subplot2grid((3, 2), (0, 0), colspan=2, rowspan=2)
        
        # Velocity profile plot
        ax_velocity = plt.subplot2grid((3, 2), (2, 0), colspan=2)
        
        # Draw circles with think time color coding
        self._draw_circles_with_think_time(ax_main, circles, result['segments'], config)
        
        # Draw lines with speed color coding
        self._draw_lines_with_speed(ax_main, result['segments'])
        
        # Draw velocity profile
        self._draw_velocity_profile(ax_velocity, result['segments'])
        
        # Set main plot properties
        ax_main.set_aspect('equal')
        ax_main.set_xlim(0, 100)
        ax_main.set_ylim(0, 100)  # No inversion needed - coordinates are already correct
        ax_main.set_xlabel('X Position (%)')
        ax_main.set_ylabel('Y Position (%)')
        
        # Add title with participant info
        condition = 'A (Sequential)' if trail_id == 'trail2' else 'B (Alternating)'
        ax_main.set_title(
            f"Participant: {result['participant_id']} - Trail {condition}\n"
            f"Total Time: {result['total_time']:.2f}s | Errors: {result['total_errors']}",
            fontsize=14, fontweight='bold'
        )
        
        # Add legend for main plot
        self._add_main_legend(ax_main)
        
        # Set velocity plot properties
        ax_velocity.set_xlabel('Time (seconds)')
        ax_velocity.set_ylabel('Velocity (% screen/second)')
        ax_velocity.set_title('Velocity Profile Throughout Task')
        ax_velocity.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        else:
            plt.savefig(
                self.individual_dir / f"{result['participant_id']}_{trail_id}.png",
                dpi=150, bbox_inches='tight'
            )
        
        plt.close()
    
    def _draw_circles_with_think_time(self, ax, circles: Dict[str, CircleTarget], 
                                     segments: List[LineSegment], config: Dict):
        """Draw circles with color coding based on think time"""
        # Calculate think times for each circle
        circle_think_times = {}
        for segment in segments:
            if not segment.is_error and segment.start_label in circles:
                if segment.start_label not in circle_think_times:
                    circle_think_times[segment.start_label] = []
                circle_think_times[segment.start_label].append(segment.think_time)
        
        # Calculate mean think time for each circle
        mean_think_times = {}
        for label, times in circle_think_times.items():
            mean_think_times[label] = np.mean(times) if times else 0
        
        # Normalize think times for color mapping
        if mean_think_times:
            max_think = max(mean_think_times.values())
            min_think = min(mean_think_times.values())
            if max_think > min_think:
                norm = mcolors.Normalize(vmin=min_think, vmax=max_think)
            else:
                norm = mcolors.Normalize(vmin=0, vmax=1)
        else:
            norm = mcolors.Normalize(vmin=0, vmax=1)
        
        cmap = plt.cm.YlOrRd  # Yellow to Red for think time
        
        # Draw each circle
        for label, circle in circles.items():
            think_time = mean_think_times.get(label, 0)
            color = cmap(norm(think_time))
            
            # Draw circle
            circle_patch = patches.Circle(
                (circle.cx, circle.cy), circle.radius,
                facecolor=color, edgecolor='black', linewidth=1.5, alpha=0.7
            )
            ax.add_patch(circle_patch)
            
            # Add label
            ax.text(circle.cx, circle.cy, circle.label,
                   ha='center', va='center', fontsize=10, fontweight='bold')
            
            # Add think time annotation
            if think_time > 0:
                ax.text(circle.cx, circle.cy + circle.radius + 5,
                       f'{think_time:.2f}s',
                       ha='center', va='top', fontsize=8, alpha=0.7)
    
    def _draw_lines_with_speed(self, ax, segments: List[LineSegment]):
        """Draw lines with felt-tip marker effect: variable width based on speed and dots for pauses"""
        # Collect all velocities for normalization
        all_velocities = []
        for segment in segments:
            if not segment.is_error and segment.velocities:
                all_velocities.extend(segment.velocities)
        
        if all_velocities:
            v_min, v_max = np.percentile(all_velocities, [5, 95])
            norm = mcolors.Normalize(vmin=v_min, vmax=v_max)
            
            # Calculate pause threshold (bottom 10th percentile of speeds)
            pause_threshold = np.percentile(all_velocities, 10)
            
            # Calculate reference speed for line width scaling
            median_speed = np.median(all_velocities)
        else:
            norm = mcolors.Normalize(vmin=0, vmax=100)
            pause_threshold = 10
            median_speed = 50
        
        cmap = plt.cm.viridis  # Use viridis for speed
        
        # Parameters for felt-tip marker effect
        base_width = 1.5  # Base line width
        max_width = 8.0   # Maximum width for very slow movement
        min_width = 0.8   # Minimum width for fast movement
        
        # Draw each segment
        for segment in segments:
            if len(segment.points) < 2:
                continue
            
            points = segment.points[['x', 'y']].values
            times = segment.points['seconds'].values if 'seconds' in segment.points else None
            
            if segment.is_error:
                # Draw error lines in red with dashed style
                ax.plot(points[:, 0], points[:, 1], 'r--', alpha=0.5, linewidth=1)
            else:
                # Draw correct lines with felt-tip marker effect
                if segment.velocities and len(points) > 1:
                    # Create line segments for color mapping and variable width
                    segments_array = []
                    colors = []
                    widths = []
                    pause_points = []  # Store pause locations
                    pause_durations = []  # Store pause durations
                    
                    for i in range(len(points) - 1):
                        segments_array.append([points[i], points[i + 1]])
                        
                        # Get velocity for this segment
                        if i < len(segment.velocities):
                            velocity = segment.velocities[i]
                        else:
                            velocity = segment.mean_speed
                        
                        colors.append(velocity)
                        
                        # Calculate line width inversely proportional to speed
                        # Slower movement = wider line (like ink pooling)
                        if velocity > 0:
                            # Inverse relationship: slower = wider
                            width_factor = median_speed / velocity
                            width = base_width * min(max(width_factor, 0.5), 5.0)
                            width = min(max_width, max(min_width, width))
                        else:
                            width = max_width
                        
                        widths.append(width)
                        
                        # Detect pauses for dot overlay
                        if velocity < pause_threshold:
                            # Store midpoint of slow segment
                            pause_x = (points[i][0] + points[i + 1][0]) / 2
                            pause_y = (points[i][1] + points[i + 1][1]) / 2
                            pause_points.append([pause_x, pause_y])
                            
                            # Calculate duration if time data available
                            if times is not None and i + 1 < len(times):
                                duration = times[i + 1] - times[i]
                                pause_durations.append(duration)
                            else:
                                pause_durations.append(0.1)  # Default duration
                    
                    # Create LineCollection with variable widths
                    lc = LineCollection(segments_array, norm=norm, cmap=cmap,
                                      linewidths=widths, alpha=0.8)
                    lc.set_array(np.array(colors))
                    ax.add_collection(lc)
                    
                    # Add pause dots (ink pooling effect)
                    if pause_points:
                        pause_points = np.array(pause_points)
                        # Size dots based on pause duration
                        dot_sizes = np.array(pause_durations) * 200  # Scale for visibility
                        dot_sizes = np.clip(dot_sizes, 20, 200)  # Limit size range
                        
                        # Draw dots with slight transparency
                        ax.scatter(pause_points[:, 0], pause_points[:, 1],
                                 s=dot_sizes, c='darkblue', alpha=0.4,
                                 edgecolors='none', zorder=5)
                else:
                    # Fallback to simple line
                    ax.plot(points[:, 0], points[:, 1], 'b-', alpha=0.6, linewidth=1.5)
    
    def _draw_velocity_profile(self, ax, segments: List[LineSegment]):
        """Draw velocity profile over time"""
        current_time = 0
        
        for segment in segments:
            if not segment.velocities:
                current_time += segment.ink_time + segment.think_time
                continue
            
            # Create time array for this segment
            dt = (segment.ink_time) / len(segment.velocities) if segment.velocities else 0
            times = current_time + np.arange(len(segment.velocities)) * dt
            
            if segment.is_error:
                ax.plot(times, segment.velocities, 'r-', alpha=0.3, linewidth=0.8,
                       label='Error' if segment.line_number == 0 else '')
            else:
                ax.plot(times, segment.velocities, 'b-', alpha=0.7, linewidth=1,
                       label='Correct' if segment.line_number == 0 else '')
            
            # Mark hesitations
            if segment.hesitation_count > 0 and not segment.is_error:
                threshold = np.percentile(segment.velocities, 20)
                hesitation_mask = np.array(segment.velocities) < threshold
                hesitation_times = times[hesitation_mask]
                hesitation_velocities = np.array(segment.velocities)[hesitation_mask]
                ax.scatter(hesitation_times, hesitation_velocities, 
                          c='orange', s=10, alpha=0.5, marker='o')
            
            current_time += segment.ink_time + segment.think_time
        
        # Add legend
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            by_label = dict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys())
    
    def _add_main_legend(self, ax):
        """Add legend explaining color coding and felt-tip marker effect"""
        from matplotlib.lines import Line2D
        
        legend_elements = [
            Line2D([0], [0], color='b', linewidth=2, label='Correct Path'),
            Line2D([0], [0], color='r', linewidth=2, linestyle='--', label='Error Path'),
            patches.Patch(facecolor=plt.cm.YlOrRd(0.2), label='Low Think Time'),
            patches.Patch(facecolor=plt.cm.YlOrRd(0.8), label='High Think Time'),
            Line2D([0], [0], color=plt.cm.viridis(0.2), linewidth=6, label='Slow (Wide Line)'),
            Line2D([0], [0], color=plt.cm.viridis(0.8), linewidth=1, label='Fast (Thin Line)'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='darkblue', 
                   markersize=8, alpha=0.4, label='Pause/Hesitation'),
        ]
        
        ax.legend(handles=legend_elements, loc='upper right', fontsize=9)
    
    def plot_group_data(self, all_results: List[Dict], trail_id: str, 
                       save_path: Optional[str] = None):
        """
        Create group-level visualization with confidence bands
        
        Args:
            all_results: List of all participant results
            trail_id: 'trail2' or 'trail4'
            save_path: Optional path to save figure
        """
        # Filter results for specific trail
        trail_results = [r for r in all_results if r and r['trail'] == trail_id]
        
        if not trail_results:
            return
        
        circles = self.circles[trail_id]
        config = self.config[trail_id]
        
        fig, (ax_main, ax_stats) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Draw circles
        for label, circle in circles.items():
            circle_patch = patches.Circle(
                (circle.cx, circle.cy), circle.radius,
                facecolor='lightgray', edgecolor='black', linewidth=1.5, alpha=0.5
            )
            ax_main.add_patch(circle_patch)
            ax_main.text(circle.cx, circle.cy, circle.label,
                        ha='center', va='center', fontsize=10, fontweight='bold')
        
        # Collect paths for each connection
        connection_paths = {}
        
        for result in trail_results:
            for segment in result['segments']:
                if segment.is_error:
                    continue
                
                key = f"{segment.start_label}_{segment.end_label}"
                if key not in connection_paths:
                    connection_paths[key] = {
                        'paths': [],
                        'speeds': [],
                        'think_times': []
                    }
                
                # Store resampled path
                if len(segment.points) > 1:
                    # Resample to fixed number of points for averaging
                    n_points = 50
                    points = segment.points[['x', 'y']].values
                    if len(points) >= 2:
                        t = np.linspace(0, 1, len(points))
                        t_new = np.linspace(0, 1, n_points)
                        
                        from scipy import interpolate
                        fx = interpolate.interp1d(t, points[:, 0], kind='linear')
                        fy = interpolate.interp1d(t, points[:, 1], kind='linear')
                        
                        resampled = np.column_stack([fx(t_new), fy(t_new)])
                        connection_paths[key]['paths'].append(resampled)
                        connection_paths[key]['speeds'].append(segment.mean_speed)
                        connection_paths[key]['think_times'].append(segment.think_time)
        
        # Draw average paths with confidence bands
        cmap = plt.cm.plasma
        norm = mcolors.Normalize(vmin=0, vmax=len(connection_paths))
        
        for idx, (key, data) in enumerate(connection_paths.items()):
            if not data['paths']:
                continue
            
            paths_array = np.array(data['paths'])
            mean_path = np.mean(paths_array, axis=0)
            std_path = np.std(paths_array, axis=0)
            
            # Get mean speed for color
            mean_speed = np.mean(data['speeds'])
            color = cmap(norm(idx))
            
            # Draw mean path
            ax_main.plot(mean_path[:, 0], mean_path[:, 1], 
                        color=color, linewidth=2, alpha=0.8)
            
            # Draw confidence band (simplified as filled area around path)
            if len(paths_array) > 1:
                # Create confidence band using fill_between for x and y separately
                for i in range(len(mean_path) - 1):
                    x_points = [mean_path[i, 0] - std_path[i, 0],
                               mean_path[i, 0] + std_path[i, 0],
                               mean_path[i+1, 0] + std_path[i+1, 0],
                               mean_path[i+1, 0] - std_path[i+1, 0]]
                    y_points = [mean_path[i, 1] - std_path[i, 1],
                               mean_path[i, 1] + std_path[i, 1],
                               mean_path[i+1, 1] + std_path[i+1, 1],
                               mean_path[i+1, 1] - std_path[i+1, 1]]
                    
                    ax_main.fill(x_points, y_points, color=color, alpha=0.1)
        
        # Set main plot properties
        ax_main.set_aspect('equal')
        ax_main.set_xlim(0, 100)
        ax_main.set_ylim(0, 100)  # No inversion needed - coordinates are already correct
        ax_main.set_xlabel('X Position (%)')
        ax_main.set_ylabel('Y Position (%)')
        
        condition = 'A (Sequential)' if trail_id == 'trail2' else 'B (Alternating)'
        ax_main.set_title(f"Group Analysis - Trail {condition}\n"
                         f"N = {len(trail_results)} participants",
                         fontsize=14, fontweight='bold')
        
        # Create statistics plot
        self._plot_group_statistics(ax_stats, trail_results)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        else:
            plt.savefig(
                self.group_dir / f"group_{trail_id}.png",
                dpi=150, bbox_inches='tight'
            )
        
        plt.close()
    
    def _plot_group_statistics(self, ax, trail_results: List[Dict]):
        """Plot group statistics summary"""
        # Collect metrics
        metrics = {
            'Total Time': [],
            'Error Count': [],
            'Mean Think Time': [],
            'Mean Ink Time': [],
            'Mean Speed': [],
            'Path Optimality': []
        }
        
        for result in trail_results:
            metrics['Total Time'].append(result['total_time'])
            metrics['Error Count'].append(result['total_errors'])
            
            correct_segs = [s for s in result['segments'] if not s.is_error]
            if correct_segs:
                metrics['Mean Think Time'].append(np.mean([s.think_time for s in correct_segs]))
                metrics['Mean Ink Time'].append(np.mean([s.ink_time for s in correct_segs]))
                speeds = [s.mean_speed for s in correct_segs if s.mean_speed > 0]
                metrics['Mean Speed'].append(np.mean(speeds) if speeds else 0)
                opts = [s.path_optimality for s in correct_segs if s.path_optimality > 0]
                metrics['Path Optimality'].append(np.mean(opts) if opts else 0)
        
        # Create box plots
        data_to_plot = []
        labels = []
        
        for metric, values in metrics.items():
            if values and any(v > 0 for v in values):
                data_to_plot.append(values)
                labels.append(metric)
        
        bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True)
        
        # Color boxes
        colors = plt.cm.Set3(np.linspace(0, 1, len(bp['boxes'])))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
        
        ax.set_ylabel('Value')
        ax.set_title('Group Performance Metrics')
        ax.grid(True, alpha=0.3)
        
        # Rotate x-axis labels for better readability
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    def create_velocity_heatmap(self, all_results: List[Dict], trail_id: str,
                               save_path: Optional[str] = None):
        """
        Create a heatmap showing velocity profiles across all participants
        
        Args:
            all_results: List of all participant results
            trail_id: 'trail2' or 'trail4'
            save_path: Optional path to save figure
        """
        # Filter results for specific trail
        trail_results = [r for r in all_results if r and r['trail'] == trail_id]
        
        if not trail_results:
            return
        
        # Collect velocity profiles aligned by segment
        segment_velocities = {}
        
        for result in trail_results:
            for segment in result['segments']:
                if segment.is_error or not segment.velocities:
                    continue
                
                key = f"{segment.start_label}->{segment.end_label}"
                if key not in segment_velocities:
                    segment_velocities[key] = []
                
                # Normalize to 20 points for comparison
                n_points = 20
                velocities = np.array(segment.velocities)
                if len(velocities) >= 2:
                    t = np.linspace(0, 1, len(velocities))
                    t_new = np.linspace(0, 1, n_points)
                    from scipy import interpolate
                    f = interpolate.interp1d(t, velocities, kind='linear')
                    resampled = f(t_new)
                    segment_velocities[key].append(resampled)
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Prepare data for heatmap
        all_profiles = []
        segment_labels = []
        
        for key, profiles in segment_velocities.items():
            if len(profiles) >= 3:  # Only include if we have enough data
                mean_profile = np.mean(profiles, axis=0)
                all_profiles.append(mean_profile)
                segment_labels.append(key)
        
        if all_profiles:
            # Create heatmap
            data = np.array(all_profiles)
            im = ax.imshow(data, aspect='auto', cmap='YlOrRd', interpolation='bilinear')
            
            # Set labels
            ax.set_yticks(range(len(segment_labels)))
            ax.set_yticklabels(segment_labels)
            ax.set_xlabel('Normalized Position Along Path (%)')
            ax.set_ylabel('Path Segment')
            
            condition = 'A (Sequential)' if trail_id == 'trail2' else 'B (Alternating)'
            ax.set_title(f'Velocity Profiles - Trail {condition}', 
                        fontsize=14, fontweight='bold')
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('Velocity (pixels/second)', rotation=270, labelpad=20)
            
            # Add grid
            ax.grid(True, alpha=0.3, color='white', linewidth=0.5)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        else:
            plt.savefig(
                self.group_dir / f"velocity_heatmap_{trail_id}.png",
                dpi=150, bbox_inches='tight'
            )
        
        plt.close()
    
    def create_spatial_density_heatmap(self, all_results: List[Dict]):
        """
        Create side-by-side spatial density heatmaps for Trail A vs Trail B
        
        Args:
            all_results: List of all participant results
        """
        print("Generating spatial density heatmaps...")
        
        # Separate results by trail type
        trail2_results = [r for r in all_results if r and r['trail'] == 'trail2']
        trail4_results = [r for r in all_results if r and r['trail'] == 'trail4']
        
        if not trail2_results and not trail4_results:
            print("No valid trails found for heatmap generation")
            return
            
        # Create figure with side-by-side subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
        
        # Generate heatmaps for each trail
        heatmap_a, think_times_a = self._generate_single_heatmap(trail2_results, 'trail2')
        heatmap_b, think_times_b = self._generate_single_heatmap(trail4_results, 'trail4')
        
        # Determine shared color scales
        if heatmap_a is not None and heatmap_b is not None:
            # Shared scale for path density heatmaps
            vmax_path = max(np.max(heatmap_a), np.max(heatmap_b))
            vmin_path = 0
            
            # Shared scale for think times
            all_think_times = {}
            all_think_times.update(think_times_a)
            all_think_times.update(think_times_b)
            if all_think_times:
                vmax_think = max(all_think_times.values())
                vmin_think = min(all_think_times.values())
            else:
                vmax_think, vmin_think = 1, 0
        else:
            # Fallback if only one trail type exists
            vmax_path = np.max(heatmap_a) if heatmap_a is not None else np.max(heatmap_b)
            vmin_path = 0
            think_times = think_times_a if think_times_a else think_times_b
            if think_times:
                vmax_think = max(think_times.values())
                vmin_think = min(think_times.values())
            else:
                vmax_think, vmin_think = 1, 0
        
        # Plot Trail A heatmap
        if heatmap_a is not None:
            self._plot_single_heatmap(ax1, heatmap_a, think_times_a, 'trail2', 
                                    'Trail A (Sequential Numbers)', 
                                    vmin_path, vmax_path, vmin_think, vmax_think)
        else:
            ax1.text(0.5, 0.5, 'No Trail A Data', ha='center', va='center', 
                    transform=ax1.transAxes, fontsize=16)
            ax1.set_xlim(0, 100)
            ax1.set_ylim(0, 100)
        
        # Plot Trail B heatmap  
        if heatmap_b is not None:
            self._plot_single_heatmap(ax2, heatmap_b, think_times_b, 'trail4', 
                                    'Trail B (Alternating Numbers/Letters)', 
                                    vmin_path, vmax_path, vmin_think, vmax_think)
        else:
            ax2.text(0.5, 0.5, 'No Trail B Data', ha='center', va='center', 
                    transform=ax2.transAxes, fontsize=16)
            ax2.set_xlim(0, 100)
            ax2.set_ylim(0, 100)
        
        # Add shared colorbars
        if heatmap_a is not None or heatmap_b is not None:
            # Colorbar for path density (bottom)
            cbar_ax1 = fig.add_axes([0.15, 0.02, 0.7, 0.03])
            sm_path = plt.cm.ScalarMappable(cmap='RdBu_r', norm=plt.Normalize(vmin_path, vmax_path))
            sm_path.set_array([])
            cbar1 = fig.colorbar(sm_path, cax=cbar_ax1, orientation='horizontal')
            cbar1.set_label('Path Density (Normalized Time per Grid Cell)', fontsize=12)
            
            # Colorbar for think time (right)
            cbar_ax2 = fig.add_axes([0.92, 0.15, 0.03, 0.7])
            sm_think = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin_think, vmax_think))
            sm_think.set_array([])
            cbar2 = fig.colorbar(sm_think, cax=cbar_ax2, orientation='vertical')
            cbar2.set_label('Average Think Time (seconds)', rotation=270, labelpad=20, fontsize=12)
        
        plt.suptitle('Spatial Density Heatmaps: Path Traces and Think Times', 
                    fontsize=16, fontweight='bold', y=0.95)
        
        # Save the plot
        save_path = self.group_dir / 'spatial_density_heatmap.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Spatial density heatmap saved to {save_path}")
    
    def _generate_single_heatmap(self, results: List[Dict], trail_id: str) -> Tuple[np.ndarray, Dict]:
        """
        Generate spatial density heatmap for a single trail type
        
        Args:
            results: List of results for this trail type
            trail_id: 'trail2' or 'trail4'
            
        Returns:
            Tuple of (heatmap_array, think_times_dict)
        """
        if not results:
            return None, {}
        
        # Initialize 50x50 grid for aggregating across participants
        grid_size = 50
        combined_heatmap = np.zeros((grid_size, grid_size))
        think_times_by_circle = {}
        
        circles = self.circles[trail_id]
        
        # Initialize think time tracking
        for label in circles.keys():
            think_times_by_circle[label] = []
        
        participant_count = 0
        
        for result in results:
            if not result or 'segments' not in result:
                continue
                
            # Create individual participant heatmap
            participant_heatmap = np.zeros((grid_size, grid_size))
            
            # Process each segment for this participant
            for segment in result['segments']:
                if segment.is_error:
                    continue
                    
                # Add think time data
                if hasattr(segment, 'think_circle_label') and segment.think_circle_label in think_times_by_circle:
                    think_times_by_circle[segment.think_circle_label].append(segment.think_time)
                elif hasattr(segment, 'start_label') and segment.start_label in think_times_by_circle:
                    think_times_by_circle[segment.start_label].append(segment.think_time)
                
                # Extract coordinate data for path traces
                if hasattr(segment, 'points') and len(segment.points) > 1:
                    points = segment.points
                    if 'x' in points.columns and 'y' in points.columns:
                        x_coords = points['x'].values
                        y_coords = points['y'].values
                        
                        # Convert coordinates to grid indices (0-100% -> 0-49 grid)
                        x_indices = np.clip((x_coords * (grid_size - 1) / 100).astype(int), 0, grid_size - 1)
                        y_indices = np.clip((y_coords * (grid_size - 1) / 100).astype(int), 0, grid_size - 1)
                        
                        # Add time weight to each grid cell the path passes through
                        for x_idx, y_idx in zip(x_indices, y_indices):
                            participant_heatmap[y_idx, x_idx] += segment.ink_time / len(x_indices)
            
            # Normalize participant heatmap by their total path time
            total_time = participant_heatmap.sum()
            if total_time > 0:
                participant_heatmap /= total_time
                combined_heatmap += participant_heatmap
                participant_count += 1
        
        # Normalize by number of participants
        if participant_count > 0:
            combined_heatmap /= participant_count
        
        # Average think times by circle
        avg_think_times = {}
        for label, times in think_times_by_circle.items():
            if times:
                avg_think_times[label] = np.mean(times)
        
        return combined_heatmap, avg_think_times
    
    def _plot_single_heatmap(self, ax, heatmap, think_times, trail_id, title, 
                           vmin_path, vmax_path, vmin_think, vmax_think):
        """
        Plot a single heatmap with circle overlays
        
        Args:
            ax: Matplotlib axis
            heatmap: 50x50 numpy array with path density
            think_times: Dictionary of circle labels -> average think times
            trail_id: 'trail2' or 'trail4'
            title: Plot title
            vmin_path, vmax_path: Color scale limits for path density
            vmin_think, vmax_think: Color scale limits for think times
        """
        circles = self.circles[trail_id]
        
        # Plot the heatmap
        im = ax.imshow(heatmap, extent=[0, 100, 0, 100], origin='lower', 
                      cmap='RdBu_r', vmin=vmin_path, vmax=vmax_path, alpha=0.8)
        
        # Overlay circles colored by think time
        for label, circle in circles.items():
            think_time = think_times.get(label, 0)
            
            # Normalize think time for color mapping
            if vmax_think > vmin_think:
                norm_think = (think_time - vmin_think) / (vmax_think - vmin_think)
            else:
                norm_think = 0
            
            # Get color from viridis colormap
            color = plt.cm.viridis(norm_think)
            
            # Draw circle
            circle_patch = plt.Circle((circle.cx, circle.cy), circle.radius, 
                                    color=color, alpha=0.9, ec='black', linewidth=2)
            ax.add_patch(circle_patch)
            
            # Add label
            ax.text(circle.cx, circle.cy, str(label), ha='center', va='center', 
                   fontweight='bold', fontsize=10, color='white')
        
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        ax.set_xlabel('X Position (%)', fontsize=12)
        ax.set_ylabel('Y Position (%)', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')


def plot_individual_participant_worker(args):
    """
    Worker function for parallel individual visualization
    
    Args:
        args: Tuple of (result, scaled_points_path)
    
    Returns:
        Tuple of (success, participant_id, trail_id, error_msg)
    """
    result, scaled_points_path = args
    
    try:
        # Create analyzer instance for this worker
        analyzer = TrailsAnalyzer(scaled_points_path=scaled_points_path)
        visualizer = TrailsVisualizer(analyzer)
        
        # Generate visualization
        visualizer.plot_individual_participant(result)
        
        return (True, result['participant_id'], result['trail'], None)
    except Exception as e:
        return (False, result.get('participant_id', 'unknown'), 
                result.get('trail', 'unknown'), str(e))


def visualize_individual_parallel(all_results: List[Dict], analyzer: TrailsAnalyzer,
                                 n_processes: Optional[int] = None):
    """
    Generate individual visualizations using parallel processing
    
    Args:
        all_results: List of all participant results
        analyzer: TrailsAnalyzer instance (for configuration)
        n_processes: Number of parallel processes (None for auto-detect)
    
    Returns:
        Number of successful visualizations
    """
    if n_processes is None:
        n_processes = min(cpu_count(), len(all_results), 8)
    
    # Filter out None results
    valid_results = [r for r in all_results if r is not None and 'segments' in r]
    
    if not valid_results:
        print("No valid results to visualize")
        return 0
    
    print(f"Generating {len(valid_results)} individual visualizations using {n_processes} processes...")
    
    # Prepare arguments for workers
    scaled_points_path = analyzer.scaled_points_path
    worker_args = [(result, scaled_points_path) for result in valid_results]
    
    # Track progress
    start_time = time.time()
    successful = 0
    failed = 0
    
    # Process in parallel
    with Pool(processes=n_processes) as pool:
        for i, (success, pid, trail, error) in enumerate(pool.imap_unordered(
                plot_individual_participant_worker, worker_args), 1):
            
            if success:
                successful += 1
                print(f"  [{i}/{len(valid_results)}] ✓ {pid}_{trail}")
            else:
                failed += 1
                print(f"  [{i}/{len(valid_results)}] ✗ {pid}_{trail}: {error}")
            
            # Show progress every 10 visualizations
            if i % 10 == 0 or i == len(valid_results):
                elapsed = time.time() - start_time
                rate = i / elapsed
                remaining = (len(valid_results) - i) / rate if rate > 0 else 0
                print(f"    Progress: {i}/{len(valid_results)} | "
                      f"Rate: {rate:.1f} plots/sec | "
                      f"ETA: {remaining:.0f}s")
    
    elapsed = time.time() - start_time
    print(f"\nIndividual visualizations complete: {successful} successful, "
          f"{failed} failed in {elapsed:.1f}s")
    
    return successful


def visualize_all(analyzer: TrailsAnalyzer, all_results: List[Dict], 
                  participant_results: Dict, parallel: bool = True,
                  n_processes: Optional[int] = None):
    """
    Generate all visualizations
    
    Args:
        analyzer: TrailsAnalyzer instance
        all_results: List of all participant results
        participant_results: Dictionary grouped by participant ID
        parallel: Whether to use parallel processing for individual plots
        n_processes: Number of parallel processes (None for auto-detect)
    """
    visualizer = TrailsVisualizer(analyzer)
    
    print("Generating individual participant visualizations...")
    
    if parallel:
        # Use parallel processing for individual visualizations
        visualize_individual_parallel(all_results, analyzer, n_processes)
    else:
        # Serial processing (original method)
        for pid, trails in participant_results.items():
            for trail_id, result in trails.items():
                visualizer.plot_individual_participant(result)
    
    print("Generating group-level visualizations...")
    # Generate group visualizations (these are harder to parallelize effectively)
    # since they need all data at once and matplotlib isn't thread-safe for complex plots
    for trail_id in ['trail2', 'trail4']:
        visualizer.plot_group_data(all_results, trail_id)
        visualizer.create_velocity_heatmap(all_results, trail_id)
    
    # Generate spatial density heatmap
    visualizer.create_spatial_density_heatmap(all_results)
    
    print(f"Visualizations saved to {visualizer.vis_dir}")


if __name__ == "__main__":
    # This will be called from the main analysis script
    pass
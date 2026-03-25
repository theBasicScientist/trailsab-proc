"""
Trails A/B Task Analysis System
Analyzes digitized trails task data including timing, errors, and movement patterns
"""

import pandas as pd
import numpy as np
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from scipy import interpolate, signal
from scipy.spatial.distance import euclidean
import warnings
warnings.filterwarnings('ignore')


@dataclass
class CircleTarget:
    """Represents a target circle in the trails task"""
    order: int
    cx: float
    cy: float
    label: str
    radius: float
    
    def contains_point(self, x: float, y: float, tolerance: float = 1.5) -> bool:
        """Check if a point is within the circle (with tolerance multiplier)"""
        distance = np.sqrt((x - self.cx)**2 + (y - self.cy)**2)
        return distance <= (self.radius * tolerance)


@dataclass
class LineSegment:
    """Represents a line drawn between two circles"""
    start_label: str
    end_label: str
    points: pd.DataFrame
    is_error: bool
    line_number: int
    
    # Computed metrics
    ink_time: float = 0.0
    think_time: float = 0.0
    think_circle_label: str = ""  # Which circle the think time applies to
    distance: float = 0.0  # Total distance drawn outside circles
    mean_speed: float = 0.0
    speed_variance: float = 0.0
    path_optimality: float = 0.0
    smoothness: float = 0.0  # Based on curvature changes
    hesitation_count: int = 0
    hesitation_duration: float = 0.0
    velocities: List[float] = field(default_factory=list)
    accelerations: List[float] = field(default_factory=list)


class TrailsAnalyzer:
    """Main analysis class for Trails A/B task data"""
    
    def __init__(self, config_path: str = None, scaled_points_path: str = "trails_points_scaled.json",
                 data_dir: str = "trails_responses", preprocessed_dir: str = "preprocessed", 
                 output_dir: str = "output"):
        """
        Initialize the analyzer with configuration and directory paths
        
        Args:
            config_path: Path to trail_making_config.json (deprecated, kept for compatibility)
            scaled_points_path: Path to trails_points_scaled.json with circle locations
            data_dir: Directory containing raw trail response CSV files
            preprocessed_dir: Directory containing preprocessed CSV files
            output_dir: Directory for analysis outputs
        """
        self.data_dir = Path(data_dir)
        self.preprocessed_dir = Path(preprocessed_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Store scaled points path for parallel processing
        self.scaled_points_path = scaled_points_path
        
        # Load circle configurations from scaled points JSON
        self.circles = self._load_scaled_circles(scaled_points_path)
        
        # Create a dummy config for compatibility (for circle ordering)
        self.config = self._create_config_from_circles()
    
    def _load_scaled_circles(self, filepath: str) -> Dict[str, Dict[str, CircleTarget]]:
        """Load circle configurations from trails_points_scaled.json"""
        # Check file extension and load accordingly
        if filepath.endswith('.json'):
            with open(filepath, 'r') as f:
                trails_data = json.load(f)
            
            circles = {}
            for trail_screen, trail_points in trails_data.items():
                trail_circles = {}
                for idx, point in enumerate(trail_points):
                    order = idx + 1
                    circle = CircleTarget(
                        order=order,
                        cx=point['x'],
                        cy=point['y'],
                        label=str(point['label']),
                        radius=point['radius']
                    )
                    trail_circles[circle.label] = circle
                
                # Store circles for this trail
                circles[trail_screen] = trail_circles
        else:
            # Fallback to CSV loading for backward compatibility
            df = pd.read_csv(filepath)
            circles = {}
            
            # Group by trail_screen
            for trail_screen in df['trail_screen'].unique():
                trail_data = df[df['trail_screen'] == trail_screen]
                trail_circles = {}
                
                for idx, row in trail_data.iterrows():
                    # Create circle with order based on row index within trail
                    order = len(trail_circles) + 1
                    circle = CircleTarget(
                        order=order,
                        cx=row['x'],
                        cy=row['y'],
                        label=str(row['label']),
                        radius=row['radius']
                    )
                    trail_circles[circle.label] = circle
                
                # Map trail1->trail2 and trail3->trail4 for analysis
                if trail_screen == 'trail1':
                    circles['trail2'] = trail_circles  # Use trail1 config for trail2 analysis
                elif trail_screen == 'trail2':
                    circles['trail2'] = trail_circles  # This is the actual trail2 (TrailsA)
                elif trail_screen == 'trail3':
                    circles['trail4'] = trail_circles  # Use trail3 config for trail4 analysis
                elif trail_screen == 'trail4':
                    circles['trail4'] = trail_circles  # This is the actual trail4 (TrailsB)
        
        return circles
    
    def _create_config_from_circles(self) -> Dict:
        """Create a config dict from circles for compatibility"""
        config = {}
        for trail_id, trail_circles in self.circles.items():
            config[trail_id] = {
                'items': [
                    {
                        'order': circle.order,
                        'cx': circle.cx,
                        'cy': circle.cy,
                        'label': circle.label
                    }
                    for circle in trail_circles.values()
                ]
            }
        return config
    
    def load_participant_data(self, filepath: str) -> pd.DataFrame:
        """Load and validate participant trail data"""
        df = pd.read_csv(filepath)
        
        # Handle different column formats
        if 'error' in df.columns and 'is_error' not in df.columns:
            # Convert error column to is_error boolean
            # E0 means no error, E1 means error
            df['is_error'] = df['error'].apply(lambda x: False if x == 'E0' else True)
        
        # Ensure required columns exist
        required_cols = ['line_number', 'x', 'y', 'correct_path', 'actual_path',
                        'seconds', 'is_error', 'total_time', 'total_number_of_errors']
        missing = set(required_cols) - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        
        # Data quality check: Ensure paradigm correctness (24 unique paths expected)
        unique_paths = df['correct_path'].nunique()
        if unique_paths != 24:
            raise ValueError(f"Invalid paradigm: Expected 24 unique paths, found {unique_paths}. "
                           f"Skipping file with shorter experimental approach.")
        
        # No coordinate transformation needed - data and circles are both in 0-100 scale
        return df
    
    def resample_to_60hz(self, points: pd.DataFrame, start_time: float, 
                         end_time: float) -> pd.DataFrame:
        """
        Resample trajectory points to uniform 60Hz
        
        Args:
            points: DataFrame with x, y, seconds columns
            start_time: Start time in seconds
            end_time: End time in seconds
        
        Returns:
            Resampled DataFrame at 60Hz
        """
        if len(points) < 2:
            return points
        
        # Create uniform 60Hz timeline
        dt = 1.0 / 60.0  # 60Hz sampling period
        uniform_time = np.arange(start_time, end_time, dt)
        
        # Skip if already well-sampled
        if len(uniform_time) < 2:
            return points
        
        try:
            # Interpolate x and y coordinates
            fx = interpolate.interp1d(points['seconds'].values, points['x'].values, 
                                     kind='linear', fill_value='extrapolate')
            fy = interpolate.interp1d(points['seconds'].values, points['y'].values,
                                     kind='linear', fill_value='extrapolate')
            
            resampled = pd.DataFrame({
                'seconds': uniform_time,
                'x': fx(uniform_time),
                'y': fy(uniform_time)
            })
            
            # Preserve other columns from first row
            for col in points.columns:
                if col not in ['x', 'y', 'seconds']:
                    resampled[col] = points[col].iloc[0]
            
            return resampled
        except:
            # If interpolation fails, return original
            return points
    
    def segment_lines(self, df: pd.DataFrame, trail_id: str) -> List[LineSegment]:
        """
        Segment data into individual lines drawn between circles
        
        Args:
            df: Participant data DataFrame
            trail_id: 'trail2' or 'trail4'
        
        Returns:
            List of LineSegment objects
        """
        segments = []
        circles = self.circles[trail_id]
        
        # First try to segment by actual_path changes (more accurate)
        unique_paths = df['actual_path'].unique()
        
        if len(unique_paths) > 1:
            # Segment by actual_path changes
            segment_counter = 0
            for path in unique_paths:
                if pd.isna(path) or '~' not in path:
                    continue
                    
                path_data = df[df['actual_path'] == path].copy()
                
                if len(path_data) == 0:
                    continue
                
                # Get line endpoints
                start_label, end_label = path.split(' ~ ')
                
                # Create line segment
                segment = LineSegment(
                    start_label=start_label.strip(),
                    end_label=end_label.strip(),
                    points=path_data,
                    is_error=path_data['is_error'].iloc[0],
                    line_number=segment_counter
                )
                
                segments.append(segment)
                segment_counter += 1
        else:
            # Fall back to grouping by line_number
            for line_num in df['line_number'].unique():
                line_data = df[df['line_number'] == line_num].copy()
                
                if len(line_data) == 0:
                    continue
                
                # Get line endpoints from actual_path
                path = line_data['actual_path'].iloc[0]
                if pd.isna(path) or '~' not in path:
                    continue
                    
                start_label, end_label = path.split(' ~ ')
                
                # Create line segment
                segment = LineSegment(
                    start_label=start_label.strip(),
                    end_label=end_label.strip(),
                    points=line_data,
                    is_error=line_data['is_error'].iloc[0],
                    line_number=int(line_num)
                )
                
                segments.append(segment)
        
        return segments
    
    def compute_segment_metrics(self, segment: LineSegment, trail_id: str) -> LineSegment:
        """
        Compute metrics for a line segment (excluding think time which is calculated separately)
        
        Args:
            segment: LineSegment object
            trail_id: Trail identifier for circle lookup
        
        Returns:
            LineSegment with computed metrics
        """
        circles = self.circles[trail_id]
        points = segment.points.copy()
        
        if len(points) < 2:
            return segment
        
        # Get circle objects
        if segment.start_label not in circles or segment.end_label not in circles:
            return segment
        
        start_circle = circles[segment.start_label]
        end_circle = circles[segment.end_label]
        
        # Reset index to ensure we can iterate properly
        points = points.reset_index(drop=True)
        
        # NOTE: Think time is now calculated separately using consecutive segments
        # This method only handles ink time and movement metrics
        
        # Find ink time (time from leaving start circle to entering end circle)
        ink_start_idx = None
        ink_end_idx = None
        
        # Start looking for ink trajectory after leaving start circle
        for idx in range(len(points)):
            row = points.iloc[idx]
            
            # Mark when we leave the start circle
            if not start_circle.contains_point(row['x'], row['y']) and ink_start_idx is None:
                ink_start_idx = idx
            
            # Mark when we enter the end circle
            if ink_start_idx is not None and end_circle.contains_point(row['x'], row['y']):
                ink_end_idx = idx
                break
        
        # If we have a valid ink trajectory
        if ink_start_idx is not None and ink_end_idx is not None and ink_end_idx > ink_start_idx:
            ink_points = points.iloc[ink_start_idx:ink_end_idx + 1].copy()
            
            if len(ink_points) >= 2:
                # Calculate ink time
                ink_start = ink_points.iloc[0]['seconds']
                ink_end = ink_points.iloc[-1]['seconds']
                segment.ink_time = ink_end - ink_start
                
                # Resample to 60Hz if duration is sufficient
                if segment.ink_time > 0.05:  # Only resample if duration > 50ms
                    ink_points = self.resample_to_60hz(ink_points, ink_start, ink_end)
                
                # Calculate velocities and speeds
                if len(ink_points) >= 2:
                    dx = np.diff(ink_points['x'].values)
                    dy = np.diff(ink_points['y'].values)
                    dt = np.diff(ink_points['seconds'].values)
                    
                    # Avoid division by zero
                    dt[dt == 0] = 1e-6
                    
                    distances = np.sqrt(dx**2 + dy**2)
                    velocities = distances / dt
                    segment.velocities = velocities.tolist()
                    
                    # Calculate mean speed and variance
                    segment.mean_speed = np.mean(velocities)
                    segment.speed_variance = np.var(velocities)
                    
                    # Calculate accelerations
                    if len(velocities) >= 2:
                        segment.accelerations = np.diff(velocities).tolist()
                    
                    # Path optimality
                    total_distance = np.sum(distances)
                    segment.distance = total_distance  # Store the total distance
                    optimal_distance = euclidean(
                        [start_circle.cx, start_circle.cy],
                        [end_circle.cx, end_circle.cy]
                    ) - start_circle.radius - end_circle.radius
                    
                    if optimal_distance > 0:
                        segment.path_optimality = optimal_distance / total_distance
                    
                    # Smoothness (based on curvature changes)
                    if len(ink_points) >= 3:
                        segment.smoothness = self._calculate_smoothness(ink_points)
                    
                    # Hesitation detection
                    segment.hesitation_count, segment.hesitation_duration = \
                        self._detect_hesitations(velocities, dt[0] if len(dt) > 0 else 1/60)
        elif ink_start_idx is not None:
            # Line started but never reached destination - use all remaining points
            ink_points = points.iloc[ink_start_idx:].copy()
            if len(ink_points) >= 2:
                segment.ink_time = ink_points.iloc[-1]['seconds'] - ink_points.iloc[0]['seconds']
        
        return segment
    
    def calculate_think_times(self, segments: List[LineSegment], trail_id: str) -> List[LineSegment]:
        """
        Calculate think times using consecutive segments approach
        
        Think time at a circle = time from entering the circle (end of incoming segment) 
                                to leaving the circle (start of outgoing segment)
        
        Args:
            segments: List of LineSegment objects in order
            trail_id: Trail identifier for circle lookup
        
        Returns:
            Updated list of segments with think times calculated
        """
        circles = self.circles[trail_id]
        
        # Sort segments to ensure proper order (by their actual sequence in the task)
        # For trails, we can use the circle order from the configuration
        trail_config = self.config[trail_id]
        circle_order = {item['label']: item['order'] for item in trail_config['items']}
        
        # Sort segments by the order of their start circles
        segments = sorted(segments, key=lambda s: circle_order.get(s.start_label, 999))
        
        # Calculate think times for consecutive segments
        for i in range(len(segments) - 1):
            current_seg = segments[i]
            next_seg = segments[i + 1]
            
            # The destination of current segment should equal origin of next segment
            if current_seg.end_label == next_seg.start_label:
                circle_label = current_seg.end_label
                
                if circle_label not in circles:
                    continue
                
                circle = circles[circle_label]
                
                # Find when we ENTERED this circle (working backwards from end of current segment)
                # We want the first point of the contiguous run inside the circle at the
                # tail of the segment — i.e. the actual moment of entry, not the last point.
                entry_time = None
                current_points = current_seg.points.reset_index(drop=True)

                for idx in range(len(current_points) - 1, -1, -1):  # Work backwards
                    point = current_points.iloc[idx]
                    if circle.contains_point(point['x'], point['y']):
                        entry_time = point['seconds']
                    else:
                        # We've found the last point outside the circle before the
                        # contiguous inside-run, so entry_time is already set to the
                        # earliest point inside the circle at the tail end.
                        break
                
                # Find when we LEFT this circle (working forwards from start of next segment)
                exit_time = None
                next_points = next_seg.points.reset_index(drop=True)
                
                for idx in range(len(next_points)):
                    point = next_points.iloc[idx]
                    if not circle.contains_point(point['x'], point['y']):
                        exit_time = point['seconds']
                        break
                
                # If we couldn't find exit time, use the first point of next segment
                if exit_time is None and len(next_points) > 0:
                    exit_time = next_points.iloc[0]['seconds']
                
                # Calculate think time and assign to the next segment (the one leaving this circle)
                if entry_time is not None and exit_time is not None and exit_time > entry_time:
                    think_time = exit_time - entry_time
                    next_seg.think_time = think_time
                    
                    # Also store which circle this think time applies to
                    next_seg.think_circle_label = circle_label
        
        # Handle the first segment - check if user started inside the first circle
        if segments and len(segments[0].points) > 0:
            first_seg = segments[0]
            if first_seg.start_label in circles:
                start_circle = circles[first_seg.start_label]
                first_points = first_seg.points.reset_index(drop=True)
                
                # Find how long user stayed in starting circle
                exit_time = None
                for idx in range(len(first_points)):
                    point = first_points.iloc[idx]
                    if not start_circle.contains_point(point['x'], point['y']):
                        exit_time = point['seconds']
                        break
                
                if exit_time is not None:
                    # Think time for first segment is from start to leaving first circle
                    start_time = first_points.iloc[0]['seconds']
                    first_seg.think_time = exit_time - start_time
                    first_seg.think_circle_label = first_seg.start_label
        
        return segments
    
    def _calculate_smoothness(self, points: pd.DataFrame) -> float:
        """
        Calculate path smoothness based on curvature changes
        Lower values indicate smoother paths
        """
        if len(points) < 3:
            return 0.0
        
        x = points['x'].values
        y = points['y'].values
        
        # Calculate angles between consecutive segments
        angles = []
        for i in range(1, len(x) - 1):
            v1 = np.array([x[i] - x[i-1], y[i] - y[i-1]])
            v2 = np.array([x[i+1] - x[i], y[i+1] - y[i]])
            
            # Skip if either vector is zero
            if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0:
                continue
            
            # Normalize vectors
            v1 = v1 / np.linalg.norm(v1)
            v2 = v2 / np.linalg.norm(v2)
            
            # Calculate angle change
            cos_angle = np.clip(np.dot(v1, v2), -1.0, 1.0)
            angle = np.arccos(cos_angle)
            angles.append(angle)
        
        if not angles:
            return 0.0
        
        # Return standard deviation of angle changes (in degrees)
        return np.std(np.degrees(angles))
    
    def _detect_hesitations(self, velocities: np.ndarray, dt: float, 
                           threshold_percentile: int = 20) -> Tuple[int, float]:
        """
        Detect hesitations as periods of significantly reduced velocity
        
        Returns:
            (hesitation_count, total_hesitation_duration)
        """
        if len(velocities) < 3:
            return 0, 0.0
        
        # Define hesitation as velocity below threshold
        threshold = np.percentile(velocities, threshold_percentile)
        hesitations = velocities < threshold
        
        # Count distinct hesitation periods
        hesitation_changes = np.diff(hesitations.astype(int))
        hesitation_starts = np.where(hesitation_changes == 1)[0] + 1
        hesitation_count = len(hesitation_starts)
        
        # Add one if starting with hesitation
        if hesitations[0]:
            hesitation_count += 1
        
        # Calculate total hesitation duration
        hesitation_duration = np.sum(hesitations) * dt
        
        return hesitation_count, hesitation_duration
    
    def process_participant(self, participant_file: str) -> Dict:
        """
        Process all trails for a single participant
        
        Args:
            participant_file: Path to participant CSV file
        
        Returns:
            Dictionary with participant metrics
        """
        filepath = Path(participant_file)
        filename = filepath.name
        
        # Extract participant ID and trail number
        parts = filename.replace('.csv', '').split('-')
        if len(parts) >= 2:
            participant_id = parts[0]
            trail_num = parts[-1]  # e.g., 'trail2' or 'trail4'
        else:
            return None
        
        # Only process trails 2 and 4
        if trail_num not in ['trail2', 'trail4']:
            return None
        
        # Load data with quality checks
        try:
            df = self.load_participant_data(filepath)
        except ValueError as e:
            # Skip files that don't meet quality criteria
            print(f"  Skipping: {e}")
            return None
        
        # Get basic metrics from first row
        total_time = df['total_time'].iloc[0] if pd.notna(df['total_time'].iloc[0]) else 0
        total_errors = df['total_number_of_errors'].iloc[0] if pd.notna(df['total_number_of_errors'].iloc[0]) else 0
        
        # Segment into lines
        segments = self.segment_lines(df, trail_num)
        
        # Compute metrics for each segment (excluding think time)
        for segment in segments:
            self.compute_segment_metrics(segment, trail_num)
        
        # Calculate think times using consecutive segments approach
        segments = self.calculate_think_times(segments, trail_num)
        
        # Separate correct and error segments
        correct_segments = [s for s in segments if not s.is_error]
        error_segments = [s for s in segments if s.is_error]
        
        # Calculate aggregate metrics
        result = {
            'participant_id': participant_id,
            'trail': trail_num,
            'total_time': total_time,
            'total_errors': int(total_errors),
            'segments': correct_segments + error_segments  # Errors at end
        }
        
        # Calculate time spent on errors vs correct
        if correct_segments:
            result['correct_time'] = sum(s.ink_time + s.think_time for s in correct_segments)
            result['total_ink_time'] = sum(s.ink_time for s in correct_segments)
            result['total_distance'] = sum(s.distance for s in correct_segments)
        else:
            result['correct_time'] = 0
            result['total_ink_time'] = 0
            result['total_distance'] = 0
        
        if error_segments:
            result['error_time'] = sum(s.ink_time + s.think_time for s in error_segments)
        else:
            result['error_time'] = 0
        
        return result
    
    def generate_summary_statistics(self, all_results: List[Dict]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Generate summary statistics across all participants
        
        Returns:
            Tuple of (TrailsA DataFrame, TrailsB DataFrame)
        """
        trails_a_data = []
        trails_b_data = []
        
        for result in all_results:
            if result is None:
                continue
            
            row = {
                'participant_id': result['participant_id'],
                'total_time': result['total_time'],
                'correct_time': result['correct_time'],
                'error_time': result['error_time'],
                'error_count': result['total_errors'],
                'total_ink_time': result.get('total_ink_time', 0),
                'total_distance': result.get('total_distance', 0)
            }
            
            # Add advanced metrics for correct trials
            correct_segments = [s for s in result['segments'] if not s.is_error]
            error_segments = [s for s in result['segments'] if s.is_error]
            
            if correct_segments:
                row['mean_think_time'] = np.mean([s.think_time for s in correct_segments])
                row['mean_ink_time'] = np.mean([s.ink_time for s in correct_segments])
                row['mean_speed'] = np.mean([s.mean_speed for s in correct_segments if s.mean_speed > 0])
                row['mean_speed_variance'] = np.mean([s.speed_variance for s in correct_segments if s.speed_variance > 0])
                row['mean_path_optimality'] = np.mean([s.path_optimality for s in correct_segments if s.path_optimality > 0])
                row['mean_smoothness'] = np.mean([s.smoothness for s in correct_segments])
                row['mean_hesitations'] = np.mean([s.hesitation_count for s in correct_segments])
                row['total_hesitations'] = sum([s.hesitation_count for s in correct_segments])
                row['total_hesitation_duration'] = sum([s.hesitation_duration for s in correct_segments])
            
            # Add error-specific metrics if errors exist
            if error_segments:
                row['error_mean_think_time'] = np.mean([s.think_time for s in error_segments])
                row['error_mean_ink_time'] = np.mean([s.ink_time for s in error_segments])
                row['error_mean_speed'] = np.mean([s.mean_speed for s in error_segments if s.mean_speed > 0])
                row['error_mean_speed_variance'] = np.mean([s.speed_variance for s in error_segments if s.speed_variance > 0])
            
            # Add to appropriate list based on trail
            if result['trail'] == 'trail2':
                trails_a_data.append(row)
            else:  # trail4
                trails_b_data.append(row)
        
        return pd.DataFrame(trails_a_data), pd.DataFrame(trails_b_data)
    
    def generate_item_by_item_csv(self, result: Dict, output_path: str):
        """Generate item-by-item CSV for a single participant"""
        if result is None or not result['segments']:
            return
        
        rows = []
        for segment in result['segments']:
            row = {
                'participant_id': result['participant_id'],
                'trail': result['trail'],
                'line_number': segment.line_number,
                'path': f"{segment.start_label} ~ {segment.end_label}",
                'is_error': segment.is_error,
                'think_time': segment.think_time,
                'think_circle': segment.think_circle_label,
                'ink_time': segment.ink_time,
                'distance': segment.distance,
                'mean_speed': segment.mean_speed,
                'speed_variance': segment.speed_variance,
                'path_optimality': segment.path_optimality,
                'smoothness': segment.smoothness,
                'hesitation_count': segment.hesitation_count,
                'hesitation_duration': segment.hesitation_duration,
                'mean_velocity': np.mean(segment.velocities) if segment.velocities else 0,
                'max_velocity': np.max(segment.velocities) if segment.velocities else 0,
                'min_velocity': np.min(segment.velocities) if segment.velocities else 0
            }
            rows.append(row)
        
        df = pd.DataFrame(rows)
        df.to_csv(output_path, index=False)


def main():
    """Main execution function"""
    # Initialize analyzer
    analyzer = TrailsAnalyzer(
        config_path='trail_making_config.json',
        data_dir='trails_responses',
        preprocessed_dir='preprocessed',
        output_dir='output'
    )
    
    # Find all participant files
    all_files = []
    
    # Check preprocessed directory first
    if analyzer.preprocessed_dir.exists():
        all_files.extend(analyzer.preprocessed_dir.glob('*trail2.csv'))
        all_files.extend(analyzer.preprocessed_dir.glob('*trail4.csv'))
    
    # Also check trails_responses directory
    if analyzer.data_dir.exists():
        all_files.extend(analyzer.data_dir.glob('*trail2.csv'))
        all_files.extend(analyzer.data_dir.glob('*trail4.csv'))
    
    print(f"Found {len(all_files)} trail files to process")
    
    # Process each participant
    all_results = []
    participant_results = {}  # Group by participant
    
    for filepath in all_files:
        print(f"Processing {filepath.name}...")
        result = analyzer.process_participant(str(filepath))
        
        if result:
            all_results.append(result)
            
            # Group results by participant
            pid = result['participant_id']
            if pid not in participant_results:
                participant_results[pid] = {}
            participant_results[pid][result['trail']] = result
    
    # Generate summary statistics
    summary_df = analyzer.generate_summary_statistics(all_results)
    summary_output = analyzer.output_dir / 'summary_statistics.csv'
    summary_df.to_csv(summary_output, index=False)
    print(f"Summary statistics saved to {summary_output}")
    
    # Generate item-by-item CSVs for each participant
    item_dir = analyzer.output_dir / 'item_by_item'
    item_dir.mkdir(exist_ok=True)
    
    for pid, trails in participant_results.items():
        for trail_id, result in trails.items():
            output_path = item_dir / f"{pid}_{trail_id}_items.csv"
            analyzer.generate_item_by_item_csv(result, str(output_path))
    
    print(f"Item-by-item CSVs saved to {item_dir}")
    
    return analyzer, all_results, participant_results


if __name__ == "__main__":
    analyzer, all_results, participant_results = main()
#!/usr/bin/env python3
"""
Improved trajectory visualization for single-lane traffic with traffic light
Creates both spatial and time-space diagrams
"""

import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import defaultdict
import numpy as np

def parse_fcd_data(fcd_file, start_time=300.0, end_time=600.0):
    """Parse FCD XML file and extract vehicle trajectories within time range"""
    trajectories = defaultdict(list)

    context = ET.iterparse(fcd_file, events=('start', 'end'))

    for event, elem in context:
        if event == 'end' and elem.tag == 'timestep':
            time = float(elem.get('time'))

            if start_time <= time <= end_time:
                for vehicle in elem.findall('vehicle'):
                    veh_id = vehicle.get('id')
                    x = float(vehicle.get('x'))
                    y = float(vehicle.get('y'))
                    speed = float(vehicle.get('speed'))
                    veh_type = vehicle.get('type')

                    trajectories[veh_id].append((time, x, y, speed, veh_type))

            elem.clear()

            if time > end_time:
                break

    return trajectories

def plot_comprehensive_trajectories(trajectories, start_time=300.0, end_time=600.0):
    """Create comprehensive trajectory visualization with multiple subplots"""

    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))

    # Define layout: 2 rows, 2 columns
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    ax1 = fig.add_subplot(gs[0, :])  # Time-space diagram (full width)
    ax2 = fig.add_subplot(gs[1, 0])  # Spatial view
    ax3 = fig.add_subplot(gs[1, 1])  # Speed profiles

    # Color mapping
    type_colors = {'CV': 'red', 'NC': 'blue'}

    cv_count = 0
    nc_count = 0

    # Traffic light location
    tl_location = 1000.0  # meters

    # Plot 1: Time-Space Diagram
    for veh_id, trajectory in trajectories.items():
        if len(trajectory) < 2:
            continue

        times = [t[0] for t in trajectory]
        y_coords = [t[2] for t in trajectory]
        veh_type = trajectory[0][4]

        if veh_type == 'CV':
            cv_count += 1
        else:
            nc_count += 1

        color = type_colors.get(veh_type, 'gray')
        ax1.plot(times, y_coords, color=color, alpha=0.5, linewidth=2)

        # Mark start and end points
        ax1.plot(times[0], y_coords[0], 'o', color=color, markersize=5, alpha=0.7)
        ax1.plot(times[-1], y_coords[-1], 's', color=color, markersize=5, alpha=0.7)

    # Add traffic light location line
    ax1.axhline(y=tl_location, color='orange', linestyle='--', linewidth=2,
                label='Traffic Light', alpha=0.7)

    ax1.set_xlabel('Time (s)', fontsize=12)
    ax1.set_ylabel('Position (m)', fontsize=12)
    ax1.set_title('Time-Space Diagram: Vehicle Trajectories', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(start_time, end_time)

    # Plot 2: Spatial View (X-Y trajectories)
    for veh_id, trajectory in trajectories.items():
        if len(trajectory) < 2:
            continue

        x_coords = [t[1] for t in trajectory]
        y_coords = [t[2] for t in trajectory]
        veh_type = trajectory[0][4]

        color = type_colors.get(veh_type, 'gray')
        ax2.plot(x_coords, y_coords, color=color, alpha=0.4, linewidth=1.5)
        ax2.plot(x_coords[0], y_coords[0], 'o', color=color, markersize=4, alpha=0.5)
        ax2.plot(x_coords[-1], y_coords[-1], 's', color=color, markersize=4, alpha=0.5)

    # Add traffic light line
    ax2.axhline(y=tl_location, color='orange', linestyle='--', linewidth=2,
                alpha=0.7, label='Traffic Light')

    ax2.set_xlabel('X Position (m)', fontsize=12)
    ax2.set_ylabel('Y Position (m)', fontsize=12)
    ax2.set_title('Spatial View: Lane Configuration', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper right')

    # Plot 3: Speed Profiles
    for veh_id, trajectory in trajectories.items():
        if len(trajectory) < 2:
            continue

        times = [t[0] for t in trajectory]
        speeds = [t[3] for t in trajectory]  # speed in m/s
        veh_type = trajectory[0][4]

        color = type_colors.get(veh_type, 'gray')
        ax3.plot(times, speeds, color=color, alpha=0.3, linewidth=1)

    ax3.set_xlabel('Time (s)', fontsize=12)
    ax3.set_ylabel('Speed (m/s)', fontsize=12)
    ax3.set_title('Speed Profiles Over Time', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(start_time, end_time)

    # Add overall title and legend
    fig.suptitle(f'Vehicle Trajectories Analysis: {start_time}s - {end_time}s\n' +
                 f'Total Vehicles: {len(trajectories)} (CV: {cv_count}, NC: {nc_count})',
                 fontsize=16, fontweight='bold', y=0.98)

    # Create common legend
    cv_patch = mpatches.Patch(color='red', label=f'CV (Connected): {cv_count}')
    nc_patch = mpatches.Patch(color='blue', label=f'NC (Non-Connected): {nc_count}')
    tl_line = mpatches.Patch(color='orange', label='Traffic Light @ 1000m')

    fig.legend(handles=[cv_patch, nc_patch, tl_line],
              loc='upper right', bbox_to_anchor=(0.98, 0.95), fontsize=11)

    return fig

def create_detailed_time_space_diagram(trajectories, start_time=300.0, end_time=600.0):
    """Create a detailed time-space diagram focusing on traffic light interaction"""

    fig, ax = plt.subplots(figsize=(16, 10))

    type_colors = {'CV': 'red', 'NC': 'blue'}
    tl_location = 1000.0

    cv_trajectories = []
    nc_trajectories = []

    # Separate trajectories by type
    for veh_id, trajectory in trajectories.items():
        if len(trajectory) < 2:
            continue

        times = [t[0] for t in trajectory]
        y_coords = [t[2] for t in trajectory]
        veh_type = trajectory[0][4]

        if veh_type == 'CV':
            cv_trajectories.append((times, y_coords, veh_id))
        else:
            nc_trajectories.append((times, y_coords, veh_id))

    # Plot CV trajectories
    for times, y_coords, veh_id in cv_trajectories:
        ax.plot(times, y_coords, color='red', alpha=0.6, linewidth=2.5, label='CV' if veh_id == cv_trajectories[0][2] else "")

    # Plot NC trajectories
    for times, y_coords, veh_id in nc_trajectories:
        ax.plot(times, y_coords, color='blue', alpha=0.6, linewidth=2.5, label='NC' if veh_id == nc_trajectories[0][2] else "")

    # Add traffic light location
    ax.axhline(y=tl_location, color='orange', linestyle='--', linewidth=3,
               label='Traffic Light (1000m)', alpha=0.8)

    # Add shaded regions to show approximate traffic light cycles (60s cycle)
    # Red: 30s, Green: 22s, Yellow: 3s, Red: 5s
    cycle_duration = 60  # seconds
    first_cycle_start = (start_time // cycle_duration) * cycle_duration

    for cycle_start in np.arange(first_cycle_start, end_time + cycle_duration, cycle_duration):
        # Red phase: 0-30s
        ax.axvspan(cycle_start, cycle_start + 30, alpha=0.1, color='red', zorder=0)
        # Green phase: 30-52s
        ax.axvspan(cycle_start + 30, cycle_start + 52, alpha=0.1, color='green', zorder=0)
        # Yellow phase: 52-55s
        ax.axvspan(cycle_start + 52, cycle_start + 55, alpha=0.1, color='yellow', zorder=0)
        # Red phase: 55-60s
        ax.axvspan(cycle_start + 55, cycle_start + 60, alpha=0.1, color='red', zorder=0)

    ax.set_xlabel('Time (seconds)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Position (meters)', fontsize=14, fontweight='bold')
    ax.set_title(f'Detailed Time-Space Diagram: {start_time}s - {end_time}s\n' +
                 f'Showing Traffic Light Cycles and Vehicle Progression\n' +
                 f'CV: {len(cv_trajectories)} vehicles, NC: {len(nc_trajectories)} vehicles',
                 fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.4, linestyle=':', linewidth=0.5)
    ax.set_xlim(start_time, end_time)
    ax.legend(loc='upper left', fontsize=12)

    # Add text annotation for signal timing
    textstr = 'Signal Cycle (60s):\nRed: 30s\nGreen: 22s\nYellow: 3s\nRed: 5s'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=props)

    plt.tight_layout()
    return fig

def main():
    print("Parsing FCD output file...")
    trajectories = parse_fcd_data('fcd_output.xml', start_time=300.0, end_time=600.0)

    print(f"\nFound {len(trajectories)} vehicles between 300s and 600s")
    cv_count = sum(1 for traj in trajectories.values() if traj[0][4] == 'CV')
    nc_count = len(trajectories) - cv_count
    print(f"  - Connected Vehicles (CV): {cv_count}")
    print(f"  - Non-Connected Vehicles (NC): {nc_count}")

    # Create comprehensive visualization
    print("\nCreating comprehensive trajectory visualization...")
    fig1 = plot_comprehensive_trajectories(trajectories, start_time=300.0, end_time=600.0)
    fig1.savefig('trajectory_comprehensive_300_600s.png', dpi=300, bbox_inches='tight')
    fig1.savefig('trajectory_comprehensive_300_600s.pdf', bbox_inches='tight')
    print("  - Saved: trajectory_comprehensive_300_600s.png")
    print("  - Saved: trajectory_comprehensive_300_600s.pdf")

    # Create detailed time-space diagram
    print("\nCreating detailed time-space diagram...")
    fig2 = create_detailed_time_space_diagram(trajectories, start_time=300.0, end_time=600.0)
    fig2.savefig('trajectory_timespace_300_600s.png', dpi=300, bbox_inches='tight')
    fig2.savefig('trajectory_timespace_300_600s.pdf', bbox_inches='tight')
    print("  - Saved: trajectory_timespace_300_600s.png")
    print("  - Saved: trajectory_timespace_300_600s.pdf")

    plt.close('all')
    print("\nVisualization complete!")

if __name__ == '__main__':
    main()

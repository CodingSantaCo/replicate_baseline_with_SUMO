#!/usr/bin/env python3
"""
Script to parse FCD output and create trajectory visualization between 300s and 600s
"""

import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import defaultdict
import numpy as np

def parse_fcd_data(fcd_file, start_time=300.0, end_time=600.0):
    """
    Parse FCD XML file and extract vehicle trajectories within time range

    Args:
        fcd_file: Path to FCD output XML file
        start_time: Start time in seconds (default 300s)
        end_time: End time in seconds (default 600s)

    Returns:
        Dictionary mapping vehicle_id to list of (time, x, y, type) tuples
    """
    trajectories = defaultdict(list)

    # Parse XML iteratively to handle large files
    context = ET.iterparse(fcd_file, events=('start', 'end'))

    for event, elem in context:
        if event == 'end' and elem.tag == 'timestep':
            time = float(elem.get('time'))

            # Only process timesteps within our range
            if start_time <= time <= end_time:
                for vehicle in elem.findall('vehicle'):
                    veh_id = vehicle.get('id')
                    x = float(vehicle.get('x'))
                    y = float(vehicle.get('y'))
                    veh_type = vehicle.get('type')

                    trajectories[veh_id].append((time, x, y, veh_type))

            # Clear element to save memory
            elem.clear()

            # Stop if we've passed the end time
            if time > end_time:
                break

    return trajectories

def plot_trajectories(trajectories, start_time=300.0, end_time=600.0):
    """
    Create trajectory visualization

    Args:
        trajectories: Dictionary mapping vehicle_id to trajectory data
        start_time: Start time for plot title
        end_time: End time for plot title
    """
    fig, ax = plt.subplots(figsize=(14, 10))

    # Color mapping for vehicle types
    type_colors = {
        'CV': 'red',      # Connected Vehicles
        'NC': 'blue'      # Non-Connected Vehicles
    }

    # Plot each vehicle trajectory
    cv_count = 0
    nc_count = 0

    for veh_id, trajectory in trajectories.items():
        if len(trajectory) < 2:
            continue

        # Extract data
        times = [t[0] for t in trajectory]
        x_coords = [t[1] for t in trajectory]
        y_coords = [t[2] for t in trajectory]
        veh_type = trajectory[0][3]

        if veh_type == 'CV':
            cv_count += 1
        else:
            nc_count += 1

        # Plot trajectory
        color = type_colors.get(veh_type, 'gray')
        ax.plot(x_coords, y_coords, color=color, alpha=0.3, linewidth=1.5)

        # Mark start and end points
        ax.plot(x_coords[0], y_coords[0], 'o', color=color, markersize=4, alpha=0.5)
        ax.plot(x_coords[-1], y_coords[-1], 's', color=color, markersize=4, alpha=0.5)

    # Create legend
    cv_patch = mpatches.Patch(color='red', label=f'CV (Connected Vehicles): {cv_count}')
    nc_patch = mpatches.Patch(color='blue', label=f'NC (Non-Connected Vehicles): {nc_count}')
    ax.legend(handles=[cv_patch, nc_patch], loc='upper right', fontsize=10)

    # Labels and title
    ax.set_xlabel('X Position (m)', fontsize=12)
    ax.set_ylabel('Y Position (m)', fontsize=12)
    ax.set_title(f'Vehicle Trajectories: {start_time}s - {end_time}s\n' +
                 f'Total Vehicles: {len(trajectories)} (CV: {cv_count}, NC: {nc_count})',
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_aspect('equal', adjustable='box')

    plt.tight_layout()
    return fig

def main():
    print("Parsing FCD output file...")
    print("This may take a moment for large files...")

    # Parse trajectories
    trajectories = parse_fcd_data('fcd_output.xml', start_time=300.0, end_time=600.0)

    print(f"Found {len(trajectories)} vehicles between 300s and 600s")

    # Count vehicle types
    cv_count = sum(1 for traj in trajectories.values() if traj[0][3] == 'CV')
    nc_count = len(trajectories) - cv_count
    print(f"  - Connected Vehicles (CV): {cv_count}")
    print(f"  - Non-Connected Vehicles (NC): {nc_count}")

    # Create visualization
    print("\nCreating trajectory plot...")
    fig = plot_trajectories(trajectories, start_time=300.0, end_time=600.0)

    # Save figure
    output_file = 'trajectory_300_600s.png'
    fig.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nTrajectory plot saved to: {output_file}")

    # Also create a PDF version
    pdf_file = 'trajectory_300_600s.pdf'
    fig.savefig(pdf_file, bbox_inches='tight')
    print(f"PDF version saved to: {pdf_file}")

    plt.close()

if __name__ == '__main__':
    main()

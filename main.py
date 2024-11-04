# main.py

import numpy as np
from matplotlib.animation import FuncAnimation
import matplotlib as mpl
import matplotlib.pyplot as plt  # Added import

from materials import Material
from fresnel import FresnelCalculator
from plotter import FresnelPlotter

# Enable LaTeX rendering
# mpl.rcParams['text.usetex'] = True
# mpl.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'


def define_materials() -> tuple:
    """
    Defines the materials involved in the simulation.
    
    :return: Tuple containing two Material instances.
    """
    air = Material(name="Air", mu=(1.0, 0.0), epsilon=(1.0, 0.0))
    glass = Material(name="Glass", mu=(1.0, 0.0), epsilon=(2.25, 0.0))  # n = sqrt(2.25) = 1.5
    return air, glass

def main():
    """
    Main function to set up and run the polarization animation.
    """
    incident_angle_deg = 45.0  # Constant incident angle
    theta_i_rad = np.deg2rad(incident_angle_deg)

    air, glass = define_materials()

    # Initialize Fresnel calculator with materials and incident angle
    calculator = FresnelCalculator(medium1=air, medium2=glass, theta_i_deg=incident_angle_deg)

    # Initialize the plotter
    plotter = FresnelPlotter(ray_length=1.0)
    plot_title = f"Polarization Animation at {incident_angle_deg}° Incident Angle"
    plotter.initialize(title=plot_title, medium1=air, medium2=glass)

    # Create the animation
    frames = np.linspace(0, 90, 180)  # Vary ψ from 0° to 90°
    global anim  # Make 'anim' a global variable to prevent garbage collection
    anim = FuncAnimation(
        plotter.fig, plotter.update, frames=frames,
        fargs=(theta_i_rad, air, glass, calculator),
        interval=100, blit=False
    )

    plt.show()

if __name__ == "__main__":
    main()

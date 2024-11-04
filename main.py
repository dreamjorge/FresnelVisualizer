# main.py

import numpy as np
from matplotlib import pyplot as plt  # Ensure matplotlib.pyplot is imported

from materials import Material
from plotter import FresnelPlotter

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

    air, glass = define_materials()

    # Initialize the plotter
    plotter = FresnelPlotter(ray_length=1.0)
    plot_title = f"Polarization Animation at {incident_angle_deg}° Incident Angle"
    plotter.initialize(title=plot_title, medium1=air, medium2=glass)

    # Start the animation using the plotter's animate method
    plotter.animate(theta_i_deg=incident_angle_deg, medium1=air, medium2=glass, num_frames=360, interval=50)

    # plt.show()  # Not needed since plotter.animate calls plt.show()

if __name__ == "__main__":
    main()

# plotter.py

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Arc
from matplotlib.animation import FuncAnimation
from typing import Optional

from fresnel import FresnelCalculator, FresnelCoefficients
from materials import Material

class FresnelPlotter:
    """
    Plots the incident, reflected, and transmitted rays based on Fresnel coefficients,
    including polarization states using Jones vectors.
    """
    def __init__(self, ray_length: float = 1.0):
        self.ray_length = ray_length
        self.fig, self.ax = plt.subplots(figsize=(12, 10))
        self._setup_plot()

    def _setup_plot(self):
        """
        Configures the initial plot settings.
        """
        self.ax.set_xlabel("X-axis", fontsize=16)
        self.ax.set_ylabel("Y-axis", fontsize=16)
        self.ax.grid(True)
        self.ax.axis('equal')

    def initialize(self, title: str, medium1: Material, medium2: Material):
        """
        Initializes the plot for animation with the interface and normal line.
        
        :param title: Title of the plot.
        :param medium1: First medium.
        :param medium2: Second medium.
        """
        self.ax.set_title(title, fontsize=16)
        self._draw_interface(medium1, medium2)
        self._draw_normal_line()
        limit = self.ray_length * 1.5
        self.ax.set_xlim(-limit, limit)
        self.ax.set_ylim(-limit, limit)

    def update(self, frame, theta_i_rad: float,
               medium1: Material, medium2: Material,
               calculator: FresnelCalculator):
        """
        Updates the plot for each frame in the animation.
        
        :param frame: Current frame value representing the polarization angle.
        :param theta_i_rad: Incident angle in radians.
        :param medium1: First medium.
        :param medium2: Second medium.
        :param calculator: Instance of FresnelCalculator with computed coefficients.
        """
        self.ax.cla()
        self._setup_plot()
        self.initialize(self.ax.get_title(), medium1, medium2)

        # Polarization state using Jones vector
        psi_deg = frame
        psi_rad = np.deg2rad(psi_deg)
        delta = 0  # Phase difference, can be varied if needed

        # Incident Jones vector
        incident_jones = np.array([
            np.cos(psi_rad),
            np.sin(psi_rad) * np.exp(1j * delta)
        ], dtype=complex)
        incident_jones /= np.linalg.norm(incident_jones)

        # Determine polarization state description
        polarization_state = self._determine_polarization_state(psi_deg)

        # Update the title with polarization state
        theta_i_deg = np.rad2deg(theta_i_rad)
        self.ax.set_title(
            f"Polarization Animation at {theta_i_deg:.1f}° Incident Angle\n{polarization_state}",
            fontsize=16
        )

        # Apply Jones matrices to incident Jones vector
        reflected_jones = calculator.jones_matrix_reflection @ incident_jones
        if calculator.theta_t_rad is not None:
            transmitted_jones = calculator.jones_matrix_transmission @ incident_jones
        else:
            transmitted_jones = np.array([0, 0], dtype=complex)

        # Calculate magnitudes of reflected and transmitted fields
        E_ref_magnitude = np.linalg.norm(reflected_jones)
        E_trans_magnitude = np.linalg.norm(transmitted_jones)

        # Check for Total Internal Reflection (TIR)
        tir = calculator.theta_t_rad is None

        if calculator.theta_t_rad is not None:
            theta_t_rad = calculator.theta_t_rad
        else:
            theta_t_rad = None

        # Draw rays with adjusted lengths
        self._draw_incident_ray(theta_i_rad)
        self._draw_reflected_ray(theta_i_rad, E_ref_magnitude)
        if not tir and theta_t_rad is not None:
            self._draw_transmitted_ray(theta_t_rad, E_trans_magnitude)

        # Annotate angles and coefficients
        self._annotate_angles(theta_i_rad, theta_t_rad, tir)
        self._annotate_coefficients(calculator.fresnel, psi_deg)

        # Plot Jones vectors
        self._plot_jones_vectors(theta_i_rad, theta_t_rad,
                                  incident_jones, reflected_jones,
                                  transmitted_jones, tir)

        # Handle Total Internal Reflection annotation
        if tir:
            self.ax.text(
                0.5, 0.9, "Total Internal Reflection Occurs",
                transform=self.ax.transAxes, fontsize=16,
                color='magenta', ha='center',
                bbox=dict(boxstyle="round,pad=0.5", fc="pink", alpha=0.5)
            )

    def _determine_polarization_state(self, psi_deg: float) -> str:
        """
        Determines the polarization state based on the polarization angle.

        :param psi_deg: Polarization angle in degrees.
        :return: Description of the polarization state.
        """
        psi_mod = psi_deg % 360
        if np.isclose(psi_mod % 180, 0, atol=1e-2):
            return "Linear Polarization"
        elif np.isclose(abs(psi_mod % 360), 45, atol=1e-2) or np.isclose(abs(psi_mod % 360), 135, atol=1e-2):
            return "Circular Polarization"
        else:
            return f"Elliptical Polarization (ψ = {psi_deg:.1f}°)"

    def _draw_interface(self, medium1: Material, medium2: Material):
        """
        Draws the interface line between two media.

        :param medium1: First medium.
        :param medium2: Second medium.
        """
        interface_x = [-self.ray_length, self.ray_length]
        interface_y = [0, 0]
        self.ax.plot(interface_x, interface_y, 'k-', linewidth=2,
                     label=f"Interface ({medium1.name} | {medium2.name})")

    def _draw_normal_line(self):
        """
        Draws the normal line perpendicular to the interface.
        """
        normal_length = self.ray_length * 1.2
        self.ax.arrow(0, 0, 0, normal_length, head_width=0.02, head_length=0.04,
                      fc='gray', ec='gray', linewidth=1, label='Normal')
        self.ax.arrow(0, 0, 0, -normal_length, head_width=0.02, head_length=0.04,
                      fc='gray', ec='gray', linewidth=1)

    def _draw_incident_ray(self, theta_i: float):
        """
        Draws the incident ray based on the incident angle.

        :param theta_i: Incident angle in radians.
        """
        x_start = -self.ray_length * np.sin(theta_i)
        y_start = self.ray_length * np.cos(theta_i)
        dx = self.ray_length * np.sin(theta_i) * 0.95
        dy = -self.ray_length * np.cos(theta_i) * 0.95

        self.ax.arrow(x_start, y_start, dx, dy,
                      head_width=0.03, head_length=0.05,
                      fc='blue', ec='blue', linewidth=2, label="Incident Ray")

    def _draw_reflected_ray(self, theta_i: float, magnitude: float):
        """
        Draws the reflected ray with length scaled by its magnitude.

        :param theta_i: Incident angle in radians.
        :param magnitude: Magnitude scaling factor for the reflected ray.
        """
        dx = magnitude * self.ray_length * np.sin(theta_i) * 0.95
        dy = magnitude * self.ray_length * np.cos(theta_i) * 0.95

        self.ax.arrow(0, 0, dx, dy,
                      head_width=0.03 * magnitude, head_length=0.05 * magnitude,
                      fc='red', ec='red', linewidth=2, label="Reflected Ray")

    def _draw_transmitted_ray(self, theta_t: float, magnitude: float):
        """
        Draws the transmitted ray with length scaled by its magnitude.

        :param theta_t: Transmitted angle in radians.
        :param magnitude: Magnitude scaling factor for the transmitted ray.
        """
        dx = magnitude * self.ray_length * np.sin(theta_t) * 0.95
        dy = -magnitude * self.ray_length * np.cos(theta_t) * 0.95

        self.ax.arrow(0, 0, dx, dy,
                      head_width=0.03 * magnitude, head_length=0.05 * magnitude,
                      fc='green', ec='green', linewidth=2, label="Transmitted Ray")

    def _annotate_angles(self, theta_i_rad: float, theta_t_rad: Optional[float], tir: bool):
        """
        Annotates the incident, reflected, and transmitted angles on the plot.

        :param theta_i_rad: Incident angle in radians.
        :param theta_t_rad: Transmitted angle in radians.
        :param tir: Boolean indicating if Total Internal Reflection occurs.
        """
        theta_i_deg = np.rad2deg(theta_i_rad)
        theta_r_deg = theta_i_deg  # Reflection angle equals incident angle
        theta_t_deg = np.rad2deg(theta_t_rad) if theta_t_rad is not None else None

        # Draw Incident Angle
        self._draw_angle_arc(0, 0, 90, 90 + theta_i_deg, 'blue', r'$\theta_i$')

        # Draw Reflected Angle
        self._draw_angle_arc(0, 0, 90 - theta_r_deg, 90, 'red', r'$\theta_r$')

        # Draw Transmitted Angle if not TIR
        if not tir and theta_t_deg is not None:
            self._draw_angle_arc(0, 0, 270, 270 + theta_t_deg, 'green', r'$\theta_t$')

    def _draw_angle_arc(self, x_center: float, y_center: float,
                        theta_start_deg: float, theta_end_deg: float,
                        color: str, label: str, offset: float = 0.15):
        """
        Draws an arc representing an angle and labels it.

        :param x_center: X-coordinate of the arc's center.
        :param y_center: Y-coordinate of the arc's center.
        :param theta_start_deg: Starting angle in degrees.
        :param theta_end_deg: Ending angle in degrees.
        :param color: Color of the arc and label.
        :param label: Label for the angle.
        :param offset: Offset for the arc's radius.
        """
        arc_radius = offset
        arc = Arc((x_center, y_center),
                  2 * arc_radius, 2 * arc_radius,
                  angle=0,
                  theta1=theta_start_deg,
                  theta2=theta_end_deg,
                  edgecolor=color,
                  linestyle='--',
                  linewidth=2)
        self.ax.add_patch(arc)

        mid_theta_deg = (theta_start_deg + theta_end_deg) / 2
        mid_theta_rad = np.deg2rad(mid_theta_deg)

        label_x = x_center + arc_radius * np.cos(mid_theta_rad) * 1.3
        label_y = y_center + arc_radius * np.sin(mid_theta_rad) * 1.3

        self.ax.text(label_x, label_y, label, color=color,
                     fontsize=14, ha='center', va='center')

    def _annotate_coefficients(self, fresnel_coeffs: FresnelCoefficients, psi_deg: float):
        """
        Displays the Fresnel coefficients and total R and T on the plot.

        :param fresnel_coeffs: Instance containing Fresnel coefficients.
        :param psi_deg: Polarization angle in degrees.
        """
        Rs_mag = np.abs(fresnel_coeffs.Rs)
        Rp_mag = np.abs(fresnel_coeffs.Rp)
        Ts_mag = np.abs(fresnel_coeffs.Ts)
        Tp_mag = np.abs(fresnel_coeffs.Tp)

        # Calculate total Reflectance (R) and Transmittance (T) using squared magnitudes
        psi_rad = np.deg2rad(psi_deg)
        R = (Rs_mag**2) * (np.cos(psi_rad)**2) + (Rp_mag**2) * (np.sin(psi_rad)**2)
        T = (Ts_mag**2) * (np.cos(psi_rad)**2) + (Tp_mag**2) * (np.sin(psi_rad)**2)

        # Normalize R and T to ensure R + T = 1 (if necessary)
        total = R + T
        if not np.isclose(total, 1.0, atol=1e-4):
            R /= total
            T /= total

        # Ensure R and T are within [0,1]
        R = np.clip(R, 0, 1)
        T = np.clip(T, 0, 1)

        annotation_text = (f"$R_s = {Rs_mag**2:.3f}$\n"
                           f"$R_p = {Rp_mag**2:.3f}$\n"
                           f"$T_s = {Ts_mag**2:.3f}$\n"
                           f"$T_p = {Tp_mag**2:.3f}$\n\n"
                           f"Total Reflectance $R = {R*100:.1f}\%$\n"
                           f"Total Transmittance $T = {T*100:.1f}\%$")
        self.ax.annotate(annotation_text,
                         xy=(0.05, 0.95),
                         xycoords='axes fraction',
                         fontsize=14,
                         verticalalignment='top',
                         bbox=dict(boxstyle="round,pad=0.5",
                                   fc="yellow", alpha=0.5))

    def _plot_jones_vectors(self,
                            theta_i_rad: float,
                            theta_t_rad: Optional[float],
                            incident_jones: np.ndarray,
                            reflected_jones: np.ndarray,
                            transmitted_jones: np.ndarray,
                            tir: bool):
        """
        Annotates the Jones vectors on the plot.

        :param theta_i_rad: Incident angle in radians.
        :param theta_t_rad: Transmitted angle in radians.
        :param incident_jones: Jones vector for the incident ray.
        :param reflected_jones: Jones vector for the reflected ray.
        :param transmitted_jones: Jones vector for the transmitted ray.
        :param tir: Boolean indicating if Total Internal Reflection occurs.
        """
        # INCIDENT RAY
        origin_incident = (-self.ray_length * np.sin(theta_i_rad),
                           self.ray_length * np.cos(theta_i_rad))
        incident_offset = (0.4, 0.0)
        incident_jones_latex = (
            f'$|E_{{\\text{{inc}}}}\\rangle = '
            f'[{self._format_complex(incident_jones[0])}, '
            f'{self._format_complex(incident_jones[1])}]$'
        )
        rotation_incident = np.rad2deg(theta_i_rad)
        self.ax.text(
            origin_incident[0] + incident_offset[0],
            origin_incident[1] + incident_offset[1],
            incident_jones_latex,
            color='blue', fontsize=14, ha='right', va='top',
            rotation=-(90 - rotation_incident),
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8)
        )

        # REFLECTED RAY
        origin_reflected = (0, 0)
        reflected_offset = (0.1, 0.0)
        reflected_jones_latex = (
            f'$|E_{{\\text{{ref}}}}\\rangle = '
            f'[{self._format_complex(reflected_jones[0])}, '
            f'{self._format_complex(reflected_jones[1])}]$'
        )
        rotation_reflected = np.rad2deg(theta_i_rad)
        self.ax.text(
            origin_reflected[0] + reflected_offset[0],
            origin_reflected[1] + reflected_offset[1],
            reflected_jones_latex,
            color='red', fontsize=14, ha='left', va='bottom',
            rotation=(90 - rotation_reflected),
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8)
        )

        # TRANSMITTED RAY
        if not tir and theta_t_rad is not None:
            origin_transmitted = (0, 0)
            transmitted_offset = (0.1, -0.1)
            transmitted_jones_latex = (
                f'$|E_{{\\text{{trans}}}}\\rangle = '
                f'[{self._format_complex(transmitted_jones[0])}, '
                f'{self._format_complex(transmitted_jones[1])}]$'
            )
            rotation_transmitted = np.rad2deg(theta_t_rad)
            self.ax.text(
                origin_transmitted[0] + transmitted_offset[0],
                origin_transmitted[1] + transmitted_offset[1],
                transmitted_jones_latex,
                color='green', fontsize=14, ha='left', va='top',
                rotation=-90 + rotation_transmitted,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8)
            )

    def _format_complex(self, z: complex) -> str:
        """
        Formats a complex number into a string for display.

        :param z: Complex number to format.
        :return: Formatted string.
        """
        real_part = z.real
        imag_part = z.imag
        if np.isclose(imag_part, 0, atol=1e-3):
            return f'{real_part:.2f}'
        elif np.isclose(real_part, 0, atol=1e-3):
            return f'{imag_part:.2f}j'
        else:
            sign = '+' if imag_part >= 0 else '-'
            return f'{real_part:.2f} {sign} {abs(imag_part):.2f}j'

    def animate(self, theta_i_deg: float, medium1: Material, medium2: Material, num_frames: int = 360, interval: int = 50):
        """
        Creates and displays the animation of polarization states.

        :param theta_i_deg: Incident angle in degrees.
        :param medium1: First medium.
        :param medium2: Second medium.
        :param num_frames: Number of frames in the animation.
        :param interval: Delay between frames in milliseconds.
        """
        theta_i_rad = np.deg2rad(theta_i_deg)
        calculator = FresnelCalculator(theta_i_rad, medium1, medium2)

        anim = FuncAnimation(self.fig, self.update, frames=np.linspace(0, 360, num_frames),
                             fargs=(theta_i_rad, medium1, medium2, calculator),
                             interval=interval, blit=False)

        plt.legend()
        plt.show()
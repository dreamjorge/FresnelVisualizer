# snell.py

from typing import Optional
import numpy as np

class Snell:
    """
    Handles angle calculations based on Snell's Law.
    """
    def __init__(self, n1: complex, n2: complex, theta_i_rad: float):
        """
        Initializes the Snell calculator with refractive indices and incident angle.
        
        :param n1: Refractive index of medium 1.
        :param n2: Refractive index of medium 2.
        :param theta_i_rad: Incident angle in radians.
        """
        self.n1 = n1
        self.n2 = n2
        self.theta_i_rad = theta_i_rad
        self.theta_t_rad = self.calculate_theta_t()

    def calculate_theta_t(self) -> Optional[float]:
        """
        Calculates the transmitted angle using Snell's Law.
        Returns None if Total Internal Reflection occurs.
        """
        sin_theta_i = np.sin(self.theta_i_rad)
        sin_theta_t = (self.n1 / self.n2) * sin_theta_i
        sin_theta_t_mag = np.abs(sin_theta_t)
        if sin_theta_t_mag > 1:
            return None  # Total Internal Reflection
        # Handle real angles; complex angles require more advanced treatment
        return np.arcsin(sin_theta_t.real)

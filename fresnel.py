# fresnel.py

from dataclasses import dataclass
import numpy as np
from typing import Optional

from snell import Snell
from materials import Material

@dataclass
class FresnelCoefficients:
    """
    Stores Fresnel reflection and transmission coefficients for s and p polarizations.
    """
    Rs: complex
    Rp: complex
    Ts: complex
    Tp: complex

class FresnelCalculator:
    """
    Calculates Fresnel coefficients and constructs Jones matrices between two media.
    """
    def __init__(self, medium1: Material, medium2: Material, theta_i_deg: float):
        """
        Initializes the FresnelCalculator with two media and an incident angle.
        
        :param medium1: Incident medium.
        :param medium2: Transmitting medium.
        :param theta_i_deg: Incident angle in degrees.
        """
        self.medium1 = medium1
        self.medium2 = medium2
        self.theta_i_deg = theta_i_deg
        self.theta_i_rad = np.deg2rad(theta_i_deg)

        # Initialize Snell's Law calculator
        self.snell = Snell(
            n1=self.medium1.refractive_index,
            n2=self.medium2.refractive_index,
            theta_i_rad=self.theta_i_rad
        )

        # Angles
        self.theta_r_rad = self.theta_i_rad  # Reflection angle equals incident angle
        self.theta_t_rad = self.snell.theta_t_rad  # Transmitted angle

        # Fresnel coefficients
        self.fresnel = self.compute_fresnel_coefficients()

        # Jones matrices
        self.jones_matrix_reflection = self.construct_jones_matrix_reflection()
        self.jones_matrix_transmission = self.construct_jones_matrix_transmission()

    def compute_fresnel_coefficients(self) -> FresnelCoefficients:
        """
        Computes the Fresnel reflection and transmission coefficients.
        """
        n1 = self.medium1.refractive_index
        n2 = self.medium2.refractive_index

        theta_i = self.theta_i_rad
        theta_t = self.theta_t_rad

        if theta_t is None:
            # Total Internal Reflection: No transmission
            Rs = Rp = 1 + 0j  # Perfect reflection
            Ts = Tp = 0 + 0j
            return FresnelCoefficients(Rs=Rs, Rp=Rp, Ts=Ts, Tp=Tp)

        cos_theta_i = np.cos(theta_i)
        cos_theta_t = np.cos(theta_t)

        Rs = (n1 * cos_theta_i - n2 * cos_theta_t) / (n1 * cos_theta_i + n2 * cos_theta_t)
        Rp = (n2 * cos_theta_i - n1 * cos_theta_t) / (n2 * cos_theta_i + n1 * cos_theta_t)
        Ts = 1 + Rs
        Tp = 1 + Rp

        return FresnelCoefficients(Rs=Rs, Rp=Rp, Ts=Ts, Tp=Tp)

    def construct_jones_matrix_reflection(self) -> np.ndarray:
        """
        Constructs the Jones matrix for reflection.
        """
        return np.array([
            [self.fresnel.Rs, 0],
            [0, self.fresnel.Rp]
        ], dtype=complex)

    def construct_jones_matrix_transmission(self) -> np.ndarray:
        """
        Constructs the Jones matrix for transmission.
        """
        return np.array([
            [self.fresnel.Ts, 0],
            [0, self.fresnel.Tp]
        ], dtype=complex)

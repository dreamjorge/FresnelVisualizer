# fresnel.py

import numpy as np
from dataclasses import dataclass
from typing import Optional

from materials import Material

@dataclass
class FresnelCoefficients:
    Rs: float  # Reflectance for s-polarization
    Rp: float  # Reflectance for p-polarization
    Ts: float  # Transmittance for s-polarization
    Tp: float  # Transmittance for p-polarization
    diffuse: float  # Diffuse reflection component
    specular: float  # Specular reflection component

class FresnelCalculator:
    """
    Calculates Fresnel coefficients and Fresnel vectors for incident,
    reflected, and transmitted rays based on polarization.
    """
    def __init__(self, theta_i_rad: float, medium1: Material, medium2: Material):
        """
        Initializes the FresnelCalculator with incident angle and media.

        :param theta_i_rad: Incident angle in radians.
        :param medium1: First medium (incident medium).
        :param medium2: Second medium (transmitting medium).
        """
        self.theta_i_rad = theta_i_rad
        self.medium1 = medium1
        self.medium2 = medium2
        self.theta_t_rad: Optional[float] = None  # Transmitted angle
        self.fresnel: FresnelCoefficients = self.calculate_fresnel_coefficients()
        self.jones_matrix_reflection = self._compute_jones_matrix_reflection()
        self.jones_matrix_transmission = self._compute_jones_matrix_transmission()
        
        # Initialize Fresnel vectors
        self.incident_vector = None
        self.reflected_vector = None
        self.transmitted_vector = None

    def calculate_fresnel_coefficients(self) -> FresnelCoefficients:
        n1 = self.medium1.refractive_index
        n2 = self.medium2.refractive_index
        theta_i = self.theta_i_rad

        # Snell's Law
        sin_theta_t = n1 / n2 * np.sin(theta_i)
        if np.abs(sin_theta_t) > 1.0:
            # Total Internal Reflection
            self.theta_t_rad = None
            return FresnelCoefficients(Rs=1.0, Rp=1.0, Ts=0.0, Tp=0.0, diffuse=0.0, specular=1.0)
        else:
            self.theta_t_rad = np.arcsin(sin_theta_t)

        theta_t = self.theta_t_rad

        # Fresnel Equations for s-polarization
        Rs = ((n1 * np.cos(theta_i) - n2 * np.cos(theta_t)) /
            (n1 * np.cos(theta_i) + n2 * np.cos(theta_t))) ** 2
        # Fresnel Equations for p-polarization
        Rp = ((n2 * np.cos(theta_i) - n1 * np.cos(theta_t)) /
            (n2 * np.cos(theta_i) + n1 * np.cos(theta_t))) ** 2

        # Transmittance
        Ts = 1 - Rs
        Tp = 1 - Rp

        # Diffuse and Specular Components
        diffuse = 0.1  # Example value for diffuse reflection (adjust as needed)
        specular = 0.9  # Example value for specular reflection (adjust as needed)

        return FresnelCoefficients(Rs=Rs, Rp=Rp, Ts=Ts, Tp=Tp, diffuse=diffuse, specular=specular)


    def _compute_jones_matrix_reflection(self) -> np.ndarray:
        """
        Computes the Jones matrix for reflection based on Fresnel coefficients.

        :return: 2x2 Jones matrix for reflection.
        """
        Rs = self.fresnel.Rs
        Rp = self.fresnel.Rp
        return np.array([
            [np.sqrt(Rs), 0],
            [0, np.sqrt(Rp)]
        ], dtype=complex)  

    def _compute_jones_matrix_transmission(self) -> np.ndarray:
        """
        Computes the Jones matrix for transmission based on Fresnel coefficients.

        :return: 2x2 Jones matrix for transmission.
        """
        Ts = self.fresnel.Ts
        Tp = self.fresnel.Tp
        return np.array([
            [np.sqrt(Ts), 0],
            [0, np.sqrt(Tp)]
        ], dtype=complex)  

    def set_polarization_angle(self, psi_deg: float):
        """
        Sets the polarization angle and updates Fresnel vectors accordingly.

        :param psi_deg: Polarization angle in degrees.
        """
        psi_rad = np.deg2rad(psi_deg)
        # Incident Jones vector assuming linear polarization at angle psi
        incident_jones = np.array([
            np.cos(psi_rad),
            np.sin(psi_rad)
        ], dtype=float)  # Changed dtype to float
        incident_jones /= np.linalg.norm(incident_jones)

        self.incident_vector = incident_jones
        self.reflected_vector = self.jones_matrix_reflection @ incident_jones

        if self.theta_t_rad is not None:
            self.transmitted_vector = self.jones_matrix_transmission @ incident_jones
        else:
            self.transmitted_vector = np.array([0.0, 0.0], dtype=float)
            
    def calculate_total_reflectance_and_transmittance(self, psi_deg: float) -> tuple:
        """
        Calculates the total reflectance and transmittance, including diffuse and specular components.

        :param psi_deg: Polarization angle in degrees.
        :return: Tuple of total reflectance and total transmittance.
        """
        psi_rad = np.deg2rad(psi_deg)
        R_specular = (self.fresnel.Rs * (np.cos(psi_rad)**2) + self.fresnel.Rp * (np.sin(psi_rad)**2)).real
        T_specular = (self.fresnel.Ts * (np.cos(psi_rad)**2) + self.fresnel.Tp * (np.sin(psi_rad)**2)).real

        # Calculate total reflectance and transmittance
        R_total = R_specular + self.fresnel.diffuse
        T_total = T_specular * (1 - self.fresnel.diffuse)

        # Ensure values are within [0, 1]
        R_total = np.clip(R_total, 0, 1)
        T_total = np.clip(T_total, 0, 1)

        return R_total, T_total
    
    # fresnel.py

    def calculate_diffuse_reflection(self, incident_intensity: float, angle_of_incidence: float) -> float:
        """
        Calculates the intensity of the diffuse reflection using Lambert's Cosine Law.

        :param incident_intensity: Intensity of the incident light.
        :param angle_of_incidence: Angle of incidence in radians.
        :return: Intensity of the diffusely reflected light.
        """
        rho = 0.1  # Example value for diffuse reflectance coefficient (adjust based on material)
        diffuse_intensity = incident_intensity * rho * np.cos(angle_of_incidence)
        return max(diffuse_intensity, 0)  # Ensure the intensity is non-negative


    def get_incident_vector(self) -> Optional[np.ndarray]:
        """
        Returns the incident Jones vector.

        :return: 2-element numpy array representing the incident Jones vector.
        """
        return self.incident_vector

    def get_reflected_vector(self) -> Optional[np.ndarray]:
        """
        Returns the reflected Jones vector.

        :return: 2-element numpy array representing the reflected Jones vector.
        """
        return self.reflected_vector

    def get_transmitted_vector(self) -> Optional[np.ndarray]:
        """
        Returns the transmitted Jones vector.

        :return: 2-element numpy array representing the transmitted Jones vector.
        """
        return self.transmitted_vector

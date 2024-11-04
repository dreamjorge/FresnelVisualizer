# materials.py

from dataclasses import dataclass
from typing import Tuple
import numpy as np

@dataclass
class Material:
    """
    Represents an optical material with its magnetic permeability and electric permittivity.
    """
    name: str
    mu: Tuple[float, float]         # (mu_r, mu_i)
    epsilon: Tuple[float, float]    # (epsilon_r, epsilon_i)

    @property
    def refractive_index(self) -> complex:
        """
        Calculates the complex refractive index n = sqrt(mu * epsilon).
        """
        mu_complex = complex(*self.mu)
        epsilon_complex = complex(*self.epsilon)
        n_squared = mu_complex * epsilon_complex
        return np.sqrt(n_squared)

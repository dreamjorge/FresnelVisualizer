
# FresnelVisualizer

FresnelVisualizer is a Python project that simulates the behavior of light as it reflects and transmits between different materials, incorporating concepts such as specular and diffuse reflection. It uses Fresnel equations to model the interaction of polarized light with different media and visualizes these phenomena with animated plots.

## Features

- **Fresnel Coefficients Calculation**: Computes the reflectance and transmittance for s- and p-polarized light.
- **Diffuse Reflection**: Models and visualizes the scattering of light on rough surfaces using Lambert's Cosine Law.
- **Polarization State Animation**: Shows how the polarization state of light changes upon reflection and transmission.
- **Interactive Visualization**: Uses `matplotlib` animations to provide a dynamic visualization of light behavior.

## Requirements

- Python 3.6 or later
- `numpy`
- `matplotlib`

You can install the dependencies using:

```bash
pip install -r requirements.txt
```

## Structure

- `main.py`: Main script to run the visualization.
- `fresnel.py`: Contains the `FresnelCalculator` class for calculating Fresnel coefficients and handling light behavior.
- `materials.py`: Defines the `Material` class for representing optical properties of different media.
- `plotter.py`: Handles the visualization of light behavior, including both specular and diffuse reflections.

## Usage

1. **Run the Simulation**:
   ```bash
   python main.py
   ```
   This will open a window displaying an animated visualization of light reflection and transmission.

2. **Modify Parameters**: You can adjust the incident angle, material properties, or polarization state in the `main.py` script to explore different scenarios.

## Explanation of Key Concepts

### 1. Fresnel Equations
These equations describe how light is partially reflected and partially transmitted at an interface between two materials with different refractive indices. The reflectance and transmittance depend on the polarization of the light and the angle of incidence.

### 2. Diffuse Reflection
Modeled using Lambert's Cosine Law, diffuse reflection occurs when light strikes a rough surface and is scattered uniformly in all directions. This is in contrast to specular reflection, which occurs on smooth surfaces.

### 3. Polarization
The simulation uses Jones vectors to represent the polarization state of light and how it changes upon interaction with the interface.

## Future Enhancements

- Add support for more complex materials with wavelength-dependent refractive indices.
- Include a GUI for easier parameter adjustments.
- Enhance the visualization with more realistic light scattering effects.

## License

This project is open-source and available under the MIT License.

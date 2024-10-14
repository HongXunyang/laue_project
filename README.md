# Contour finding, Sample re-orientation and all that
This is a sub-project of the larger project *Automizing Sample re-orientation for Neutron Scattering*. The main focus of this subproject is: 
1. Digitalizing Samples and sample holder (assuming there are ~ 100 samples on the holder)
2. Determining the Orientation of each sample (asssuming only phi offset, this is a orientation-finding project)
3. Finding the countour of each sample using open CV (Jasmin's project)
4. Generating the sample contour after re-orientation (Jasmin's project)
5. Finding the most compact configuration of samples on the holder (Sissi's project)
6. If possible, generate a CAD file of the sample holder engraved with the re-oriented sample contours. 

The expected outcome of this project would be with a GUI where the user can upload:
1. sample holder image
2. Laue image of each sample 

the program will automaticall detect the contour of each sample using the sample holder image (open cv library); and determine the offset of each sample using the Laue image of each sample (orientation-finding project). and then produce the re-oriented sample contour. 

After this, the program will find the most compact configuration, and ideally generate a CAD file of the engraved sample holder (if that's possibl).

**Assumptions to simplify the prolem**
- Samples are flakes and c-axis is well-aligned. The only offset is in phi.


## Overview

This project provides a Python implementation for managing samples and their placement on a sample holder, a critical part of experiments involving multiple crystal samples. It defines two primary classes:
- `Sample`: Represents a single sample with its attributes such as position, contour, and orientation.
- `SampleHolder`: Manages a collection of `Sample` objects on a grid, allowing for visualization and manipulation of samples.

## Project Structure

The project is structured as follows:

```
workspace/
   ├── packages/
   │   ├── __init__.py                # Package-level initialization for the 'packages' module
   │   ├── sample_class.py             # Defines the Sample class
   │   ├── sampleholder_class.py       # Defines the SampleHolder class
   ├── scripts/
   ├── __init__.py                     # Root-level package initialization (empty)
   ├── main.py                         # main file
   └── README.md                       # Project documentation (this file)
```

### Main Components

- **Sample (`sample_class.py`)**:
  - A class representing individual samples with properties like `id`, `name`, `index`, `contour`, and `position`.
  - Future work includes defining the origin of the coordinate system and the sample itself.

- **SampleHolder (`sampleholder_class.py`)**:
  - A class representing the sample holder, designed as a grid where samples can be placed.
  - Provides methods to add, remove, and manage samples, along with the ability to visualize the sample holder's grid and samples.

## Installation

To get started with the project:

1. Clone this repository.
2. Install the required packages, such as NumPy and Matplotlib, using the following command:
   ```bash
   pip install -r requirements.txt
   ```
3. Ensure that you are running Python 3.6 or later.

## Usage

The core functionality is demonstrated in the `main.py` file. Here’s how you can run the code:

### Running the Script

1. Navigate to the `workspace/` directory:
   ```bash
   cd workspace
   ```

2. Run the `main.py` script:
   ```bash
   python main.py
   ```

This script imports the `Sample` and `SampleHolder` classes from the `packages/` directory and allows you to add samples to the holder and visualize their placement.

### Sample Code

Here’s an example of how to use the `Sample` and `SampleHolder` classes:

```python
from packages import Sample, SampleHolder

# Create a sample holder with a 10x10 grid
holder = SampleHolder(grid_size=(10, 10))

# Create a sample with id=1, name='sample1', placed at index (1, 2)
sample = Sample(id=1, name='sample1', index=(1, 2))

# Add the sample to the sample holder
holder.add_sample(sample)

# Visualize the sample holder
holder.visualize()
```

### Visualization

The `SampleHolder` class has a `visualize` method that plots the current state of the sample holder. It uses Matplotlib to display the grid and indicates the presence of samples with dots.

Each sample's ID is labeled next to the dot, and if a sample has a defined `phi_offset`, it is visualized with a line.

## To-Do List

- Define the origin of the coordinate system.
- Define the origin of the sample itself for better positioning and orientation.

---

### Future Improvements
- Incorporate better handling of sample re-orientation and close packing strategies.
- Add more robust error handling and validation for sample placement.
- Extend visualization with additional features such as 3D plotting or alignment indicators.


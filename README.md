# Overview
This is a sub-project of the larger project *Automizing Sample re-orientation for Neutron Scattering*. The main focus of this subproject is: 
1. [ **Finished** ] Digitalizing Samples and sample holder (assuming there are ~ 100 samples on the holder). This is done in the `classes/` packages. contour, sample, and sample holder are defined as classes. 
2. [ **To-do** ] Determining the Orientation of each sample (asssuming only phi offset, this is a orientation-finding project) by sweeping through the samples on the sample holder using Laue diffraction. This involves: (1) Automatic control of the robot arm to sweep through the samples, involving coordinate transformation; (2) Auto control of the Laue machine, involving both taking X-ray image and saving images; (3) Auto detection of the orientation of the sample based on the Laue images; (4) Based on everything above, assigning the orientation to each sample, re-oriente the sample contour.
3. [ **Finished** ] Finding the countour of each sample using open CV (Post Jasmin's project). This is done in the `contour_finding/` package. It loads the image and generate sample holder, samples, and contours objects.
5. [ **Finished** ] Finding the most compact configuration of samples on the holder (Post Sissi's project). Currently this is done in the `close_packing/` package. Simulated annealing + gravity effect is used to find the compact configuration.
6. [ **To-do** ] If possible, generate a CAD file of the sample holder engraved with the re-oriented sample contours. 
7. [ **In progress** ] Create a GUI to unify all the above. This is under construction in the `gui/` package. 

**Expected workflow**: 
- The User upload only the image of the sample holder (with samples on).
- The program will detect the contour and the position of each sample using the sample holder image; it tells the Laue machine and robot arm how to work out the orientation of each sample. Then it produces the *re-oriented* sample contours. 
- After this, the program will find the most compact configuration
- Ideally it willgenerate a CAD file of the engraved sample holder.

**Assumptions to simplify the prolem**
- Samples are flakes and c-axis is well-aligned. The only offset is in phi.
- Sample contour is approimated by its convex hull. 


# Detailed Dig-in
## Project Structure
The project is structured as follows:
```
workspace/
   ├── classes/
   │   ├── __init__.py           # Package-level initialization
   │   ├── class_contour.py      # Defines the Contour class
   │   ├── class_sample.py       # Defines the Sample class
   │   ├── class_sampleholder.py # Defines the SampleHolder class
   │   ├── helper_functions.py   # internal helper functions
   │   ├── visualization.py      # internal visualization functions
   ├── close_packing/
   │   ├── __init__.py           # Package-level initialization
   │   ├── optimization.py       # close packing optimization
   │   ├── helper_functions.py   # internal helper functions
   │   ├── visualization.py      # internal visualization functions
   ├── contour_finding/
   │   ├── __init__.py           # Package-level initialization
   │   ├── image_processing.py   # image processing functions
   ├── config/
   │ ├── config.json             # parameters: image processing or close packing
   │ ├── gui_styles.css          # CSS for the GUI
   │ ├── stylesheet.json         # plot setting parameters
   ├── gui/
   │   ├── __init__.py           # Package-level initialization
   │   ├── main_window.py        # main window widget
   │   ├── image_display.py      # image drop-in and display widget
   │   ├── matplotlib_canvas.py  # results display panel widget
   │   ├── helper_functions.py   # helper functions only for internal use
   ├── utils/
   │   ├── __init__.py           # Package-level initialization 
   │   ├── visualization.py      # general visualization tools/functions
   ├── main_close _packing.py    # run this to test close packing
   ├── main_contour _finding.py  # run this to test contour finding
   ├── main_gui.py               # run this to create a GUI
   ├── requirements.txt          # required libraries
   └── README.md                 # Project documentation (this file)
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


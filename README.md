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
(*Updated on 2024-10-20*)  Currently I am working on the (1) Contour finding; (2) Close packing; (3) GUI design parts. 

## Folder Structure
The project folder is structured as follows:
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
   │   ├── config.json           # parameters: image processing or close packing
   │   ├── gui_styles.css        # CSS for the GUI
   │   ├── stylesheet.json       # plot setting parameters
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
### Data structure
the contour, sample and sample holder are defined as classes in the `classes/` package.

**Contour**: the contour class is defined in `class_contour.py`. It has the following attributes:
- `contour`: (`cv2` contour) When no nesting allowed, it's a `(N, 1, 2)` numpy array, `int32` before the relocation/reorientation, `float32` after
- `hull`: this is the convex hull of the contour. It's a `(N, 1, 2)` numpy array, `int32` before the relocation/reorientation, `float32` after. Usually it has less points than the contour.
- `polygon`: (`shapely:Polygon` object) It's the polygon of the hull 
- `vertices`: almost the same as hull, (N, 2) numpy array, `float32` from the beginning. 
- `area`: the area of the hull
- `center`: (`(1,2)` numpy array) The centroid/center of the hull, calculated by taking the mean of the vertices.
- `sample`: (`Sample` object) The sample that the contour belongs to.
- `id`: (`int`) The id of the parent sample object.

and the following methods:
- `reorient`: (*not yet implemented*) reorient the contour, keep the center unchanged. This should be performed before the relocation. everything will be converted to `float32`.
- `relocate`: relocate the contour, everything will be converted to `float32`.

-----------------------

**Sample**: The `Sample` class is defined in `class_sample.py`. It represents a sample that can be reoriented and relocated. The class has the following attributes:
- `id`: (`int`) The unique identifier of the sample. 
- `name`: (`str`) The name of the sample (default is "sample").
- `grid_index`: The grid index of the sample on the sample holder (e.g., `(1, 2)`). Currently not used.
- `contour_original`: (`Contour` object) The original contour of the sample before reorientation/relocation.
- `contour_new`: The updated contour of the sample after reorientation/relocation
- `position_original`: (`numpy.ndarray`) The center of the `contour_original`
- `position_new`: The center of the `contour_new`
- `phi_offset`: (`float` in degree) The offset angle (in degrees) used for reorienting the sample, counter-clockwise direction.
- `is_reoriented`: A boolean indicating whether the sample has been reoriented.
- `is_relocated`: A boolean indicating whether the sample has been relocated.

The class also has the following methods:
- `__str__`: Returns a string representation of the sample, including its ID and original position.
- `reorient`: (*Update required*) Reorients the sample according to the `phi_offset` value. 
- `relocate`: Relocates the sample to its new position (`position_new`). Purely translation.

---------------------
**SampleHolder**: The `SampleHolder` class is defined in `class_sampleholder.py`. It represents a container for managing multiple samples and includes functionality to add, access, and manage samples. The class has the following attributes:
- `name`: (`str`) The name of the sample holder
- `size`: The dimensions of the sample holder (in mm). Set to `None` initially.
- `samples_list`: A list storing the sample objects added to the sample holder.
- `_id2sample`: A dictionary mapping sample IDs to their corresponding sample objects.
- `_id2list_index`: A dictionary mapping sample IDs to their list index in `samples_list`.

The class also has the following methods:
- `__str__`: Returns a string representation of the sample holder, including its name and the number of samples.
- `print_samples`: Prints information about each sample on the sample holder.
- `add_sample`: Adds a `Sample` object to the sample holder and assigns it to the sample holder.
- `id2sample`: Returns a sample object given its ID.
- `id2list_index`: Returns the list index of a sample given its ID.
- `list_index2id`: Returns the ID of a sample given its list index. Returns `None` if the index is out of range.
- `number_samples`: Returns the number of samples currently on the sample holder.
-----------------------------------

**FunctionalSampleHolder**: The `FunctionalSampleHolder` class is a subclass of `SampleHolder` and adds functionality for reorienting and relocating samples. It has the following methods:
- `assign_phi_offset`: Assigns a rotation angle (`phi_offset`) to a sample, which can later be used for reorientation.
- `assign_phi_offset_by_index`: Assigns a `phi_offset` to a sample based on its ID or list index.
- `reorient_sample`: Reorients a sample according to its `phi_offset` and updates the sample's contour.
- `reorient_sample_by_index`: Reorients a sample based on its ID or list index.
- `reorient`: Reorients all samples currently on the sample holder.
- `relocate_sample`: Relocates a sample to a specified position.
- `relocate_sample_by_index`: Relocates a sample based on its ID or list index to a specified position.

The `FunctionalSampleHolder` class enhances the basic `SampleHolder` by providing methods for adjusting the orientation and position of each sample to achieve specific spatial arrangements, such as close packing.

## Installation
To get started with the project:
1. Install the required packages, such as NumPy and Matplotlib, using the following command:
   ```bash
   pip install -r requirements.txt
   ```
2. Ensure that you are running Python 3.6 or later.
## Usage
Under development. `main_contour_finding` can be used to test the contour finding part. `main_close_packing` can be used to test the close packing part. `main_gui` can be used to test the GUI part. 

Currently both `main_contour_finding.py` and `main_close_packing.py` requires manual input of the image path. You also need your own image to test the contour finding part. This is not super optimal at the moment. Only `main_gui.py` is fully functional.

### Running the Script
1. Navigate to the `workspace/` directory:
   ```bash
   cd workspace
   ```
2. Run the `main_gui.py` script:
   ```bash
   python main.py
   ```
3. The GUI window will open. Drag and drop an image file onto the window to start the process. 
4. Click *select points* to select three points for the stripes, and three points for the background. Keep an eye on the output window for the instructions.
5. Click *Image processing* to process the image.
6. You will get the contours of the samples.
7. Nothing else you can do with this GUI for the moment. 

---

### Future Improvements
- Incorporate better handling of sample re-orientation and close packing strategies.
- Add more robust error handling and validation for sample placement.
- Extend visualization with additional features such as 3D plotting or alignment indicators.

----------------------------
# Appendix
### Simulated Annealing for Close Packing

The `batch_optimization` and `optimization` functions are implemented to optimize the configuration of polygons representing samples on a sample holder. The goal of this optimization is to achieve close packing of the samples by minimizing the area of the convex hull that contains all samples. This process uses a simulated annealing approach, the temperature parameters of which controls the random fluctuations of the configuration. 

1. **Initial Setup**:
   - The initial temperature is set to a high value (default 1000) and gradually reduced to one percent of its initial value over the course of iterations.
   - A list of vertices representing the samples is prepared. If `is_rearrange_vertices` is `True`, the initial positions are rearranged randomly to facilitate a better starting point for optimization.

2. **Temperature and Step Size Decay**:
   - The temperature is linearly decreased after each iteration, which reduces the probability of accepting worse configurations as the optimization progresses. 
   - The step size, which determines how far a polygon can move in each iteration, also decreases over time.

3. **Iteration Process**:
   - In each iteration, a random polygon is selected, and a movement vector is generated. The movement may be influenced by gravitational forces between the polygons if `is_gravity` is `True`. Otherwise, the movement is purely random within the step size limit.
   - A new position for the selected polygon is proposed by adding the movement vector to the original position.
   - The algorithm checks if the new configuration is valid, i.e., if it results in any overlap between polygons. If no overlap, go to the next step. If overlap is detected, attempt to repeat the last step and generate another movement again (3 attempts max)
   - If the new configuration is valid, it further checks the current area of the convex hull. If the area drops, accept. But even if the area increases, it may still be accepted with a probability related to the temperature (high temperature -> more likely to accept worse solutions).

4. **Acceptance Criteria**:
   - If the new configuration has a smaller area than the previous one, it is always accepted.
   - If the area is larger, the new configuration is accepted with a probability given by `exp(-(new_area - area) / temperature)`. This probability decreases as the temperature drops, making the algorithm less likely to accept worse solutions as it converges.

A suitable order of magnitude for the temperature should be roughly: `scale * step_size`. Where `Scale` is the scale of the overall convex hull. It shouldn't be 10 times larger or smaller than this value.

5. **Updating Sample Positions**:
   - If both checks are passed, the new configuration or the new position of the polygon is accepted

6. **Batch Optimization**:
   - The `batch_optimization` function allows running multiple instances of the optimization process in parallel, each with potentially different starting conditions. This approach increases the chances of finding a better global solution.
   - After completing the optimizations, the configurations are plotted to visually compare their areas and determine the most optimal arrangement.

**Parameters**:
- `number_of_iteration`: Controls how many iterations the algorithm runs, which affects convergence quality.
- `step_size`: Sets the maximum distance a sample can move in each iteration, which influences how quickly the configuration changes.
- `temperature`: The initial temperature, which controls the randomness of the movement and acceptance of worse solutions.
- `is_gravity`: Determines whether gravitational forces between samples affect the movement, potentially guiding the optimization towards more compact arrangements.
- `is_update_sampleholder`: If `True`, the sample holder's internal representation is updated after optimization to reflect the new positions of the samples.

This simulated annealing-based optimization effectively allows for exploring different configurations to achieve a close-packed arrangement of samples on a sample holder, balancing randomness and systematic reduction of overlap to reach a stable and efficient configuration.


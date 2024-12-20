# Overview
This is a sub-project of the larger project *Automizing Sample re-orientation for Neutron Scattering*. The main focus of this subproject is: 
1. [ **Finished** ] Digitalizing Samples and sample holder (assuming there are ~ 100 samples on the holder). This is done in the `classes/` packages. contour, sample, and sample holder are defined as classes. 
2. [ **To-do** ] Determining the Orientation of each sample (asssuming only phi offset, this is a orientation-finding project) by sweeping through the samples on the sample holder using Laue diffraction. This involves: (1) Automatic control of the robot arm to sweep through the samples, involving coordinate transformation; (2) Auto control of the Laue machine, involving both taking X-ray image and saving images; (3) Auto detection of the orientation of the sample based on the Laue images; (4) Based on everything above, assigning the orientation to each sample, re-oriente the sample contour.
3. [ **Finished** ] Finding the countour of each sample using open CV (Post Jasmin's project). This is done in the `contour_finding/` package. It loads the image and generate sample holder, samples, and contours objects.
5. [ **Finished** ] Finding the most compact configuration of samples on the holder (Post Sissi's project). Currently this is done in the `close_packing/` package. Simulated annealing + gravity effect is used to find the compact configuration.
6. [ **Finished** ] Generate a CAD file of the sample holder engraved with the re-oriented sample contours. This is in the `to_cad/` package.
7. [ **Finished** ] Create a GUI to unify all the above. This is in the `gui/` package. 

**Expected workflow**: 
- The User upload only the image of the sample holder (with samples on).
- The program will detect the contour and the position of each sample using the sample holder image; it tells the Laue machine and robot arm how to work out the orientation of each sample. Then it produces the *re-oriented* sample contours. 
- After this, the program will find the most compact configuration
- Ideally it willgenerate a CAD file of the engraved sample holder.

**Assumptions to simplify the prolem**
- Samples are flakes and c-axis is well-aligned. The only offset is in phi.
- Sample contour is approimated by its convex hull. 


# Installation and Setup
### Prerequisites
Ensure that you are running Python 3.6 or later. The program can be run either through Python scripts or the GUI interface.

### Installation Steps
1. Clone the repository:
   ```bash
   git clone git@github.com:HongXunyang/laue_project.git
   cd laue_project/
   ```

2. Install the package and its dependencies:
   ```bash
   pip install -e .
   ```
   This will install the package in "editable" or "development" mode, which means:
   - All dependencies are properly installed
   - Changes to the source code take effect immediately without reinstalling
   - The package can be imported from any directory


4. Create a folder for temporary outputs:
   ```bash
   mkdir temporary_output
   ```

# Usage Instructions
### GUI Interface
The GUI now features an improved workflow with ordered buttons and new functionality. To use the GUI:

1. Launch the GUI:
   ```bash
   python src/main_gui.py
   ```

2. Follow the button sequence as indicated by their names. The typical workflow is:

   a. "**Upload Image**" - Drag and drop an image. Please crop the image before hand, removing
   everything that could interfere with the detection of the sample holder. For example, the edge of
   the sample holder, the numbers on the sample holder, etc.
   
   b. "**Select Points**" - Define stripe and background colors. Keep an eye on the output log panel to see the instructions and the progress. The user will be required to select three points on the image (directly click on the image) to define the color vectors for stripe detection; after this, the the user will need to select another three points to define the background color vectors. See [Image processing](#image-processing) for more details. 
   
   c. "**Process Image**" - Process the image and detect contours. On the top of the middle panel, the user can define the contour-finding parameters. When left empty, the parameters will be set to default values (indicated by the gray placeholder text). Once the parameters are set, click the "**Process Image**" button to start the contour-finding process. The program will display the processed image with the detected contours. If the results are not satisfactory, the user might need to quit the GUI and restart it again to reprocess the image with different parameters. (*I know I know this is not ideal but as I said the GUI is still under development...*)
   
   d. "**Input Phi Offset**" - Use the drop-in widget on the top right to specify orientation
   offsets. The phi offset is determined by the laue measurement on each samples. This data should
   be stored in a .TXT file. the format is like this: 
   ```txt
   10,15,20,5,-45,30
   ```
   The ordering is the same as the order of the indeces assigned directly by the GUI after the
   contour finding process. The phi offset is in degree, positive value means the angle is counter-clockwise; negative value means the angle is clockwise.
   
   e. "**Start Close Packing**" - Begin optimization process. After the contours are found, and
   before starting the close packing process, the user is suggested to adjust the parameters for the
   close packing in the middle panel. P.S. For now, I will suggest `NO. of system = 3` and
   `number_of_iterations = 3000` to start a trial run before any commitment. (*In the future, a
   "test close packing" need to be implemented for the user to find out the best optimization
   parameters...*) Once the parameters are set, click the "**Start Close Packing**" button to begin the optimization process. On the output log panel, the progress of the optimization will be displayed. The final optimized configuration and the evolution of the optimization process will be shown on the bottom left panel. You can re-run the optimization process by re-clicking the "**Start Close Packing**" button again. You don't need to quit the GUI. 


   f. "**Convert to CAD**" - Convert the optimized configuration to a CAD file. the parameter
   `mm_per_pixel` is the conversion factor between pixels and millimeters. This value is not
   necessary if you are not planning to 3D print this out.

   g. "**Generate Report**" - Generate a report of the optimization process

3. Most of the intermediate and final outputs will be saved in the `temporary_output/` folder. The
   report in .HTML format will be opened automatically once generated. The user can also find the
   report in the `temporary_output/report_*/` folder by draging the .HTML file into the web browser.

### Python Script
It's recommended to run the python script within an IDE (e.g., VScode or PyCharm) to better understand the procedure and debug if necessary. The following steps outline how to run the script:

The script `src/main.py` in the root directory contains:
- (1) image processing and contour finding;
- (2) close packing optimization; 
- (3) converting the optimized configuration to a CAD file; 
- (4) visualization by animating the close packing process. 

Please follow the steps below:

**[Trial Run] Usage**: 

Before doing any real application, users are suggested to go for a "trial run" to get a better understanding of the program. In the trial run, the image to process is a classic example. This image allows for a easy and quick contour finding. In this trial run, the users do not need to worry about the contour finding. The key point of this run is to figure out the best parameters for the optimization process.

The dictionary `STEP_CONTROL` in the script controls the steps of the program. The following steps are the recommended steps: 

1. Run the scripts with `STEP_CONTROL["test"] = True`. This will generate an animation of the close packing process. Look closely at the optimization process and determine the best parameters for the contour finding. Users are encouraged to adjust these parameters for the `optimization` function: 
    - `number_of_iterations: int`
    - `step_size: int`
    - `temperature: float`
    - `gravity_multiplier: float`
    - `gravity_off_at: int`

Please refer to section [Close Packing](#close-packing) for more detailed information about these parameters.

2. Once you find the approprite parameters:
    - set `STEP_CONTROL["test"] = False`, and everything else to `True`. When this set to `False`, a more serious optimization process will be launched, i.e. `batch_optimization` where multiple systems will be involved and get optimized in parallel. This will help to find a better configuratio for the system. 
    - modify the corresponding parameters in `batch_optimization_kwargs` in `config/config.py`. Type in the appropriate parameters you found in the first step. They will be the parameters used for the bacth close packing process. For any serious senarios, please use `config/config.py` for an easier control on the parameters.
    - additional parameters for `batch_optimization` are expected to be modified as well: `number_system: int`: the number is expected to lie in the range [3, 1000]. Of course it would not cause any problem to type in 2 or any number more than 1000. If the number is too large, it might cause a big burden to your computer. (*For the moment this number cannot be 1... need to fix in the future. Also a better handle of large number of systems is also required to implement in future develoment.*)
    - if not sure about what the other keywords arguements mean, please keep them as default.

3. Run the script again, the results will be plotted and saved in the `temporary_output/` folder. The output files include: 
    - different stages of the processed images.
    - The figure of the optimized configuration
    - the JSON file of the sample holder, containing information of the optimized configuration, the ratio, and etc.
    - the evolution of the optimization process.
    - the CAD/STL file of the engraved sample holder
    - the CAD/STL file of the samples configuration

4. The above files can help the user to better visualize the process and results. Here are how the user can use the output:

- With the intermediate processed images, the user can check whether the parameters for the image processing is appropriate. IF NOT, modify the `image2contours_kwargs` in `config/config.py` and redo the process. In the trial run, the default parameters should be good enough. In case you lost the default value, here are them:
    ```Python
      image2contours_kwargs = dict(
         epsilon=2.5,
         lowercut=100,
         area_lowercut=1000,
         threshold=50,
         gaussian_window=np.array([5, 5]),
         is_gaussian_filter=False,
      )
    ```
- With the `area_evolution.jpg` the user can check whether the close-packing parameters are properly set. 

----------------------------------------------

**[Real-life Application] Usage:**

For any real-life application, the users are suggested to supervise the whole process from the beginning, i.e. the image processing. 

All the parameters can be found and adjusted in the `config/config.py` file. It's recommended to adjust it in the config file for a better overall global controll of the program.

Please follow the steps below:

1. Image processing: Set `STEP-CONTROL["contour_finding"]= True` and EVERYTHING ELSE to `False`. Please refer to the section [Image processing](#image-processing) for the detailed information of the image processing. Long story short: the processing involves: manual selection of stripes color vectors, and background color vectors for the program to filter out stripes and background. The users need to adjust the following parameters:
    - `stripes_vectors`
    - `background_vectors`
    - `target_background_vector`
    - `epsilon`
    - `lowercut`
    - `area_lowercut`
    - `threshold`
    - `gaussian_window`
    - `is_gaussian_filter`

Check the `temporary_output/` folder for the processed images. If the contours are not found as expected, please adjust the parameters and redo the process.

2. Test Optimization: Once you obtain a satisfactory image processing result, set `STEP_CONTROL["test"]=True` to start a test run. This is again to determine the best parameters for the optimization process. Everything else is the same as the trial run.

3. Batch Optimization: Once you find the best parameters for the optimization process, set `STEP_CONTROL["test"]=False` and `STEP_CONTROL["close_packing"]=True` and `STEP_CONTROL["convert_to_cad"]=True`. Everything else is the same as the trial run. The results will be saved in the `temporary_output/` folder.



# Detailed Dig-in 

## Folder Structure
The project folder is structured as follows:
```
workspace/
   ├── src/                     # Source code directory
   │   ├── main.py             # Main execution script
   │   └── scripts/            # Utility and test scripts
   │       ├── case_study_first.py
   │       ├── test.py
   │       └── useless.py
   ├── classes/
   │   ├── __init__.py           # Package-level initialization
   │   ├── class_contour.py      # Defines the Contour class
   │   ├── class_sample.py       # Defines the Sample class
   │   ├── class_sampleholder.py # Defines the SampleHolder class
   │   ├── helper_functions.py   # internal helper functions
   │   └── visualization.py      # internal visualization functions
   ├── close_packing/
   │   ├── __init__.py           # Package-level initialization
   │   ├── optimization.py       # close packing optimization
   │   ├── helper_functions.py   # internal helper functions
   │   └── visualization.py      # internal visualization functions
   ├── contour_finding/
   │   ├── __init__.py           # Package-level initialization
   │   └── image_processing.py   # image processing functions
   ├── config/
   │   ├── config.py             # configuration parameters and variables
   │   ├── gui_styles.css        # CSS for the GUI
   │   └── stylesheet.json       # plot setting parameters
   ├── gui/
   │   ├── __init__.py           # Package-level initialization
   │   ├── main_window.py        # main window widget
   │   ├── image_display.py      # image drop-in and display widget
   │   ├── matplotlib_canvas.py  # results display panel widget
   │   ├── helper_functions.py   # helper functions only for internal use
   │   └── worker.py             # worker thread for close packing
   ├── to_cad/
   │   ├── __init__.py           # Package-level initialization
   │   ├── to_cad.py             # functions for converting optimized configuration to CAD
   │   └── helper_functions.py   # internal helper functions
   ├── utils/
   │   ├── __init__.py           # Package-level initialization 
   │   ├── visualization.py      # general visualization tools/functions
   │   └── helper_functions.py   # general helper functions
   ├── setup.py                  # Package installation script
   ├── requirements.txt          # Dependencies list
   ├── .gitignore               # gitignore file
   └── README.md                # Project documentation
```
### Data structure
the contour, sample and sample holder are defined as classes in the `classes/` package.

**Contour**: The `Contour` class is defined in `class_contour.py`. It has the following attributes:

- `contour`: (cv2 contour) A numpy array representing the contour points, initially as `(N, 1, 2)` with `int32` datatype; changes to `float32` after relocation.
- `hull`: The convex hull of the contour, structured similarly to `contour` but typically containing fewer points.
- `polygon`: A `shapely.Polygon` object representing the convex hull.
- `vertices`: A simplified version of `hull`, as an `(N, 2)` numpy array with `float32` datatype.
- `area`: The area of the convex hull.
- `center`: The centroid of the hull, calculated as the mean of the vertices.
- `sample`: The associated `Sample` object.
- `id`: An integer identifying the parent sample.

The class includes these methods:

- `reorient`: (*not yet implemented*) Rotates the contour while keeping the center unchanged; all data is converted to `float32`.
- `relocate`: Translates the contour to a new position, converting data to `float32` for accuracy.

---------------------

**Sample**: The `Sample` class is defined in `class_sample.py`. It represents a sample with attributes for reorientation and relocation:

- `id`: An integer identifier for the sample.
- `name`: A string name for the sample.
- `grid_index`: Position in a grid on the sample holder (e.g., `(1, 2)`, not currently used).
- `contour_original`: The `Contour` object before reorientation or relocation.
- `contour_new`: Updated `Contour` object after reorientation or relocation.
- `position_original`: The center position of `contour_original`.
- `position_new`: The center position after reorientation or relocation.
- `phi_offset`: The angle offset in degrees (counter-clockwise) for reorientation.
- `is_reoriented`: Boolean indicating if the sample is reoriented.
- `is_relocated`: Boolean indicating if the sample is relocated.

The class has the following methods:

- `__str__`: Provides a string representation with sample ID and original position.
- `reorient`: (*Update required*) Adjusts orientation based on `phi_offset`.
- `relocate`: Translates the sample to the new position defined by `position_new`.

---------------------


**SampleHolder**: The `SampleHolder` class is defined in `class_sampleholder.py`. It represents a container for managing multiple samples and includes functionality to add, access, and manage samples.

- **Attributes**:
  - `name`: (str) Name of the sample holder.
  - `shape`: Shape of the sample holder (e.g., "circle" or "rectangle").
  - `size`: Dimensions of the sample holder in mm, initialized to `None`.
  - `radius`: Radius of the sample holder if circular, `None` initially.
  - `thickness`: Thickness of the sample holder.
  - `sample_thickness`: Thickness of the samples.
  - `center`: Center position of the sample holder as an ndarray.
  - `samples_area`: Total area of samples on the holder.
  - `convex_hull`: Convex hull contour of the sample holder.
  - `samples_list`: List of `Sample` objects contained in the holder.
  - `vertices_list`: List of vertices representing the sample holder.
  - `ratio`: Ratio of total sample area to sample holder area.
  - `_id2sample`: Dictionary mapping sample IDs to their corresponding `Sample` objects.
  - `_id2list_index`: Dictionary mapping sample IDs to their index in `samples_list`.

- **Methods**:
  - `__str__`: Returns a string representation with the name and sample count.
  - `print_samples`: Prints details of each sample on the holder.
  - `add_sample`: Adds a `Sample` object to the holder and updates mappings.
  - `update_convex_hull`: Updates the holder’s convex hull based on current sample positions.
  - `update_min_circle`: Computes the minimum enclosing circle for the convex hull.
  - `calculate_samples_area`: Calculates total area of all samples.
  - `calculate_ratio_of_samples`: Computes ratio of sample area to holder area.
  - `update_vertices_list`: Updates the list of holder vertices.
  - `update`: Refreshes holder parameters (convex hull, min circle, etc.) based on current samples.
  - `id2sample`: Returns a sample object by its ID.
  - `id2list_index`: Returns the index of a sample by its ID.
  - `list_index2id`: Returns the ID of a sample by its index.
  - `number_samples`: Returns the total number of samples on the holder.

---

**FunctionalSampleHolder**: The `FunctionalSampleHolder` class is a subclass of `SampleHolder` that adds methods for reorienting and relocating samples. (*The method currently under development. Operate with attention plz.*)

- **Methods**:
  - `assign_phi_offset`: Sets a rotation angle (`phi_offset`) for a sample.
  - `assign_phi_offset_by_index`: Sets `phi_offset` for a sample based on ID or list index.
  - `reorient_sample`: Reorients a sample according to its `phi_offset`.
  - `reorient_sample_by_index`: Reorients a sample by ID or list index.
  - `reorient`: Reorients all samples on the holder.
  - `relocate_sample`: Relocates a sample to a specified position.
  - `relocate_sample_by_index`: Relocates a sample by ID or list index to a specified position.

The `FunctionalSampleHolder` class enhances `SampleHolder` by enabling reorientation and repositioning of samples for specific spatial arrangements, like close packing.


### Future Improvements
- Incorporate better handling of sample re-orientation and close packing strategies.
- Add more robust error handling and validation for sample placement.
- Extend visualization with additional features such as 3D plotting or alignment indicators.

----------------------------
# Algorithm Details

### Image Processing
The image processing is implemented in the package `contour_finding`.

This process includes stripe removal, background unification, grayscale conversion, binarization, and contour finding. Each step is designed to optimize the quality of the detected contours, ensuring accurate results.


1. **Filter out stripes**: The `remove_stripes` function removes unwanted stripe patterns by analyzing the color vectors (`stripes_vectors`) of the stripes in the image. The color of each point in the image is represented by a BGR (Blue Green Red) vector, `np.array([100, 255, 1], dtype=np.uint8)` for example.

    - `stripes_vectors`: A list of BGR color vectors representing the stripe colors.
    - `target_background_vector`: A BGR vector representing the target color for the background after stripe removal.
    - `min_R` and `max_R`: Minimum and maximum distance for  defining the color range around the stripe vectors.

   All the colors that are close enough to the mean vector of the `stripes_vectors` will be set to `target_background_vector`. This will more or less remove the stripes from the image.


2. **Unify the background**: This function, `unify_background`, aims to smooth the image background by setting a uniform background color. Using sample vectors (`background_vectors`) from the background, it calculates the center of mass and defines a radius (`R`) within which pixels are replaced with the target background color (`target_background_vector`). Parameters:
    - `background_vectors`: A list of BGR color vectors representing sampled areas of the background.
    - `target_background_vector`: The color that will unify the background.
    - `min_R` and `max_R`: Minimum and maximum radius for defining the color range for the background vectors.

3. **Convert to grayscale**: After stripe removal and background unification, the image is converted to grayscale using `cv2.cvtColor`. Grayscale conversion simplifies further processing by reducing color information, leaving only intensity data, which is essential for effective contour detection.

4. **Binarize the image**: The grayscale image is then binarized using a threshold set by the `threshold` parameter, creating a high-contrast image where contour regions are highlighted. In this binary form, pixels above the threshold become white, and those below become black. Parameter:
    - `threshold`: The pixel intensity value used to binarize the image. All pixels with intensity above this threshold will be set to white, and those below will be set to black.

5. **Find contours**: The `cv2.findContours` function identifies all contours in the binary image. Detected contours are then simplified and filtered to ensure only significant shapes remain. Using the Ramer-Douglas-Peucker algorithm, contours are approximated with fewer points, which is controlled by `epsilon`. Additionally, the `contours2approximated_contours` function filters out contours below a certain perimeter (`lowercut`) or area (`area_lowercut`). Parameters (all in the unit of pixels or pixel^2):
    - `epsilon`: Approximation accuracy for contour simplification. Lower values retain more detail.
    - `lowercut`: Minimum contour perimeter required to keep the contour.
    - `area_lowercut`: Minimum contour area required to retain the contour.

6. **Get convex hulls**: Finally, the `contours2hulls` function converts the simplified contours into convex hulls. Convex hulls provide a boundary around each contour that minimizes points, ensuring the shapes are as simplified and closed as possible. 

In our close packing process, all the contours are approximted by their convex hulls.

**Parameters:**

- `image`: The input image to be processed.
- `is_preprocess`: If `True`, performs stripe removal and background unification.
- `stripes_vectors`: Color vectors of stripes to remove.
- `background_vectors`: Color vectors of background areas.
- `target_background_vector`: The target background color for unified background.
- `min_R` / `max_R`: Control color similarity tolerance for stripe and background adjustments.
- `threshold`: The intensity threshold for binarizing the image.
- `epsilon`: Accuracy of contour simplification; lower values retain more detail.
- `lowercut`: Minimum perimeter of contours to retain after simplification.
- `area_lowercut`: Minimum area of contours to retain after simplification.
- `gaussian_window`: Size of the Gaussian filter window to smooth the grayscale image.
- `is_gaussian_filter`: If `True`, applies Gaussian filtering to the grayscale image.
- `is_output_image`: If `True`, saves images at each processing step for review.

--------------------

### Close Packing

The `batch_optimization` and `optimization` functions optimize the configuration of polygons (representing samples) on a sample holder to achieve a close-packed arrangement, minimizing the area of the encclosing circle that contains all samples. This optimization is implemented using a **Simulated Annealing** algorithm, which iteratively explores configurations by adjusting positions based on temperature and step size parameters, controlling movement randomness and acceptance of less optimal configurations.


1. **Initial Setup**:
   - The initial temperature (e.g., 300) is high, exponentially reducing to about 20% of its starting value by the `gravity_off_at` iteration, helping the optimization converge smoothly.
   - The function uses `is_rearrange_vertices` to randomly rearrange initial sample positions, helping avoid poor local minima. Samples may also have a slight contour buffer applied (`contour_buffer_multiplier`), which inflates each sample’s contour to prevent overlap. 1.01 means the contour is expanded by 1%. 

2. **Temperature and Step Size Decay**:
   - **Temperature**: Controls how likely it is to accept worse configurations. Intuitively, this controls the fluctuations of the samples, i.e. they could move more randomly when the temperature is hight. This temperature decreases exponentially over the optimization iterations. 
   - **Step Size**: Determines the maximum movement a polygon can make in a single iteration, in pixel, linearly decreasing to 50% of its initial value. 

3. **Iteration Process**:
   - A random polygon/sample is selected in each iteration, and a **movement vector** is generated. The direction of movement may be influenced by gravitational forces between polygons (if `is_gravity=True`), guiding them towards a more compact configuration. The gravity effect is to accelerate the movement of the polygons towards each other at the very early stage. But this gravity also need to be turned off at some point because it could also hinder the movement of the polygons once they are already close enough to each other. set `gravity_off_at: int` to control at which iteration the gravity is turned off.
   - The proposed position is checked for overlap with other polygons. If overlap is detected, the movement is retried up to 3 times before discarding the attempt.
   - If the new position avoids overlap, the  enclosing-circle area of the new configuration is calculated. The change in area is evaluated against the acceptance criteria to determine if the move should be accepted.

4. **Acceptance Criteria**:
   - New configurations with a smaller area are always accepted.
   - Configurations with larger areas may still be accepted with a probability determined by `exp(-(new_area - area) / temperature)`, allowing the optimization to escape local minima at higher temperatures.

   **Note**: The temperature's magnitude should be roughly proportional to `0.1 * scale_hull * step_size` (where `scale_hull` represents the convex hull’s scale) to maintain effective control over movement randomness.

5. **Updating Sample Positions**:
   - Accepted moves update the configuration and new positions are set. The step size and temperature decay continue to refine the configuration towards an optimal state as the iterations progress.

6. **Batch Optimization**:
   - The `batch_optimization` function enables running multiple optimization instances in parallel, increasing the chances of finding a global optimum. Each batch may have varied initial conditions.
   - Results are visualized post-optimization, including the final configuration and area evolution over iterations, to evaluate the optimal configuration visually.

**Parameters**

- `number_of_iterations`: Defines the total number of iterations for convergence.
- `step_size`: Maximum pixel movement per iteration, controlling the rate of configuration changes.
- `temperature`: Initial temperature for simulated annealing, influencing the fluctuation of the sample movement.
- `contour_buffer_multiplier`: Multiplier to slightly expand each sample’s contour, preventing edge touching.
- `optimize_shape`: Shape to minimize, either `convex_hull` or `min_circle`, determining the target area type. 
- `is_rearrange_vertices`: If `True`, samples are randomly rearranged at the start of the process. 
- `is_gravity`: If `True`, gravitational forces influence movement direction, facilitating a compact arrangement.
- `gravity_multiplier` and `gravity_off_at`: Control gravity strength and when it is disabled, improving compactness early on without interfering in later iterations. multiplier = 1 means the movement of the polygons are always along the gravity force; 0.5 means the movement is half along the gravity force and half random. gravity_off_at = 2500 means the gravity is turned off at the 2500th iteration.
- `is_update_sampleholder`: If `True`, updates the sample holder object after the process.

This simulated annealing-based approach systematically explores configurations, balancing randomness
with structured optimization to achieve an efficient, close-packed layout of samples.


# To-do
- Slider for the contour finding parameters. 
- Rearrange the coordinates of the samples according to their position.
- Automize orientation feed-in for the re-orientation.
- auto-generation of report in the format of HTML.
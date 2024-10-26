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


# Usage
### Installation
Ensure that you are running Python 3.6 or later. There are two main ways to run the program: Python script or GUI. Currently, the python script is more stable and recommended. GUI is still under development but still can be used for contour finding and close packing.

0. Clone the repository to your local machine:
   ```bash
   git clone git@github.com:HongXunyang/laue_project.git
   cd laue_project/
   ```
1. Install the required packages, using the following command:
   ```bash
   pip install -r requirements.txt
   ```
2. Optional: create a folder `temporary_output/` in the root directory to store temporary output files generated during the process.
   ```bash
   mkdir temporary_output
   ```

------------------
### Usage Instructions: Python Script
It's recommended to run the python script within an IDE (e.g., VScode or PyCharm) to better understand the procedure and debug if necessary. The following steps outline how to run the script:

The script `main.py` in the root directory contains:
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

-------
### Usage Instructions: GUI
The GUI can be opened up by running the `main_gui.py` script. The GUI provides a user-friendly interface for uploading images, processing contours, and optimizing sample configurations. Currently, the GUI is under development and coversion to CAD file is not yet supported.

1. Run the GUI script using the following command:
   ```bash
   python main_gui.py
   ```
   or use an IDE to run the script.

2. The GUI window will open. Drag and drop an image of the sample holder with samples onto the designated area in the GUI. 

3. Click the "*Select Points*" button in the controls panel. Keep an eye on the output log panel to see the instructions and the progress. The user will be required to select three points on the image (directly click on the image) to define the color vectors for stripe detection; after this, the the user will need to select another three points to define the background color vectors. See [Image processing](#image-processing) for more details. 

4. On the top of the middle panel, the user can define the contour-finding parameters. When left empty, the parameters will be set to default values (indicated by the gray placeholder text). 

5. Once the parameters are set, click the "Process Image" button to start the contour-finding process. The program will display the processed image with the detected contours. If the results are not satisfactory, the user might need to quit the GUI and restart it again to reprocess the image with different parameters. (*I know I know this is not ideal but as I said the GUI is still under development...*)

6. After the contours are found, and before starting the close packing process, the user is suggested to adjust the parameters for the close packing in the middle panel. P.S. For now, I will suggest `NO. of system = 3` and `number_of_iterations = 3000` to start a trial run before any commitment. (*In the future, a "test close packing" need to be implemented for the user to find out the best optimization parameters...*)

7. Once the parameters are set, click the "Start Close Packing" button to begin the optimization process. On the output log panel, the progress of the optimization will be displayed. The final optimized configuration and the evolution of the optimization process will be shown on the bottom left panel. You can re-run the optimization process by re-clicking the "Start Close Packing" button again. You don't need to quit the GUI. 


# Detailed Dig-in 

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

This simulated annealing-based approach systematically explores configurations, balancing randomness with structured optimization to achieve an efficient, close-packed layout of samples.
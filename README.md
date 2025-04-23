# Image Inpainting Algorithms Implementation

This project implements the Laplace equation-based Partial Differential Equation (PDE) image inpainting algorithm and compares it with traditional filtering methods (mean filtering and Gaussian filtering). It demonstrates the differences in the repair effects of various methods and their performance metrics, aiming to explore and compare the effects and characteristics of different image inpainting techniques in practical applications and provide references for image inpainting tasks.

$$u_{i,j} = \frac{1}{4}(u_{i+1,j} + u_{i-1,j} + u_{i,j+1} + u_{i,j-1})$$

![comparison_all_methods](https://github.com/user-attachments/assets/b81e1831-8a47-41d3-94fe-1a18c5bfd50a)


## Directory Structure
```
.
├── img/                       # Input image folder  
│   └── 0.jpeg                 # Example input image  
├── results/                   # Output results folder  
│   ├── original.png           # Original image  
│   ├── damaged.png            # Damaged image  
│   ├── inpainted_laplace.png  # Image inpainted by Laplace PDE  
│   ├── mean_filter.png        # Image inpainted by Mean Filter  
│   ├── gaussian_filter.png    # Image inpainted by Gaussian Filter  
│   └── mask.png               # Drawn mask  
│   └── comparison_all_methods.png# Comparison chart of all methods  
└── inpaint.py                    # Main code file of the project  
```
## Function Overview

  * **Mask Drawing** : An interactive interface is provided through OpenCV, allowing users to manually draw a mask on the original image to simulate the damaged area for subsequent repair operations.
  * **Laplace PDE Image Inpainting** : The Laplace equation-based PDE method is used to implement the image inpainting function. The pixel values in the damaged area are iteratively calculated and filled to achieve image repair.
  * **Traditional Filtering Methods for Inpainting** : Mean filtering and Gaussian filtering are used to process the damaged image to achieve simple image repair effects for comparison with the Laplace PDE method.
  * **Repair Effect Evaluation** : PSNR (Peak Signal-to-Noise Ratio) and SSIM (Structural Similarity Index) are used as evaluation indicators to quantitatively evaluate the repair effects of different methods from the perspectives of brightness error and structural information retention. The evaluation results for each channel are output.
  * **Result Visualization and Saving** : The original image, damaged image, various repaired images, and drawn mask are saved as image files in the specified folder. A chart comparing the repair effects of all methods is also generated to facilitate intuitive viewing and comparison of the repair effects of different methods.

## Running Environment and Dependencies

  * **Python Version** : 3.x
  * **Dependencies** :
    * OpenCV (cv2) : Used for image processing operations such as image reading, display, mask drawing, and filtering.
    * NumPy (numpy) : Provides efficient array operations and numerical computing functions for image data storage and processing as well as numerical calculations in the Laplace equation solution process.
    * Matplotlib (matplotlib) : Used for visualizing and saving the results of images.
    * Scikit-image (skimage) : Provides tools related to image processing, such as image data type conversion and calculation of image quality evaluation indicators (PSNR and SSIM).

## Running Steps

  1. **Prepare Input Image** : Place the image to be repaired in the `img/` folder. The code currently uses `img/0.jpeg` as the default input image. You can replace it with other image files as needed.
  2. **Install Dependencies** : Install the project's dependencies using the following command in the terminal or command prompt (if you haven't installed them yet):

     * `pip install opencv-python numpy matplotlib scikit-image`

  3. **Run the Code** : Navigate to the project root directory in the terminal or command prompt and run the following command:

     * `python main.py`

  4. **Draw the Mask** : After running the code, a window named "Draw Mask (ESC to continue)" will pop up, displaying the input image. Hold down the left mouse button and drag it on the image to draw the mask, which will be marked in red. After completing the drawing, press the ESC key to exit the mask drawing interface and the program will continue to perform the subsequent image repair and evaluation operations.
  5. **View the Results** : After the program finishes running, you can find various output result image files in the `results/` folder. At the same time, a chart named `comparison_all_methods.png` will be generated in the same directory, showing a comparison of the repair effects of the original image, damaged image, and images repaired by different methods. In addition, the PSNR and SSIM evaluation results for each channel and overall will be output in the terminal or command prompt.

## Code Explanation

  * **Mask Drawing Section** : The interactive mask drawing is implemented through OpenCV's mouse callback function. Global variables are used to record the drawing status and coordinate information, and the image with the mask is updated in real-time during the drawing process.
  * **Laplace PDE Inpainting Implementation** : The `laplace_inpaint_channel` function is defined to process each channel of the image separately. In the repair process, the `np.roll` function is used to access and calculate the neighboring pixel values. Based on the principle of the Laplace equation, the pixel values in the damaged area are iteratively updated to gradually complete the image repair.
  * **Traditional Filtering Methods Implementation** : The `cv2.GaussianBlur` and `cv2.blur` functions provided by OpenCV are used to perform Gaussian filtering and mean filtering operations respectively to simply repair the damaged image.
  * **Evaluation Indicator Calculation** : The `peak_signal_noise_ratio` and `structural_similarity` functions from the Scikit-image library are used to calculate the PSNR and SSIM values of the damaged area between the original image and the repaired image, quantitatively evaluating the repair effects of different methods.
  * **Result Saving and Visualization** : The Matplotlib library's plotting functions are used to draw different stages of image results and the repaired images of various methods on the same chart. The `imsave` function is used to save the result images to the specified folder.

## Project Results and Comparative Analysis

  * **Repair Effect Comparison** : From the generated comparison chart and saved image results, it can be intuitively seen that the Laplace PDE repair method can better maintain the edge and structural information of the image when repairing the damaged image, and the visual effect is closer to that of the original image. On the other hand, traditional filtering methods (mean filtering and Gaussian filtering) may cause image details to become blurred during the repair process, especially when repairing larger damaged areas, and their effects may not be as ideal as the Laplace PDE method.
  * **Evaluation Indicator Comparison** : The PSNR and SSIM numerical results output show that the Laplace PDE method has relatively higher PSNR and SSIM values in each channel and overall, indicating that it performs better in brightness error control and structural information retention. This further verifies its effectiveness in image repair tasks.

## Project Expansion and Improvement Directions

  * **Integration of More Repair Algorithms** : Consider integrating other advanced image repair algorithms, such as deep learning-based image repair methods (e.g., image repair algorithms using convolutional neural networks), and conduct a more comprehensive comparison and analysis with existing methods to explore the applicability and advantages of different algorithms in different types of images and damaged situations.
  * **Optimization of Algorithm Performance** : The current Laplace PDE repair algorithm may have low computational efficiency when processing larger images or higher-resolution images. Optimization techniques such as multi-threading computation and GPU acceleration can be attempted to improve the algorithm's speed and efficiency, enabling it to better handle large-scale image repair tasks in practice.
  * **Improvement of Interactive Mask Drawing Function** : The interactive mask drawing interface can be further improved, for example, by adding undo and redo functions and providing various drawing tools (such as rectangles, polygons, etc.), to enhance the convenience and flexibility for users to draw masks and more accurately specify the damaged areas of the image.
  * **Adaptive Parameter Selection** : In the current Laplace PDE repair implementation, the number of iterations and other parameters are fixed. Research can be conducted on how to adaptively select suitable parameters based on the image's characteristics (such as the size of the damaged area and the complexity of the image content) to improve the stability and adaptability of the repair effect.

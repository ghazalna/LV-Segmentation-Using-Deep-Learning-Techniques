# **LV Segmentation Using Deep Learning Techniques**

## **Problem Statement**
Heart disease is the primary cause of death in the US for both men and women, taking 610,000 lives annually [1]. To evaluate the structural and functional properties of the heart non-invasively for the purpose of managing cardiovascular disease, doctors do Magnetic Resonance Imaging (MRI) scans. Heart disease can be detected by measuring the left ventricle's (LV) end-systolic and end-diastolic volumes (EDV) and ejection fraction (EF). These measurements can be obtained from the segmented contours of the left ventricle (LV); hence, reliable and consistent LV segmentation from MRI images is essential for accurate ESV, EDV, and EF measurements as well as non-invasive cardiac illness diagnosis.

Accurate segmentation of the left ventricle (LV) in cardiac MRI images is a significant challenge in medical imaging. The primary goal is to develop a model that can segment the LV with high precision, enabling better diagnosis and treatment of cardiac diseases.


<img width="871" alt="Screenshot 2024-09-13 at 20 11 09" src="https://github.com/user-attachments/assets/dfc35359-ee94-483c-9edf-9714b3b2f5c1">





## **Project Goals**
This project aims to enhance the accuracy and efficiency of Left Ventricle (LV) segmentation in cardiac MRI images using advanced deep learning models. LV segmentation is crucial for medical imaging, helping clinicians assess and diagnose cardiac conditions with greater precision. By leveraging models such as U-Net, Attention U-Net, and V-Net, we aim to provide a robust solution for accurate and fast LV segmentation.
- **Improve Segmentation Accuracy**: Develop a model that accurately detects LV boundaries.
- **Reduce Processing Time**: Utilize efficient methods that maintain accuracy while decreasing processing time.
- **Enhance Generalization**: Ensure the model performs well across various clinical settings and MRI data types.

## **Data**
The project utilizes publicly available datasets for LV segmentation, such as the ACDC dataset. Proper preprocessing, normalization, and augmentation techniques are applied to improve the model's performance.

## Methodology

### Image Preprocessing and Augmentation

To ensure consistent and effective training of the model, images and masks undergo several preprocessing and augmentation steps:

#### Preprocessing Steps
- **Loading and Normalization:**
  - Images and masks are loaded from specified paths.
  - Pixel values are normalized to the range [0, 1] for both images and masks.

- **Data Augmentation:**
  To enhance the model's robustness and generalization, various data augmentation techniques are applied:
  - **Flipping:** Random horizontal flipping of images with a 50% chance.
  - **Offsetting:** Random shifts applied in the x and y directions.
  - **Scaling:** Random scaling within a specified range.
  - **Rotation:** Random rotations within a defined angle range.
  - **Noise Addition:** Gaussian noise is added to images to simulate real-world variability.

- **Dataset Creation:**
  - **Splitting:** Images and masks are divided into training and validation sets.
  - **TensorFlow Datasets:** Created for efficient loading, preprocessing, and batching.

- **Visualization:**
  - **Subset Visualization:** A subset of images and masks is visualized to verify preprocessing and augmentation processes.

### Model Selection

Different deep learning models are evaluated for their effectiveness in segmentation:

- **U-Net:** A widely-used encoder-decoder architecture for biomedical image segmentation.
- **Attention U-Net:** An enhanced version of U-Net incorporating an attention mechanism to focus on critical regions.
- **V-Net:** A 3D convolutional network tailored for volumetric medical image segmentation.

### Model Training

- **Training:** Selected models are trained on the prepared data.
- **Hyperparameter Optimization:** Techniques are employed to optimize hyperparameters and enhance learning.

### Evaluation

- **Performance Metrics:** Model performance is assessed using metrics such as Dice Coefficient and Intersection over Union (IoU).

### Improvement and Fine-Tuning

- **Refinement:** Models are iteratively refined by adjusting hyperparameters, applying advanced techniques, and conducting error analysis to improve performance.

## Results

The performance of the model was evaluated using several metrics. Here are the results:

- **Loss:** 0.0341
- **Dice Coefficient:** 0.8909
- **IoU Score:** 0.8033
- **Precision:** 0.9022
- **Recall:** 0.8800
- **F1 Score:** 0.8909

These metrics indicate that the model performs well in terms of both segmentation accuracy and overall performance.




## **Installation**

To run this project locally, please ensure you have the following dependencies installed:

- Python 3.9+
- PyTorch
- NumPy
- OpenCV
- Matplotlib
- Scikit-Learn
- Jupyter Notebook

## **Refrences**
- CDC. “Know About the Facts Heart Disease.” Centers for Disease Control and Prevention, www.cdc.gov/heartdisease/facts.htm
 







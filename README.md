# **LV Segmentation Using Deep Learning Techniques**

## **Overview**
Heart disease is the primary cause of death in the US for both men and women, taking 610,000 lives annually [1]. To evaluate the structural and functional properties of the heart non-invasively for the purpose of managing cardiovascular disease, doctors do Magnetic Resonance Imaging (MRI) scans. Heart disease can be detected by measuring the left ventricle's (LV) end-systolic and end-diastolic volumes (EDV) and ejection fraction (EF). These measurements can be obtained from the segmented contours of the left ventricle (LV); hence, reliable and consistent LV segmentation from MRI images is essential for accurate ESV, EDV, and EF measurements as well as non-invasive cardiac illness diagnosis.


This project aims to enhance the accuracy and efficiency of Left Ventricle (LV) segmentation in cardiac MRI images using advanced deep learning models. LV segmentation is crucial for medical imaging, helping clinicians assess and diagnose cardiac conditions with greater precision. By leveraging models such as U-Net, Attention U-Net, and V-Net, we aim to provide a robust solution for accurate and fast LV segmentation.

<img width="871" alt="Screenshot 2024-09-13 at 20 11 09" src="https://github.com/user-attachments/assets/dfc35359-ee94-483c-9edf-9714b3b2f5c1">


## **Problem Statement**
Accurate segmentation of the left ventricle (LV) in cardiac MRI images is a significant challenge in medical imaging. The primary goal is to develop a model that can segment the LV with high precision, enabling better diagnosis and treatment of cardiac diseases.

## **Project Goals**
- **Improve Segmentation Accuracy**: Develop a model that accurately detects LV boundaries.
- **Reduce Processing Time**: Utilize efficient methods that maintain accuracy while decreasing processing time.
- **Enhance Generalization**: Ensure the model performs well across various clinical settings and MRI data types.

## **Data**
The project utilizes publicly available datasets for LV segmentation, such as the ACDC dataset. Proper preprocessing, normalization, and augmentation techniques are applied to improve the model's performance.

## **Methodology**
1. **Data Preparation**: Preprocessing, augmentation, and splitting the data into training, validation, and test sets.
   To ensure consistent and effective training of the model, images and masks undergo several preprocessing steps:
   **A.Loading and Normalization:**
   - Images and masks are loaded from specified paths.
   - Pixel values are normalized to the range [0, 1] for both images and masks.
   **B.Data Augmentation:**
     To improve the model's robustness and generalization, various data augmentation techniques are applied:
   - Flipping: Images are randomly flipped horizontally with a 50% chance.
   - Offsetting: Random shifts in the x and y directions are applied.
   - Scaling: Random scaling is applied within a specified range.
   - Rotation: Random rotations are performed within a defined angle range.
   - Noise Addition: Gaussian noise is added to images to simulate real-world variability.
   **C.Dataset Creation:**
   - Images and masks are split into training and validation sets.
   - TensorFlow Datasets are created for efficient loading, preprocessing, and batching.
   **D.Visualization:**
         A subset of images and masks is visualized to verify the preprocessing and augmentation processes These preprocessing and augmentation steps are designed to enhance model performance and ensure that the model generalizes well across various image conditions.


3. **Model Selection**: Evaluation of different deep learning models:
   - **U-Net**: A popular encoder-decoder architecture for biomedical image segmentation.
   - **Attention U-Net**: An improved version of U-Net with an attention mechanism for focusing on important regions in the image.
   - **V-Net**: A 3D convolutional network specifically designed for volumetric medical image segmentation.
4. **Model Training**: Training selected models on prepared data, optimizing hyperparameters, and using advanced techniques to enhance learning.
5. **Evaluation**: Assessing model performance using metrics like Dice Coefficient and Intersection over Union (IoU).
6. **Improvement and Fine-Tuning**: Iteratively refining models by adjusting hyperparameters, applying advanced techniques, and conducting error analysis.

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
 







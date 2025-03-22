# Breast Cancer Classification with Machine Learning
![BS cancer](https://github.com/arjunthilak05/Project-2-Breast-Cancer-Prediction/blob/main/output.png?raw=true)
## Project Overview
This project implements a machine learning pipeline to classify breast cancer tumors as malignant (M) or benign (B) using a dataset of digitized breast mass features. The pipeline includes data loading, exploratory data analysis, preprocessing, dimensionality reduction with PCA, and classification using Support Vector Machines.

## Dataset
The dataset used contains features computed from a digitized image of a fine needle aspirate (FNA) of a breast mass. The features describe characteristics of the cell nuclei present in the image.

Features include:
- Radius (mean of distances from center to points on the perimeter)
- Texture (standard deviation of gray-scale values)
- Perimeter
- Area
- Smoothness (local variation in radius lengths)
- Compactness (perimeter² / area - 1.0)
- Concavity (severity of concave portions of the contour)
- Concave points (number of concave portions of the contour)
- Symmetry
- Fractal dimension

Each feature is computed for the mean, standard error, and "worst" or largest (mean of the three largest values) of these features, resulting in 30 features.

## Project Structure
The project is organized into the following sections:

1. **Data Loading and Exploration**
   - Loading the dataset
   - Basic information display
   - Checking for null values and data types
   - Summary statistics

2. **Data Preprocessing**
   - Handling data types and missing values
   - Converting the target variable to categorical
   - Examining target distribution

3. **Exploratory Data Analysis (EDA)**
   - Calculating and visualizing correlation matrices
   - Plotting histograms of feature distributions
   - Data transformation (log transformation)
   - Handling infinite values

4. **Feature Selection**
   - Identifying highly correlated variables
   - Creating a subset of relevant features

5. **Principal Component Analysis (PCA)**
   - Dimensionality reduction
   - Scree plot visualization
   - Selecting optimal number of components

6. **Data Visualization with PCA**
   - Scatter plot of principal components
   - Color-coding by diagnosis (M/B)

7. **Model Training and Evaluation**
   - Train-test split (65%-35%)
   - SVM model with polynomial kernel
   - Performance metrics (accuracy, precision, recall, F1-score)

## Requirements
- Python 3.7+
- Libraries:
  - numpy
  - pandas
  - scikit-learn
  - matplotlib
  - tabulate

## Installation
```bash
pip install numpy pandas scikit-learn matplotlib tabulate
```

## Usage
1. Place the breast cancer dataset (`data.csv`) in the appropriate directory.
2. Update the file path in the code if necessary.
3. Run the script to execute the entire pipeline.

## Model Performance
The Support Vector Machine with polynomial kernel achieves:
- High accuracy in classifying tumors
- Good balance between precision and recall
- Effective discrimination between malignant and benign cases

## Visualizations
The project includes various visualizations:
- Bar and pie charts of diagnosis distribution
- Histograms of mean variables by diagnosis
- Correlation matrix heatmaps
- PCA scree plot
- PC1 vs PC2 scatter plot with diagnosis color-coding

## Future Work
- Implement additional classification algorithms for comparison
- Perform hyperparameter tuning for SVM
- Explore ensemble methods
- Develop a web interface for interactive classification

## Author
R ARJUN THILAK

## License
This project is licensed under the MIT License - see the LICENSE file for details.

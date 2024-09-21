# -*- coding: utf-8 -*-
# author: Willa Cheng

# this file includes functions caclulating similarities and distance 
# based on chapter 3.4 of book Data Mining by Han, Kamber and Pei

import numpy as np

# describe center of the data
def center(data, method=None):
    """
    Calculate the mean, median, or mode of a NumPy array based on data type.
    Works for multi-dimensional arrays, computing the statistic along the last axis.
    
    Parameters:
    data (numpy.ndarray): Input array (can be multi-dimensional).
    method (str, optional): The method to use ('mean', 'median', 'mode'). Defaults based on data type.
    
    Returns:
    numpy.ndarray: The calculated statistic based on the specified method, computed along the last axis.
    """
    
    if not isinstance(data, np.ndarray):
        raise ValueError("Input data must be a NumPy array.")
    
    # Check if the data is numerical
    if np.issubdtype(data.dtype, np.number):
        if method is None:
            method = 'mean'  # Default to mean for numerical data
    else:
        # Assume nominal data if not numerical
        if method is None:
            method = 'mode'  # Default to mode for nominal data
    
    # Calculate the statistic based on the specified method
    if method == 'mean':
        return np.mean(data, axis=-1)
    
    elif method == 'median':
        return np.median(data, axis=-1)
    
    elif method == 'mode':
        # Calculate mode using NumPy
        # Use np.unique to find unique values and their counts
        mode_values = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=-1, arr=data)
        return mode_values  # Return the mode values
    
    else:
        raise ValueError("Invalid method. Use 'mean', 'median', or 'mode'.")


# correlation coefficient for numeric data
def chi_square(data):
    """
    Calculate the Chi-Square statistic for two categorical variables.
    
    Parameters:
    data (numpy.ndarray): A 2D NumPy array with two categorical variables (contingency table).
    
    Returns:
    float: The Chi-Square statistic.
    """
    
    # Check if data is a 2D NumPy array
    if not isinstance(data, np.ndarray) or data.ndim != 2:
        raise ValueError("Input data must be a 2D NumPy array.")
    
    # Check if the array contains only two variables (two columns)
    if data.shape[1] != 2:
        raise ValueError("Input data must have exactly two variables (two columns).")
    
    # Check if the data is numerical
    if not np.issubdtype(data.dtype, np.number):
        raise ValueError("Input data must be numerical.")
    
    # Calculate the observed frequencies
    observed_freq = np.bincount(data[:, 0])  # Counts of the first variable
    total = np.sum(observed_freq)

    # Calculate the expected frequencies
    expected_freq = np.outer(observed_freq, np.bincount(data[:, 1])) / total

    # Calculate the Chi-Square statistic
    chi_square_statistic = np.sum((observed_freq - expected_freq) ** 2 / expected_freq)

    return chi_square_statistic

# covariance for numeric data
def covariance(data, y_index):
    """
    Calculate the covariance between multiple x variables and one y variable.
    
    Parameters:
    data (numpy.ndarray): A 2D NumPy array where each column is a variable.
    y_index (int): The index of the y variable (column).
    
    Returns:
    numpy.ndarray: Covariance values between each x variable and y.
    """
    
    # Check if data is a 2D NumPy array
    if not isinstance(data, np.ndarray) or data.ndim != 2:
        raise ValueError("Input data must be a 2D NumPy array.")
    
    # Check if the array contains numerical data
    if not np.issubdtype(data.dtype, np.number):
        raise ValueError("Input data must be numerical.")
    
    # Check if y_index is valid
    if y_index < 0 or y_index >= data.shape[1]:
        raise ValueError("Invalid index for y variable.")
    
    # Separate x variables and y variable
    x_variables = np.delete(data, y_index, axis=1)  # All columns except the y variable
    y_variable = data[:, y_index]                    # y variable
    
    # Calculate the means
    mean_x = np.mean(x_variables, axis=0)
    mean_y = np.mean(y_variable)
    
    # Calculate covariance
    covariances = np.array([(np.sum((x_variables[:, i] - mean_x[i]) * (y_variable - mean_y)) /
                                    (len(y_variable) - 1)) for i in range(x_variables.shape[1])])
    
    return covariances

# dwt

def dwt(data, levels=1, padding_value=0, method='haar'):
    """
    Perform a multi-level Discrete Wavelet Transform (DWT) using the specified wavelet method.
    Pads the input data with a specified value until its length along the last axis is a power of 2.
    
    Parameters:
    data (numpy.ndarray): Input data array (can be 1D, 2D, 3D, etc.).
    levels (int): Number of DWT levels to perform.
    padding_value (float): Value to use for padding.
    method (str): Wavelet method to use ('haar' or 'daubechies').
    
    Returns:
    (list of numpy.ndarray): List of tuples containing approximation coefficients and detail coefficients for each level.
    """
    
    def haar_transform(sub_array):
        """ Perform Haar transform on a 1D array. """
        approx = (sub_array[0::2] + sub_array[1::2]) / 2 # weighted sum
        detail = (sub_array[0::2] - sub_array[1::2]) / 2 # weighted difference
        return approx, detail
    
    def daubechies_transform(sub_array):
        """ Perform Daubechies transform on a 1D array. """
        # Daubechies 2 coefficients
        h0 = (1 + np.sqrt(3)) / 4
        h1 = (3 + np.sqrt(3)) / 4
        h2 = (3 - np.sqrt(3)) / 4
        h3 = (1 - np.sqrt(3)) / 4
        
        # Convolve and downsample
        approx = (h0 * sub_array[0::2] + h1 * sub_array[1::2] + h2 * sub_array[2::2] + h3 * sub_array[3::2])
        detail = (h0 * sub_array[0::2] - h1 * sub_array[1::2] + h2 * sub_array[2::2] - h3 * sub_array[3::2])
        return approx, detail
    
    results = []
    current_data = data
    
    for level in range(levels):
        shape = current_data.shape
        last_dim_length = shape[-1]
        
        # Calculate the next power of 2 for the last dimension
        next_power_of_2 = 2 ** np.ceil(np.log2(last_dim_length)).astype(int)
        
        # Pad the data along the last axis
        pad_width = [(0, 0)] * (data.ndim - 1) + [(0, next_power_of_2 - last_dim_length)]
        padded_data = np.pad(current_data, pad_width, mode='constant', constant_values=padding_value)
        
        # Initialize arrays for approximation and detail coefficients
        approx_shape = shape[:-1] + (next_power_of_2 // 2,)
        detail_shape = shape[:-1] + (next_power_of_2 // 2,)
        
        approx = np.zeros(approx_shape)
        detail = np.zeros(detail_shape)
        
        # Perform DWT along the last axis
        for idx in np.ndindex(padded_data.shape[:-1]):
            sub_array = padded_data[idx]
            if method == 'haar':
                approx[idx], detail[idx] = haar_transform(sub_array)
            elif method == 'daubechies':
                approx[idx], detail[idx] = daubechies_transform(sub_array)
            else:
                raise ValueError("Invalid method. Use 'haar' or 'daubechies'.")
        
        results.append((approx, detail))
        
        # Update current data for the next level (use approximation coefficients)
        current_data = approx
    
    return results

# pca
def pca(data, n_components):
    """
    Perform Principal Component Analysis (PCA) on the input data.
    
    Parameters:
    data (numpy.ndarray): A 2D NumPy array where rows are samples and columns are features.
    n_components (int): The number of principal components to return.
    
    Returns:
    numpy.ndarray: The transformed data in the new PCA space.
    numpy.ndarray: The principal components (eigenvectors).
    numpy.ndarray: The explained variance for each principal component.
    """
    
    # Step 1: Normalize the input data
    data_mean = np.mean(data, axis=0)
    data_normalized = data - data_mean
    data_std = np.std(data_normalized, axis=0)
    data_normalized /= data_std
    
    # Step 2: Compute the covariance matrix
    covariance_matrix = np.cov(data_normalized, rowvar=False)
    
    # Step 3: Compute the eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
    
    # Step 4: Sort the eigenvalues and eigenvectors in descending order
    sorted_indices = np.argsort(eigenvalues)[::-1]
    sorted_eigenvalues = eigenvalues[sorted_indices]
    sorted_eigenvectors = eigenvectors[:, sorted_indices]
    
    # Step 5: Select the top k eigenvectors (principal components)
    principal_components = sorted_eigenvectors[:, :n_components]
    
    # Step 6: Transform the data to the PCA space
    transformed_data = np.dot(data_normalized, principal_components)
    
    return transformed_data, principal_components, sorted_eigenvalues[:n_components]
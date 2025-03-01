import numpy as np
import scipy

def lr_schedule(epoch):
    return 1e-10 * 10**(epoch/10)

def gaussian_label_smoothing(x, num_bins=512, sigma=3):
    # Create a soft probability distribution centered at x using a Gaussian.
    bins = np.arange(num_bins)
    soft_label = scipy.stats.norm.pdf(bins, loc=x, scale=sigma)  # Gaussian centered at x
    soft_label /= soft_label.sum()  # Normalize to sum to 1 (valid probability distribution)
    
    return soft_label

# Function to convert (x, y) coordinate labels into one-hot probability maps
def convert_labels_to_heatmaps(x_coords, y_coords, img_size=512, sigma = 3):
    x_labels = np.array([gaussian_label_smoothing(x, img_size, sigma) for x in x_coords])
    y_labels = np.array([gaussian_label_smoothing(y, img_size, sigma) for y in y_coords])
    return x_labels, y_labels
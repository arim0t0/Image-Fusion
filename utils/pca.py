import numpy as np
from sklearn.decomposition import PCA
from skimage.exposure import match_histograms


def pca_fusion_images(multispectral_image, pan_image):
    """
    Fuses a multispectral image and a panchromatic image using PCA and histogram matching.

    Args:
        multispectral_image: A NumPy array of shape (height, width, channels).
        pan_image: A NumPy array of shape (height, width).

    Returns:
        A NumPy array of shape (height, width, channels) representing the fused image.
    """

    # 1. PCA transform on multispectral image
    pca = PCA(n_components=3)
    pca_components = pca.fit_transform(multispectral_image.reshape(-1, 3))

    # 2. Histogram matching of panchromatic image to first PC
    matched_pan_image = match_histograms(pan_image, pca_components[:, 0].reshape(multispectral_image.shape[:2]))

    # 3. Replace first PC with matched panchromatic image
    pca_components[:, 0] = matched_pan_image.flatten()

    # 4. Inverse PCA transform
    fused_image = pca.inverse_transform(pca_components).reshape(multispectral_image.shape)

    return fused_image
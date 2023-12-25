import numpy as np
from scipy import ndimage
from sklearn.decomposition import PCA

def pca_hpf_fusion(multispectral_img, panchromatic_img, sigma=3, gain=1):
    """
    Performs image fusion using PCA and High Pass Filter.

    Args:
        multispectral_img: The resampled multispectral image.
        panchromatic_img: The panchromatic image.
        sigma: The standard deviation of the Gaussian filter (default: 3).
        gain: The gain parameter for detail injection (default: 1).

    Returns:
        The pan-sharpened multispectral image.
    """

    # PCA transformation
    pca = PCA()
    pc_components = pca.fit_transform(multispectral_img.reshape(-1, 3))

    # Gaussian filtering
    smoothed_pan = ndimage.gaussian_filter(panchromatic_img, sigma=sigma)

    # High spatial detail extraction
    high_detail = panchromatic_img - smoothed_pan

    # Inject high spatial detail into the first principal component
    pc1_std = np.std(pc_components[:, 0])
    pan_std = np.std(panchromatic_img)
    gain = gain * (pc1_std / pan_std)  # Adjust gain based on standard deviations
    new_pc1 = pc_components[:, 0] + gain * high_detail.flatten()

    # Inverse PCA transformation
    pc_components[:, 0] = new_pc1
    fused_img = pca.inverse_transform(pc_components).reshape(multispectral_img.shape)

    return fused_img
import numpy as np
import cv2
import torch
import scipy.stats as stats
# ! pip install sewar
from sewar.full_ref import vifp

# RMSE metric for 2 images (normalized by resolution)
# Expected value: Lower RMSE value means better similarity (0 means perfect similarity)
def rmse(img1, img2):
    # Convert to torch tensor
    img1 = torch.from_numpy(img1)
    img2 = torch.from_numpy(img2)
    # Convert to float
    img1 = img1.float()
    img2 = img2.float()
    # RMSE
    rmse = torch.sqrt(torch.mean((img1 - img2)**2))
    # Normalized RMSE by dividing by resolution
    rmse = rmse/img1.shape[0]
    rmse.requires_grad = True
    return rmse

# Spectral angle mapper for 2 images
# Description: SAM compares the spectral angle between two vectors, regardless of their magnitude. 
# SAM is a measure of the similarity of two spectra.
# Expected value: Lower (equal to 0)
def sam(img1, img2):
    # Convert to torch tensor
    img1 = torch.from_numpy(img1)
    img2 = torch.from_numpy(img2)
    # Convert to float
    img1 = img1.float()
    img2 = img2.float()
    # SAM
    # Dot product of 2 images
    dot = torch.sum(img1*img2)
    # Magnitude of 2 images
    mag1 = torch.sqrt(torch.sum(img1**2))
    mag2 = torch.sqrt(torch.sum(img2**2))
    # SAM
    sam = torch.acos(dot/(mag1*mag2))
    sam.requires_grad = True
    return sam


# Erreur relative adimensionnelle synthese (ERGAS) metric for 2 images
# Description: ERGAS is a measure of the mean relative squared error between two images.
# High ERGAS value indicates distortion in the fused image, low ERGAS value indicates good quality of the fused image.
# Expected value: Lower 

def ergas(img_ref, img_fuse):
    # Convert to torch tensor
    img_ref = torch.from_numpy(img_ref)
    img_fuse = torch.from_numpy(img_fuse)
    # Convert to float
    img_ref = img_ref.float()
    img_fuse = img_fuse.float()

     # h and l are spatial resolution of img2 (fusion image) and img1 (original image) respectively
    h = img_fuse.shape[0]*img_fuse.shape[1]
    l = img_ref.shape[0]*img_ref.shape[1]

    # Create loop to calculate sum of (root mean square error of each band/mean of each band)**2
    # If number of bands is 1
    if len(img_ref.shape) == 2:
        # RMSE
        rmse = torch.sqrt(torch.mean((img_ref - img_fuse)**2))
        # Mean of each band
        mean = torch.mean(img_ref)
        # Sum
        sum = (rmse/mean)**2
        ergas = 100*(h/l)*torch.sqrt(sum)
    else: 
        sum = 0
        for i in range(img_ref.shape[2]):
            # RMSE
            rmse = torch.sqrt(torch.mean((img_ref[:,:,i] - img_fuse[:,:,i])**2))
            # Mean of each band
            mean = torch.mean(img_ref[:,:,i])
            # Sum
            sum = sum + (rmse/mean)**2
        ergas = 100*(h/l)*torch.sqrt(sum/img_ref.shape[2])
    ergas.requires_grad = True
    return ergas

# Mean Bias (MB) metric for 2 images
# Description: MB is a measure of the mean difference between two images.
# Mean value of an image refers to the gray level of pixels in the image.
# Expected value: Lower (equal to 0)

def mb(img_ref, img_fuse):
    # Convert to torch tensor
    img_ref = torch.from_numpy(img_ref)
    img_fuse = torch.from_numpy(img_fuse)
    # Convert to float
    img_ref = img_ref.float()
    img_fuse = img_fuse.float()
    # MB
    mb = (torch.mean(img_ref) - torch.mean(img_fuse))/(torch.mean(img_ref))
    mb.requires_grad = True
    return mb

# Percentage of fit error (PFE) metric for 2 images
# Description: PFE measures the normalized difference between the pixels of reference and fused to 
# the norm of reference image
# Expected value: Lower (equal to 0)

def pfe(img_ref, img_fuse):
    # Convert to torch tensor
    img_ref = torch.from_numpy(img_ref)
    img_fuse = torch.from_numpy(img_fuse)
    # Convert to float
    img_ref = img_ref.float()
    img_fuse = img_fuse.float()
    # PFE = norm of (reference image - fused image)/norm of reference image
    pfe = torch.norm(img_ref - img_fuse)/torch.norm(img_ref)
    pfe.requires_grad = True
    return pfe


# Signal to noise ratio (SNR) metric for 2 images
# Description: SNR is a measure of the ratio between the signal and the noise of an image.
# Expected value: Higher 
# Range of SNR: 0 - infinity
def snr(img_ref, img_fuse):
    # Convert to torch tensor
    img_ref = torch.from_numpy(img_ref)
    img_fuse = torch.from_numpy(img_fuse)
    # Convert to float
    img_ref = img_ref.float()
    img_fuse = img_fuse.float()
    # SNR
    snr = 10*torch.log10(torch.sum(img_ref**2)/torch.sum((img_ref - img_fuse)**2))
    snr.requires_grad = True
    return snr


# Peak signal to noise ratio (PSNR) metric for 2 images
# Description: PSNR is a measure of the ratio between the maximum possible power of an 
# image and the power of corrupting noise that affects the quality of its representation.
# Expected value: Higher
# Range of PSNR: 0 - infinity

def psnr(img_ref, img_fuse):
    # Convert to torch tensor
    img_ref = torch.from_numpy(img_ref)
    img_fuse = torch.from_numpy(img_fuse)
    # Convert to float
    img_ref = img_ref.float()
    img_fuse = img_fuse.float()
    # MSE
    mse = torch.mean((img_ref - img_fuse)**2)
    # PSNR
    psnr = 20*torch.log10((torch.max(img_ref)**2)/mse)
    psnr.requires_grad = True
    return psnr


# Correlation coefficient (CC) metric for 2 images
# Description: CC is a measure of the correlation between two images.
# Expected value: Higher (equal to 1)
# Range of CC: -1 -> 1

def cc(img_ref, img_fuse):
    # Convert to torch tensor
    img_ref = torch.from_numpy(img_ref)
    img_fuse = torch.from_numpy(img_fuse)
    # Convert to float
    img_ref = img_ref.float()
    img_fuse = img_fuse.float()
    # CC = 2*covariance(img_ref, img_fuse)/(variance(img_ref) + variance(img_fuse))
    # Covariance
    cov = torch.sum((img_ref - torch.mean(img_ref))*(img_fuse - torch.mean(img_fuse)))/(img_ref.shape[0]*img_ref.shape[1] - 1)
    # Variance
    var_ref = torch.var(img_ref)
    var_fuse = torch.var(img_fuse)
    # CC
    cc = 2*cov/(var_ref + var_fuse)
    cc.requires_grad = True
    return cc


# Mutual information (MI) metric for 2 images
# Description: MI is a measure of the mutual dependence between two images.
# Expected value: Higher
# Range of MI: 0 - infinity

def mi(img_ref, img_fuse):
    from skimage.metrics import normalized_mutual_information
    mi = normalized_mutual_information(img_ref, img_fuse)
    mi = np.array(mi)
    # Convert to torch tensor
    mi = torch.from_numpy(mi)
    # Convert to float
    mi = mi.float()
    mi.requires_grad = True
    return mi


# Universal image quality index (UIQI) metric for 2 images
# Description: UIQI is a measure of the similarity between two images.
# Model a any distortion as a combination of three different factors: 
# loss of correlation, luminance distortions, and contrast distortion.
# Expected value: Higher (equal to 1)
# Range of UIQI: -1 -> 1

def uiqi(img_ref, img_fuse):
    # Convert to torch tensor
    img_ref = torch.from_numpy(img_ref)
    img_fuse = torch.from_numpy(img_fuse)
    # Convert to float
    img_ref = img_ref.float()
    img_fuse = img_fuse.float()

    # Mean of img_ref and img_fuse
    mean_ref = torch.mean(img_ref)
    mean_fuse = torch.mean(img_fuse)
    # Variance of img_ref and img_fuse
    var_ref = torch.var(img_ref)
    var_fuse = torch.var(img_fuse)
    # Covariance of img_ref and img_fuse
    cov = torch.sum((img_ref - mean_ref)*(img_fuse - mean_fuse))/(img_ref.shape[0]*img_ref.shape[1] - 1)
    # UIQI
    uiqi = (4*cov*mean_ref*mean_fuse)/((var_ref + var_fuse)*(mean_ref**2 + mean_fuse**2))
    uiqi.requires_grad = True
    return uiqi


# SSIM metric for 2 images
# Description: SSIM is a measure of the similarity between two images.
# SSIM is designed to improve on traditional methods of image quality assessment and
# has been shown to better align with human perception of image quality.
# Expected value: Higher (equal to 1)
# Range of SSIM: -1 -> 1

def ssim(img_ref, img_fuse):
    # Convert to torch tensor
    img_ref = torch.from_numpy(img_ref)
    img_fuse = torch.from_numpy(img_fuse)

    # Convert to float
    img_ref = img_ref.float()
    img_fuse = img_fuse.float()

    # Mean of img_ref and img_fuse
    mean_ref = torch.mean(img_ref)
    mean_fuse = torch.mean(img_fuse)
    # Variance of img_ref and img_fuse
    var_ref = torch.var(img_ref)
    var_fuse = torch.var(img_fuse)
    # Covariance of img_ref and img_fuse
    cov = torch.sum((img_ref - mean_ref)*(img_fuse - mean_fuse))/(img_ref.shape[0]*img_ref.shape[1] - 1)
    # SSIM
    ssim = (2*mean_ref*mean_fuse + 0.01)*(2*cov + 0.03)/((mean_ref**2 + mean_fuse**2 + 0.01)*(var_ref + var_fuse + 0.03))
    ssim.requires_grad = True
    return ssim

# Visual information fidelity (VIF) metric for 2 images
# Description: VIF measures the fidelity or faithfulness of the fused image 
# to the originals in terms of visual information. It evaluates the preservation of 
# important visual details, textures, and structures from the input images in the fused image.
# Expected value: Higher (equal to 1)
# Range of VIF: 0 -> 1

def vif(img_ref, img_fuse):
    from sewar.full_ref import vifp
    vif = vifp(img_ref, img_fuse)
    vif = np.array(vif)
    vif = torch.from_numpy(vif)
    vif = vif.float()
    vif.requires_grad = True
    return vif

# All metrics with reference image
def all_metrics_ref(img_ref, img_fuse):
    if len(img_ref.shape) == 2:
        # Calculate and print all metrics
        rmse_val = rmse(img_ref, img_fuse).item()
        print ("RMSE: ", rmse_val)
        sam_val = sam(img_ref, img_fuse).item()
        print ("SAM: ", sam_val)
        ergas_val = ergas(img_ref, img_fuse).item()
        print ("ERGAS: ", ergas_val)
        mb_val = mb(img_ref, img_fuse).item()
        print ("MB: ", mb_val)
        pfe_val = pfe(img_ref, img_fuse).item()
        print ("PFE: ", pfe_val)
        snr_val = snr(img_ref, img_fuse).item()
        print ("SNR: ", snr_val)
        psnr_val = psnr(img_ref, img_fuse).item()
        print ("PSNR: ", psnr_val)
        cc_val = cc(img_ref, img_fuse).item()
        print ("CC: ", cc_val)
        mi_val = mi(img_ref, img_fuse).item()
        print ("MI: ", mi_val)
        ssim_val = ssim(img_ref, img_fuse).item()
        print ("SSIM: ", ssim_val)
        vif_val = vif(img_ref, img_fuse).item()
        print ("VIF: ", vif_val)

    else:
        num_bands = img_ref.shape[2]
        # Calculate all metrics
        rmse_sum = 0
        for i in range(num_bands):
            rmse_sum = rmse_sum + rmse(img_ref[:,:,i], img_fuse[:,:,i]).item()
        rmse_val = rmse_sum/num_bands
        print ("RMSE: ", rmse_val)
        sam_sum = 0
        for i in range(num_bands):
            sam_sum = sam_sum + sam(img_ref[:,:,i], img_fuse[:,:,i]).item()
        sam_val = sam_sum/num_bands
        print ("SAM: ", sam_val)
        ergas_val = ergas(img_ref, img_fuse).item()
        print ("ERGAS: ", ergas_val)
        mb_sum = 0
        for i in range(num_bands):
            mb_sum = mb_sum + mb(img_ref[:,:,i], img_fuse[:,:,i]).item()
        mb_val = mb_sum/num_bands
        print ("MB: ", mb_val)
        pfe_sum = 0
        for i in range(num_bands):
            pfe_sum = pfe_sum + pfe(img_ref[:,:,i], img_fuse[:,:,i]).item()
        pfe_val = pfe_sum/num_bands
        print ("PFE: ", pfe_val)
        snr_sum = 0
        for i in range(num_bands):
            snr_sum = snr_sum + snr(img_ref[:,:,i], img_fuse[:,:,i]).item()
        snr_val = snr_sum/num_bands
        print ("SNR: ", snr_val)
        psnr_sum = 0
        for i in range(num_bands):
            psnr_sum = psnr_sum + psnr(img_ref[:,:,i], img_fuse[:,:,i]).item()
        psnr_val = psnr_sum/num_bands
        print ("PSNR: ", psnr_val)
        cc_sum = 0
        for i in range(num_bands):
            cc_sum = cc_sum + cc(img_ref[:,:,i], img_fuse[:,:,i]).item()
        cc_val = cc_sum/num_bands
        print ("CC: ", cc_val)
        mi_sum = 0
        for i in range(num_bands):
            mi_sum = mi_sum + mi(img_ref[:,:,i], img_fuse[:,:,i]).item()
        mi_val = mi_sum/num_bands
        print ("MI: ", mi_val)
        ssim_sum = 0
        for i in range(num_bands):
            ssim_sum = ssim_sum + ssim(img_ref[:,:,i], img_fuse[:,:,i]).item()
        ssim_val = ssim_sum/num_bands
        print ("SSIM: ", ssim_val)
        vif_val = vif(img_ref, img_fuse).item()
        print ("VIF: ", vif_val)

    list_metric = [rmse_val, sam_val, ergas_val, mb_val, pfe_val, snr_val, psnr_val, cc_val, mi_val, ssim_val, vif_val]
    return list_metric


# Standard deviation metric for fused image
def std(img_fuse):
    # Convert to torch tensor
    img_fuse = torch.from_numpy(img_fuse)
    # Convert to float
    img_fuse = img_fuse.float()
    # Standard deviation
    std = torch.std(img_fuse)
    std.requires_grad = True
    return std

# Entropy metric for fused image
def entropy(img_fuse):
    from skimage.measure import shannon_entropy
    entropy = shannon_entropy(img_fuse)
    entropy = np.array(entropy)
    # Convert to torch tensor
    entropy = torch.from_numpy(entropy)
    # Convert to float
    entropy = entropy.float()
    return entropy

# Fusion mutual information (FMI) metric for 3 images
# Expected value: Higher 
def fmi(img_1, img_2, img_fuse):
    # Calculate FMI
    fmi = mi(img_1, img_fuse) + mi(img_2, img_fuse) 
    return fmi

# Fusion quality index (FQI) metric for 3 images
# Expected value: Higher (equal to 1)
def fqi(img_1, img_2, img_fuse):
    # Calculate UIQI of img_1 and img_fuse
    uiqi_1 = uiqi(img_1, img_fuse)
    # Calculate UIQI of img_2 and img_fuse
    uiqi_2 = uiqi(img_2, img_fuse)
    # Calculate FQI using UIQI of img_1 and img_fuse and img_2 and img_fuse but the range of FQI is 0 -> 1
    fqi = (uiqi_1 + uiqi_2) / 2
    return fqi


# All metrics without reference image (Input: 2 material images and 1 fused image)
def all_metrics_non_ref(img_ref, img_vis, img_fuse):
    # Convert all images to single channel in numpy array
    if len(img_ref.shape) == 3:
        img_ref = img_ref[:,:,0]*0.299 + img_ref[:,:,1]*0.587 + img_ref[:,:,2]*0.114
    if len(img_vis.shape) == 3:
        img_vis = img_vis[:,:,0]*0.299 + img_vis[:,:,1]*0.587 + img_vis[:,:,2]*0.114
    if len(img_fuse.shape) == 3:
        img_fuse = img_fuse[:,:,0]*0.299 + img_fuse[:,:,1]*0.587 + img_fuse[:,:,2]*0.114
    print("Metrics for image fusion without reference image:")
    # Calculate and print all metrics
    std_val = std(img_fuse).item()
    print ("STD: ", std_val)
    entropy_val = entropy(img_fuse).item()
    print ("Entropy: ", entropy_val)
    fmi_val = fmi(img_ref, img_vis, img_fuse).item()
    print ("FMI: ", fmi_val)
    fqi_val = fqi(img_ref, img_vis, img_fuse).item()
    print ("FQI: ", fqi_val)
    list_metric = [std_val, entropy_val, fmi_val, fqi_val]
    return list_metric

# All metrics with reference and non-reference image (Input: 2 material images, 1 fused image)
def full_metrics(img_hyper, img_rgb, img_fuse):
    print("Metrics for image fusion with Hyperspectral image as reference image:")
    list_metric_hyper = all_metrics_ref(img_hyper, img_fuse)
    print("Metrics for image fusion with RGB image as reference image:")
    list_metric_rgb = all_metrics_ref(img_rgb, img_fuse)
    list_metric_non_ref = all_metrics_non_ref(img_hyper, img_rgb, img_fuse)
    # Concatenate 3 lists
    list_metric = list_metric_hyper + list_metric_rgb + list_metric_non_ref
    return list_metric
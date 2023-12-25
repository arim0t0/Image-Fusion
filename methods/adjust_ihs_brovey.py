import numpy as np
import cv2
from PIL import Image
Image.MAX_IMAGE_PIXELS = 1000000000
import spectral as sp
import matplotlib.pyplot as plt

class AdjustIHSBrovey:
    def __init__(self):
        pass

    def get_main_img(self, rgb_path):
        print("RGB image path: ", rgb_path)
        self.rgb_img = cv2.imread(rgb_path)
        self.rgb_img = cv2.cvtColor(self.rgb_img, cv2.COLOR_BGR2RGB)
        self.rgb_img = np.array(self.rgb_img, dtype=np.uint8)
        return self.rgb_img
    
    def get_pan_img(self, rgb_img):
        self.pan_image = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2GRAY)
        self.pan_image = np.array(self.pan_image, dtype=np.uint8)
        return self.pan_image
    
    def get_ms_img(self, ms_path, red_band, green_band, blue_band, nir_band):
        print("MS image path: ", ms_path)
        hdr2cm = sp.envi.open(ms_path)        
        dr_red = hdr2cm.read_band(red_band)
        dr_green = hdr2cm.read_band(green_band)
        dr_blue = hdr2cm.read_band(blue_band)
        self.dr_nir = hdr2cm.read_band(nir_band)

        # dr_red = np.where(dr_red > 50, 50, dr_red)
        # dr_green = np.where(dr_green > 50, 50, dr_green)
        # dr_blue = np.where(dr_blue > 50, 50, dr_blue)
        # self.dr_nir = np.where(self.dr_nir > 50, 50, self.dr_nir)

        self.ms_img = np.dstack((dr_red, dr_green, dr_blue, self.dr_nir))
        self.ms_img = np.array(self.ms_img, dtype=np.uint16)

        # # take 3 channels from the image 
        # self.ms_rgb = self.ms_img[:, :, 0:3]
        # self.ms_hsi_part = rgb2hsi.rgb2hsi(self.ms_rgb)
        # self.ms_hue = self.ms_hsi_part[0]
        # self.ms_sat = self.ms_hsi_part[1]
        # self.ms_int = self.ms_hsi_part[2]

        # del self.ms_hsi_part

        # normalize the image to 0-255 range
        # self.ms_img = (255 * (1.0 / self.ms_img.max() * (self.ms_img - self.ms_img.min()))).astype(np.uint8)
        return self.ms_img
    
    def g_ihs_bt(self, rgb_img, ms_img, k_value):
        self.k_value = k_value
        self.ms_img = ms_img
        self.pan_image = self.get_pan_img(rgb_img)
        self.intensity_first = self.ms_img[:, :, 0] + self.ms_img[:, :, 1] + self.ms_img[:, :, 2] + self.ms_img[:, :, 3]/4

        coefficients = self.pan_image / (self.intensity_first + k_value * (self.pan_image - self.intensity_first))

        red_gihs_bt = coefficients * (self.ms_img[:, :, 0] + k_value * (self.pan_image - self.intensity_first))
        green_gihs_bt = coefficients * (self.ms_img[:, :, 1] + k_value * (self.pan_image - self.intensity_first))
        blue_gihs_bt = coefficients * (self.ms_img[:, :, 2] + k_value * (self.pan_image - self.intensity_first))

        self.gihs_bt = np.dstack((red_gihs_bt, green_gihs_bt, blue_gihs_bt))
        self.gihs_bt = np.array(self.gihs_bt, dtype=np.uint8)
        self.gihs_bt = np.where(np.isnan(self.gihs_bt), 0, self.gihs_bt)
        self.gihs_bt = np.where(np.isinf(self.gihs_bt), 255, self.gihs_bt)
        self.gihs_bt = np.where(self.gihs_bt < 0, 0, self.gihs_bt)
        self.gihs_bt = np.where(self.gihs_bt > 255, 255, self.gihs_bt)
        self.gihs_bt = np.uint8(self.gihs_bt)
        return self.gihs_bt  

    def sa_ihs_bt(self, rgb_img, ms_img, k_value):
        self.k_value = k_value
        self.ms_img = ms_img
        self.pan_image = self.get_pan_img(rgb_img)
        self.intensity_second = (self.ms_img[:, :, 0] + 0.75 * self.ms_img[:, :, 1] + 0.25 * self.ms_img[:, :, 2] + self.ms_img[:, :, 3])/3

        coefficients = self.pan_image / (self.intensity_second + k_value * (self.pan_image - self.intensity_second))

        red_gihs_bt = coefficients * (self.ms_img[:, :, 0] + k_value * (self.pan_image - self.intensity_second))
        green_gihs_bt = coefficients * (self.ms_img[:, :, 1] + k_value * (self.pan_image - self.intensity_second))
        blue_gihs_bt = coefficients * (self.ms_img[:, :, 2] + k_value * (self.pan_image - self.intensity_second))

        self.gihs_bt = np.dstack((red_gihs_bt, green_gihs_bt, blue_gihs_bt))
        self.gihs_bt = np.array(self.gihs_bt, dtype=np.uint8)
        self.gihs_bt = np.where(np.isnan(self.gihs_bt), 0, self.gihs_bt)
        self.gihs_bt = np.where(np.isinf(self.gihs_bt), 255, self.gihs_bt)
        self.gihs_bt = np.where(self.gihs_bt < 0, 0, self.gihs_bt)
        self.gihs_bt = np.where(self.gihs_bt > 255, 255, self.gihs_bt)
        self.gihs_bt = np.uint8(self.gihs_bt)
        return self.gihs_bt  


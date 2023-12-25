from utils.adjust_ihs_brovey import AdjustIHSBrovey
from utils.resize import Resize
import numpy as np

if __name__ == "__main__":
    rgb_path = "C:/Users/ACER/Pictures/Wheat/Season2/RGB_3Cm_Ortho_20220913.tif"
    ms_path = "C:/Users/ACER/Pictures/Wheat./Season2/hyper_20220913_3cm.hdr"
    red_band = 55
    green_band = 33
    blue_band = 12
    nir_band = 78
    coordinates = [6420, 11370, 5460, 8910]

    adjust_ihs_brovey = AdjustIHSBrovey()
    rgb_img = adjust_ihs_brovey.get_main_img(rgb_path)
    ms_img = adjust_ihs_brovey.get_ms_img(ms_path, red_band, green_band, blue_band, nir_band)

    # resize the image
    resize = Resize(2, coordinates)
    rgb_img = resize.resize_rgb(rgb_img)
    ms_img = resize.resize_ms(ms_img)

    # adjust the image
    gihs_bt = adjust_ihs_brovey.g_ihs_bt(rgb_img, ms_img, 0.5)
    rgb_img = np.uint8(rgb_img)
    ms_img = np.uint8(ms_img)
    ms_rgb = ms_img[:, :, 0:3]
    ghis_bt = np.uint8(gihs_bt)
    
    np.save("C:/Users/ACER/Pictures/Wheat/Output/Season2/demo/gihs_bt_2.npy", gihs_bt)
    np.save("C:/Users/ACER/Pictures/Wheat/Output/Season2/demo/input/ms_img.npy", ms_rgb)
    np.save("C:/Users/ACER/Pictures/Wheat/Output/Season2/demo/input/rgb_img.npy", rgb_img)

    sa_ihs_bt = adjust_ihs_brovey.sa_ihs_bt(rgb_img, ms_img, 0.2)
    sa_ihs_bt = np.uint8(sa_ihs_bt)
    np.save("C:/Users/ACER/Pictures/Wheat/Output/Season2/demo/sa_ihs_bt.npy", sa_ihs_bt)
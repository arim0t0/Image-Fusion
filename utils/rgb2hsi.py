import numpy as np

def rgb2hsi(rgbimg):
    red = rgbimg[:,:,0] / 255
    green = rgbimg[:,:,1] / 255
    blue = rgbimg[:,:,2] / 255

    RG = red - green + 0.001
    RB = red - blue + 0.001
    GB = green - blue + 0.001

    theta = np.arccos(np.clip(((0.5 * (RG + RB)) / (RG**2 + RB * GB)**0.5), -1, 1))
    theta = np.degrees(theta)

    h = np.where(blue <= green, theta, 360 - theta)

    h = ((h - h.min()) * (1 / (h.max() - h.min()) * 360))
    
    minRGB = np.minimum(np.minimum(red, green), blue)
    s = 1 - ((3 / (red + green + blue + 0.001)) * minRGB)
    i = (red + green + blue) / 3
    return h, s, i

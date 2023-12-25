import numpy as np

def hsi2rgb(hsiimg):
    h = hsiimg[:,:,0]
    h = ((h - h.min()) * (1/(h.max() - h.min()) * 360))
    s = hsiimg[:,:,1]
    i = hsiimg[:,:,2]

    print(np.max(i), np.min(i))

    cond1 = (h <= 120) & (h >= 0)
    cond2 = (h > 120) & (h <= 240)
    cond3 = (h > 240) & (h <= 360)

    b1 = (1 - s) / 3
    r1 = 1/3 * (1 + s * np.cos(np.radians(h)) / np.cos(np.radians(60 - h)))
    g1 = 1 - (r1 + b1)

    r2 = (1 - s) / 3
    g2 = 1/3 * (1 + s * np.cos(np.radians(h-120)) / np.cos(np.radians(60 - h + 120)))
    b2 = 1 - (r2 + g2)

    g3 = (1 - s) / 3
    b3 = 1/3 * (1 + s * np.cos(np.radians(h-240)) / np.cos(np.radians(60 - h + 240)))
    r3 = 1 - (g3 + b3)

    r = r1 * cond1 + r2 * cond2 + r3 * cond3
    g = g1 * cond1 + g2 * cond2 + g3 * cond3
    b = b1 * cond1 + b2 * cond2 + b3 * cond3

    rd = np.multiply(r, 3 * i)
    gr = np.multiply(g, 3 * i)
    bl = np.multiply(b, 3 * i)

    return rd, gr, bl
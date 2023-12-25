import os
import cv2
import numpy as np
from PIL import Image
import pywt
import matplotlib.pyplot as plt
import math
from scipy import ndimage
from skimage import exposure
from skimage.exposure import match_histograms

def neighborhood_count(image, kernel_size): #calculate the number of neigherbor in a slicing window 
    h,w = image.shape
    c = 1
    count_image = np.ones(image.shape)
    padded_count = np.pad(count_image, [[kernel_size//2,kernel_size//2],[kernel_size//2,kernel_size//2]])
    count_array = np.zeros(image.shape)
    for i in range(kernel_size):
        for j in range(kernel_size):
            if i == kernel_size-1 and j == kernel_size-1:
                count_array += padded_count[kernel_size-1:, kernel_size-1:]
            elif j == 4:
                count_array += padded_count[i:-(kernel_size-i)+1, kernel_size-1:]
            elif i == 4:
                count_array += padded_count[4:, j:-(kernel_size-j)+1]
            else:
                count_array += padded_count[i:-(kernel_size-i)+1, j:-(kernel_size-j)+1]

    # count_array = np.sum(count_array, axis=2)
    return count_array

def neighborhood_average(image, kernel_size): #calculate the average of neigherbor in a slicing window 
    h,w = image.shape
    c = 1
    sum_array = np.zeros(image.shape)
    padded_image = np.pad(image, [[kernel_size//2,kernel_size//2],[kernel_size//2,kernel_size//2]])
    for i in range(kernel_size):
        for j in range(kernel_size):
            if i == kernel_size-1 and j == kernel_size-1:
                sum_array += padded_image[kernel_size-1:, kernel_size-1:]
            elif j == 4:
                sum_array += padded_image[i:-(kernel_size-i)+1, kernel_size-1:]
            elif i == 4:
                sum_array += padded_image[kernel_size-1:, j:-(kernel_size-j)+1]
            else:
                sum_array += padded_image[i:-(kernel_size-i)+1, j:-(kernel_size-j)+1]

    average_array = sum_array/neighborhood_count(image, kernel_size)
    return average_array

def neighborhood_var(image, kernel_size): #calculate the variance of neigherbor in a slicing window
    h,w = image.shape
    c = 1
    count_array = neighborhood_count(image, kernel_size)
    average = neighborhood_average(image, kernel_size)
    # average = np.stack([average for i in range(c)], axis=-1)
    padded_x = np.pad(image, [[kernel_size//2,kernel_size//2],[kernel_size//2,kernel_size//2]])
    padded_bit = np.pad(np.ones(image.shape), [[kernel_size//2,kernel_size//2],[kernel_size//2,kernel_size//2]])
    std = np.zeros(image.shape)
    for i in range(kernel_size):
        for j in range(kernel_size):
            if i == kernel_size-1 and j == kernel_size-1:
                std += (padded_x[kernel_size-1:, kernel_size-1:] - average)**2 * padded_bit[kernel_size-1:, kernel_size-1:]
            elif j == 4:
                std += (padded_x[i:-(kernel_size-i)+1, kernel_size-1:] - average)**2 * padded_bit[i:-(kernel_size-i)+1, kernel_size-1:]
            elif i == 4:
                std += (padded_x[kernel_size-1:, j:-(kernel_size-j)+1] - average)**2 * padded_bit[kernel_size-1:, j:-(kernel_size-j)+1]
            else:
                std += (padded_x[i:-(kernel_size-i)+1, j:-(kernel_size-j)+1] - average)**2 * padded_bit[i:-(kernel_size-i)+1, j:-(kernel_size-j)+1]
    # std = np.sum(std, axis=2)
    std /= count_array-1
    return std

def guided_filter(X,Y,r): #guilded_filtering with 2 image Z = G(X,Y) with kernel_size 2*r + 1
  #ak =(1/|ν|*sum(YiXi)(i∈νk) − µk*X_averagek)/(δ**(2k) + η)
  #bk = X_averagek − ak*µk
  #Zi = a_averagei * Yi + b_averagei
  kernel_size = 2*r + 1
  regular_para = 0.001
  neigh_count = neighborhood_count(Y, kernel_size)
  Y_neigh_average = neighborhood_average(Y, kernel_size)
  X_neigh_average = neighborhood_average(X, kernel_size)
  Y_neigh_var = neighborhood_var(Y, kernel_size)
  ak = np.copy(Y)
  bk = np.copy(Y)
  sum_mul_XY = neighborhood_average(np.multiply(Y,X), kernel_size)
  ak = (sum_mul_XY - X_neigh_average*Y_neigh_average)/(Y_neigh_var + regular_para)
  bk = X_neigh_average - ak*Y_neigh_average
  ak_average = neighborhood_average(ak, kernel_size)
  bk_average = neighborhood_average(bk, kernel_size)
  Z = ak_average*Y + bk_average
  return Z

def Intensity(X):#Calulate Intensity of a multi-spectral image by averaging all channels
  Intensity = np.zeros((X.shape[0],X.shape[1])) 
  for i in range(X.shape[2]):
    Intensity += X[:,:,i]
  Intensity = Intensity / X.shape[2]
  return Intensity

# Prediction for linear regression, a simple dot product
def predict(X, w):
    y_pred = np.sum(X.T*w, axis=1)
    return y_pred

# Loss function: Mean Squared Error
# loss = 1/(2*n)*sum((y_pred-y_real)**2)
# y_pred and y_real are numpy array with the shape of [n,]
def loss_fn(y_pred, y_real, n):
    loss = (y_pred-y_real)**2
    loss = 1/(2*n)*np.sum(loss)
    return loss

# Gradient Descent
# w_new = w_old - lr*derivative(loss)
# derivative(loss) = 1/n*sum((y_pred-y_real)*X)
def gradient_descent(w, y_pred, y_real, X, lr, num_feats, n):
    derivative = 1/n*np.sum((X*(y_pred-y_real)), axis=1)
    # print(w, derivative)
    w_new = w-derivative*lr
    return w_new

# Adapative ihs with filter function (Required inputs: multispectral image and pan image)
def adapativeihswithfilter(img,panchromatic):
  I = Intensity(img) #calculate Intensity
  panchromatic = match_histograms(panchromatic, I, channel_axis = -1)
  Guided_1st = guided_filter(panchromatic, I, 2) #First guided filter
  D1 = panchromatic - Guided_1st #First D1
  Guided_2st = guided_filter(Guided_1st, I, 2) #Second guided filter
  D2 = Guided_1st - Guided_2st #Second D2
  Final_detail = D1 + D2 #Final detail
  lamba = 10**(-9)
  e = 10**(-10)
  #Calculate wp and wi
  sx_pan = ndimage.sobel(panchromatic,axis = 0)
  sy_pan = ndimage.sobel(panchromatic,axis = 1)
  sobel_pan = np.sqrt(sx_pan**2 + sy_pan**2)
  wp1 = np.exp(-lamba/((sobel_pan**4) + e))
  wm1 = []
  for i in range(3):
    sx = ndimage.sobel(img[:,:,i],axis = 0)
    sy = ndimage.sobel(img[:,:,i],axis = 1)
    sobel_m = np.sqrt(sx**2 + sy**2)
    wmi = np.exp(-lamba/((sobel_m**4) + e))
    wm1.append(np.nan_to_num(wmi, nan=0))
  wm1 = np.array(wm1)
  wm = (wm1).reshape(wm1.shape[0],-1)
  wp = wp1.reshape(-1)
  b = np.zeros(img.shape[2])
# Using gradient descent to calculate the optimized b in the function: min(β1,...,βn) ||wp − sum(βiwMi)(i∈0-n)||2
# Number or iteration t
# Learning rate lr
  t = 60
  lr = 0.00000001
  # Training to predict:
  for epoch in range(t):
      # Predict y_pred
      y_pred = predict(wm, b)
      # Calculate the loss
      loss = loss_fn(y_pred, wp, img.shape[2])
      # Update new weight through gradient descent
      b = gradient_descent(b, y_pred, wp, wm, lr, wp.shape[0], img.shape[2])
  g = np.copy(img) #calculate injection gain gi
  for i in range(img.shape[2]):
    g[:,:,i] = np.nan_to_num(np.divide(img[:,:,i].astype(np.float64), I, out=np.zeros_like(img[:,:,i].astype(np.float64)), where=(I != i)),nan = 0)*((1-b[i])*wp1 + b[i]*wm1[i])
  f = np.copy(img)
  # Fi = Mi + gi*Final_img
  for i in range(f.shape[2]):
    f[:,:,i] = (img[:,:,i] + (g[:,:,i]*Final_detail))
  return f


import os
import cv2
import numpy as np
from PIL import Image
import pywt
import matplotlib.pyplot as plt
import math

def spatial_frequency(image): #calculating spatial frequency of an image 
  Fr = 0
  Fc = 0
  M = image.shape[0]
  N = image.shape[1]
  for i in range(M):
    for j in range(1,N):
      Fr += (image[i,j] - image[i,j-1])**2
  Fr = math.sqrt((1/(M*N))*Fr)
  for j in range(N):
    for i in range(1,M):
      Fc += (image[i,j] - image[i-1,j])**2
  Fc = math.sqrt((1/(M*N))*Fc)
  SF = math.sqrt(Fr**2 + Fc**2)
  return SF
from skimage import exposure
from skimage.exposure import match_histograms

def fuseCoeff(cooefM,cooefP): #fusing method for the detailed components from dwt
  matchedcooeffP = match_histograms(cooefP, cooefM, channel_axis = -1)
  Fspc = spatial_frequency(matchedcooeffP)
  Fsmc = spatial_frequency(cooefM)
  F_smc = Fsmc/(Fsmc + Fspc)
  F_spc = Fspc/(Fsmc + Fspc)
  Fused_coeff = cooefM * F_smc + matchedcooeffP * F_spc
  return Fused_coeff

def SFDWT(MI, PI): #decomposing and fusing one channel from multi and pan
  fusedCooef = []
  coeff_MI = pywt.wavedec2(MI,'db2', level = 2)
  coeff_PI = pywt.wavedec2(PI,'db2', level = 2)
  for i in range(len(coeff_MI)):
    if(i == 0):
      fusedCooef.append(coeff_MI[0])
    else:
      c1 = fuseCoeff(coeff_MI[i][0], coeff_PI[i][0])
      c2 = fuseCoeff(coeff_MI[i][1], coeff_PI[i][1])
      c3 = fuseCoeff(coeff_MI[i][2], coeff_PI[i][2])
      fusedCooef.append((c1,c2,c3))
  fusedImage = pywt.waverec2(fusedCooef, 'db2')
  return fusedImage

def Spatialfreq_wavelet(ms_image, pan_image): #Fusing the multi-spectral image and pan (required input: one multi-spectral images and one pan image)
  if(ms_image.shape[0] % 2 == 1):   
    imgfused = np.zeros((ms_image.shape[0]+1, ms_image.shape[1],ms_image.shape[2]), dtype = np.uint8)
    for i in range(ms_image.shape[2]):
      imgfused[:,:,i] = SFDWT(np.vstack([ms_image[:,:,i],np.zeros(ms_image.shape[1])]), np.vstack([pan_image,np.zeros(ms_image.shape[1])]))
    return imgfused[:-1,:,:]
  else:
    imgfused = np.zeros((ms_image.shape[0], ms_image.shape[1],ms_image.shape[2]), dtype = np.uint8)
    for i in range(ms_image.shape[2]):
      imgfused[:,:,i] = SFDWT(ms_image[:,:,i], pan_image)
    return imgfused[:,:,:] 
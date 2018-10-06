import cv2
import os
import skimage
import numpy as np
from skimage.morphology import watershed
from skimage.feature import peak_local_max
from scipy import ndimage 
import scipy
import matplotlib.pyplot as plt 

def lana(image): 
	image = cv2.imread("digits.jpg",0)
	image=cv2.GaussianBlur(image,(5,5),0)
	# image=scipy.signal.wiener(image)
	# image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
	distance = ndimage.distance_transform_edt(image)
	#print distance
	local_maxi = peak_local_max(distance, indices=False, footprint=np.ones((3, 3)), labels=image)
	markers = skimage.morphology.label(local_maxi)
	labels_ws = skimage.morphology.watershed(-distance, markers, mask=image)
	#print labels_ws 
	# fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(8, 8), sharex=True, sharey=True, subplot_kw={'adjustable':'box-forced'})
	# ax = axes.ravel()

	# ax[0].imshow(image, cmap=plt.cm.gray, interpolation='nearest')
	# ax[0].set_title("Original")

	# ax[1].imshow(markers, cmap=plt.cm.gray, interpolation='nearest')
	# ax[1].set_title("Markers")
	# ax[2].imshow(image, cmap=plt.cm.gray, interpolation='nearest')
	fig = plt.gcf()
	# x=image[labels_ws]
	# cv2.imshow("img",x)
	plt.imshow(labels_ws, cmap=plt.cm.gray, interpolation='nearest')
	plt.show(block=False)
	fig.savefig("seg.jpg") 
	# plt.set_title("Segmented")
	# for a in ax:
	#     a.axis('off')

	# fig.tight_layout()
	# plt.show()
	# cv2.imshow("ws",labels_ws)
	# cv2.waitKey(0)
	# cv2.destroyAllWindows()


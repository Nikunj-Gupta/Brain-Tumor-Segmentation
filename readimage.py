import os
import numpy as np
import SimpleITK
import matplotlib.pyplot as plt
#from myshow import myshow, myshow3d
import cv2

# Directory where the DICOM files are being stored (in this
# case the 'MyHead' folder). 
def SimpleITK_show(img, title=None, margin=0.05, dpi=40 ):
    nda = SimpleITK.GetArrayFromImage(img)
    spacing = img.GetSpacing()
    figsize = (1 + margin) * nda.shape[0] / dpi, (1 + margin) * nda.shape[1] / dpi
    extent = (0, nda.shape[1]*spacing[1], nda.shape[0]*spacing[0], 0)
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_axes([margin, margin, 1 - 2*margin, 1 - 2*margin])

    plt.set_cmap("gray")
    ax.imshow(nda,extent=extent,interpolation=None)
    
    if title:
        plt.title(title)
    
    plt.show()

fileName = "SMIR.Brain.XX.O.MR_Flair.54512.mha"
idxSlice = 70
labelGrayMatter = 1

imgT1Original = SimpleITK.ReadImage(fileName)
nda = SimpleITK.GetArrayFromImage(SimpleITK.Tile(imgT1Original[:, :, idxSlice],(2, 1, 0)))
print nda.max()
#plot intesity plot 

values, bins = np.histogram(nda,
                            bins=np.arange(256))

plt.bar(bins[:-1], values, width = 1)
plt.xlim(min(bins), max(bins))
plt.show()   
# fig, axes = plt.subplots(ncols=2, nrows=3,
#                          figsize=(8, 4))
# ax0,ax1, ax2, ax3, ax4, ax5  = axes.flat
# ax1.plot(bins[:-1], values, lw=2, c='k')
# ax1.set_xlim(xmax=256)
# ax1.set_yticks([0, 400])
# ax1.set_aspect(.2)
# ax1.set_title('Histogram', fontsize=24)

# np.histogram([1, 2, 1], bins=[0, 1, 2, 3])

SimpleITK_show(SimpleITK.Tile(imgT1Original[:,:,idxSlice],(3,1)))
# # img_T1_255 = SimpleITK.Cast(SimpleITK.RescaleIntensity(imgT1Original), SimpleITK.sitkUInt8)
# # imgT1Smooth = SimpleITK.CurvatureFlow(image1=imgT1Original,
# #                                       timeStep=0.125,
# #                                       numberOfIterations=5)
# # #SimpleITK_show(SimpleITK.Tile(imgT1Smooth[:, :, idxSlice],(2, 1, 0)))
# # lstSeeds = [(80, 178, idxSlice),
# #             (98, 165, idxSlice),
# #             (120, 145, idxSlice),
# #             (145, 180, idxSlice)]
median_filter = SimpleITK.MedianImageFilter()
median_img=median_filter.Execute(imgT1Original)
SimpleITK_show(SimpleITK.Tile(median_img[:,:,idxSlice],(3,1)))

m = SimpleITK.GetArrayFromImage(SimpleITK.Tile(median_img[:, :, idxSlice],(2, 1, 0)))

cv2.imwrite("filtered image.jpg", m) 

# # imgSeeds = SimpleITK.Image(imgT1Smooth)
# # for s in lstSeeds:
# #     imgSeeds[s] = 10000

# # #SimpleITK_show(imgSeeds[:, :, idxSlice])

# #otsu thresholding
#seg = SimpleITK.BinaryThreshold(img_T1,
 #                          lowerThreshold=100, upperThreshold=400,
  #                         insideValue=1, outsideValue=0)
# myshow(sitk.LabelOverlay(imgT1Original, seg), "Basic Thresholding")

# otsu_filter = SimpleITK.OtsuThresholdImageFilter()
# otsu_filter.SetInsideValue(0)
# otsu_filter.SetOutsideValue(1)
# seg = otsu_filter.Execute(imgT1Original)
#SimpleITK_show(SimpleITK.Tile(seg[:,:,idxSlice],(3,1)))

#print("Computed Threshold: {}".format(otsu_filter.GetThreshold()))
# #set seed
# seed = (132, 142, 96)
# seg = SimpleITK.Image(imgT1Original.GetSize(), SimpleITK.sitkUInt8)
# seg.CopyInformation(imgT1Original)
# seg[seed] = 1
# seg = SimpleITK.BinaryDilate(seg, 3)
# myshow3d(SimpleITK.LabelOverlay(img_T1_255, seg),
# xslices=range(132, 133), yslices=range(142, 143i),
# zslices=range(96, 97), title="Initial Seed")

# # seg_con = SimpleITK.ConnectedThreshold(imgT1Original, seedList=[seed],
# # lower=100, upper=190)
# # myshow3d(SimpleITK.LabelOverlay(img_T1_255, seg_con),
# # xslices=range(132, 133), yslices=range(142, 143),
# # zslices=range(96, 97), title="Connected Threshold")
# fileName2="/home/likhitha/sem7/ML project/dataset/Training/HGG/bra/home/likhitha/sem7/ML project/dataset/Training/HGG/brats_2013_pat0001_1/SMIR.Brain.XX.O.MR_T2.54515.mha"
# img_T2 = SimpleITK.ReadImage(fileName2)
# img_T2_255 = SimpleITK.Cast(SimpleITK.RescaleIntensity(img_T2), sitk.sitkUInt8)
# img_multi = SimpleITK.Compose(img_T1, img_T2)
# seg_vec = SimpleITK.VectorConfidenceConnected(img_multi, seedList=[seed],
# numberOfIterations=1,
# multiplier=4,
# initialNeighborhoodRadius=1)
# myshow3d(sitk.LabelOverlay(img_T2_255, seg_vec),
# xslices=range(132, 133), yslices=range(142, 143),
# zslices=range(96, 97), title="VectorConfidenceConnected")

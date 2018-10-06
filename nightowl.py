import cv2
import numpy as np
import os
import glob
import mahotas as mt
from sklearn.svm import LinearSVC 
import SimpleITK
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler 
from sklearn.metrics import precision_recall_fscore_support as score 
from sklearn.metrics import accuracy_score 
#import em 
import watershedalgo 
import warnings 
from sklearn.ensemble import RandomForestClassifier 
#from myshow import myshow, myshow3d
warnings.filterwarnings("ignore") 
        

# load the training dataset
train_path = "../newdata/OT" 

train_names = os.listdir(train_path) 

print train_names 
# empty list to hold feature vectors and train labels
train_features = []

train_labels = []

def extract_features(image):
        # calculate haralick texture features for 4 types of adjacency
        
        textures = mt.features.haralick(image) 
        #print textures 
        
        # take the mean of it and return it
        ht_mean = textures.mean(axis=0) 
        
        #std = StandardScaler().fit(ht_mean) 
        #ht_mean = std.transform(ht_mean) 
        
        #sift = cv2.xfeatures2d.SIFT_create() 
        #kp = sift.detect(image,None)
        
        #features = cv2.FeatureDetector_create("SIFT") 
        #desc = cv2.DescriptorExtractor_create("SIFT") 
        
        #print ht_mean 
        
        return ht_mean 
        #return kp 

# loop over the training dataset
i = 1
idxSlice = 50
print "[STATUS] Started extracting haralick textures.."
for train_name in train_names:
        cur_path = train_path + "/" + train_name
        cur_label = train_name	
        i = 1
        for fileName in glob.glob(cur_path + "/*.mha"): 
                print "Processing Image - {} in {}".format(i, cur_label)
                # read the training image 
                imgT1Original = SimpleITK.ReadImage(fileName)
                image = SimpleITK.GetArrayFromImage(SimpleITK.Tile(imgT1Original[:, :, idxSlice],(2, 1, 0))) 
                
                #cv2.imwrite("OT"+str(i)+".jpg", image) 
                
                '''watershedalgo.lana(image) 
                
                seg = cv2.imread("seg.jpg") 
                ''' 
                
                '''image = np.uint8(image) 
	
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) 
                
                #cv2.imwrite("segmentation.jpg", image) 
                
                seg = em.lana(image) 
                '''
                #color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
                
                
                #print type(image) 
                #image = cv2.imread(file)
                
                
                # convert the image to grayscale
                #gray = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
                
                # extract haralick texture from the image 
                #seg = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                features = extract_features(image) 

                # append the feature vector and label
                train_features.append(features)
                train_labels.append(cur_label)

                # show loop update 
                i += 1 

print "Training features: {}".format(np.array(train_features).shape) 
print "Training labels: {}".format(np.array(train_labels).shape) 

# create the classifier
print "[STATUS] Creating the classifier.."
#clf_svm = LinearSVC(random_state = 9) 
clf = RandomForestClassifier(max_depth=2, random_state = 0)
# fit the training data and labels
print "[STATUS] Fitting data/label to model.."
#clf_svm.fit(train_features, train_labels) 
clf.fit(train_features, train_labels)
# loop over the test images
test_path = "../data/test" 
test_label = [] 
pred_labels = [] 
for fileName in glob.glob(test_path + "/*.mha"): 
        # read the input image
        #image = cv2.imread(file) 
        test_label.append(fileName[13:16]) 
        imgT1Original = SimpleITK.ReadImage(fileName)
        image = SimpleITK.GetArrayFromImage(SimpleITK.Tile(imgT1Original[:, :, idxSlice],(2, 1, 0))) 
        
        '''image = np.uint8(image) 
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) 
        
        
        seg = em.lana(image) 
        '''
        
        ''' 
        watershedalgo.lana(image) 
                
        seg = cv2.imread("seg.jpg") 
        '''      
        
        # convert to grayscale
        
        #gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # extract haralick texture from the image
        features = extract_features(image) 

        # evaluate the model and predict label
        prediction = clf.predict(features.reshape(1, -1))[0] 
        pred_labels.append(prediction) 
        #prediction = clf.predict(features.reshape(1, -1))[0]
        #print fileName + "",  
        #print prediction 
        
        # show the label
        #cv2.putText(image, prediction, (20,30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,255), 3)

        # display the output image
        #cv2.imshow("Test_Image", image)
        #cv2.waitKey(0)     



print "Actual: ", test_label 
print "Prediction: ", pred_labels

precision, recall, fscore, support = score(test_label, pred_labels) 

print 'Accuracy:', accuracy_score(test_label, pred_labels) 
print('precision: {}'.format(precision))
print('recall: {}'.format(recall))
print('fscore: {}'.format(fscore))
print('support: {}'.format(support))
        
         
 


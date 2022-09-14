import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
    
Thinkness_map1 = nib.load('thickness_map_subject_01.nii').get_fdata()
image1 = nib.load('raw_t1_subject_01.nii').get_fdata()
image2 = nib.load('raw_t1_subject_02.nii').get_fdata()


'''

'''
def normalize(t):
    
    ma = np.max(t)
    if ma>0:
        for i in range(len(t)):
            for j in range(len(t[i])):
                t[i,j] = t[i,j]*255/ma

    
def erosion(array):
  out = np.zeros((256,256))
  
  for i in range(1, array.shape[0] - 1):
    for j in range(2, array.shape[1] - 1):
      out[i,j] = array[i,j]
  
  for i in range(1, array.shape[0] - 1):
    for j in range(2, array.shape[1] - 1):
      if array[i, j] ==0:
        for m in [0, 1, -1]:
          for n in [0, 1, -1]:
            out[i + m, j + n] = 0
      else:
        continue

  return out     

def dilation(array):
  out = np.zeros((256,256))
  for i in range(1, array.shape[0] - 1):
    for j in range(2, array.shape[1] - 1):
      if array[i, j] >0:
        for m in [0, 1, -1]:
          for n in [0, 1, -1]:
            out[i + m, j + n] = array[i, j]
      else:
        continue

  return out    

Thinkness_map1 = Thinkness_map1[:,:,60]
test1 = image1[:,:,60]
test2 = image2[:,:,60]

plt.figure(2)
plt.imshow(Thinkness_map1, 'gray')
plt.title("Sample Thinkness map of image1")

plt.figure(1)
plt.imshow(test1, 'gray')
plt.title("image1")


plt.figure(3)
plt.imshow(test2, 'gray')
plt.title("image2")
normalize(test2)       
whitematter = np.zeros((256,256))      
for i in range(len(test2)):
    for j in range(len(test2[i])):
        if test2[i,j]<140:
            whitematter[i,j] = 0
        else:
            whitematter[i,j] = 1

newwhite = np.zeros((256,256))
for i in range(len(whitematter)):
    for j in range(len(whitematter[i])):
       newwhite[i,j] = whitematter[i,j]
e = np.array((256,256))

val = len(test2)/2+1
val1 = len(test2[1])/2+1

for i in range(len(whitematter)):
    for j in range(len(whitematter[i])):
        if not((i>val-128 and j<val1-128)):
            newwhite[i,j] = 1
        if not((i > 80 and  j > 90) and (i<166 and j<174)):
            whitematter[i,j] = 0
e = np.array((256,256))



plt.figure(4)
plt.imshow(whitematter, 'gist_gray',interpolation = None)
plt.title('whitematter of image2')
Cortial_whitematter = erosion(whitematter)
Cortial_whitematter = dilation(Cortial_whitematter)
Cortial_whitematter = dilation(Cortial_whitematter)
Cortial_whitematter = dilation(Cortial_whitematter)
Cortial_whitematter = dilation(Cortial_whitematter)
Cortial_whitematter = dilation(Cortial_whitematter)
Cortial_whitematter = dilation(Cortial_whitematter)
whitematter = erosion(whitematter)
Cortial = whitematter
for i in range(len(whitematter)):
    for j in range(len(whitematter[i])):
        Cortial[i,j] =Cortial_whitematter[i,j]-whitematter[i,j]

cortial_thick = np.zeros((256,256))



for i in range(len(whitematter)):
    for j in range(len(whitematter[i])):
        if Cortial[i,j]>0 and test2[i,j]<140 and test2[i,j]>80:
            cortial_thick[i,j] = 1

Thinkness_combined = np.zeros((256,256))    
ersd_cortial = erosion(cortial_thick)
for i in range(len(whitematter)):
    for j in range(len(whitematter[i])):
       Thinkness_combined[i,j] = ersd_cortial[i,j]+cortial_thick[i,j]





plt.figure(5)
plt.imshow(cortial_thick, 'gist_gray')
plt.title('Extracted Cortial of image2')


plt.figure(6)
plt.imshow(Thinkness_combined, 'gist_gray')
plt.title('Extracted Cortial of image2')

plt.figure(7)
x = range(256)
y = range(256)
X, Y = np.meshgrid(x, y)
plt.figure(7)
ax = plt.axes(projection='3d')

Thinkness_combined[i,j] = 20
ax.plot_surface(X, Y,Thinkness_combined, cmap='Oranges')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('thickness')
Thinkness_combined[i,j] = 0
plt.figure(8)
cmap = ListedColormap(["black", "orange", "yellow"])
plt.imshow(Thinkness_combined,cmap = cmap)
plt.imsave("Thickness_map_image2.png",Thinkness_combined)
plt.title('Extracted Cortial of image2(Orange and yellow)')



plt.figure(9)
plt.imshow(test2, 'gray')
for i  in range( 256):
    for j in range( 256):
        if Thinkness_combined[i,j]==1:
            plt.scatter(j,i,s =0.3, color = 'red')
        if Thinkness_combined[i,j] == 2:
            plt.scatter(j, i, s = 0.8,color = 'orange')
        

plt.title("Overlayed")
    
Thinkness_map1 = nib.load('thickness_map_subject_01.nii').get_fdata()
image1 = nib.load('raw_t1_subject_01.nii').get_fdata()
image2 = nib.load('raw_t1_subject_02.nii').get_fdata()
whitematter3d = np.zeros((256,256,256))
cortical_thic3d = np.zeros((256,256,256))


'''

'''
def normalize(t):
    
    ma = np.max(t)
    if ma>0:
        for i in range(len(t)):
            for j in range(len(t[i])):
                t[i,j] = t[i,j]*255/ma

    
def erosion(array):
  out = np.zeros((256,256))
  
  for i in range(1, array.shape[0] - 1):
    for j in range(2, array.shape[1] - 1):
      out[i,j] = array[i,j]
  
  for i in range(1, array.shape[0] - 1):
    for j in range(2, array.shape[1] - 1):
      if array[i, j] ==0:
        for m in [0, 1, -1]:
          for n in [0, 1, -1]:
            out[i + m, j + n] = 0
      else:
        continue

  return out     

def dilation(array):
  out = np.zeros((256,256))
  for i in range(1, array.shape[0] - 1):
    for j in range(2, array.shape[1] - 1):
      if array[i, j] >0:
        for m in [0, 1, -1]:
          for n in [0, 1, -1]:
            out[i + m, j + n] = array[i, j]
      else:
        continue

  return out    


for input1 in range(256):
    test2 = image2[:,:,(int)(input1)]
    normalize(test2)       
    whitematter = np.zeros((256,256))      
    for i in range(len(test2)):
        for j in range(len(test2[i])):
            if test2[i,j]<140:
                whitematter[i,j] = 0
            else:
                whitematter[i,j] = 1

    newwhite = np.zeros((256,256))
    for i in range(len(whitematter)):
        for j in range(len(whitematter[i])):
           newwhite[i,j] = whitematter[i,j]
    e = np.array((256,256))

    val = len(test2)/2+1
    val1 = len(test2[1])/2+1

    for i in range(len(whitematter)):
        for j in range(len(whitematter[i])):
            if not((i>val-128 and j<val1-128)):
                newwhite[i,j] = 1
            if not((i > 80 and  j > 90) and (i<166 and j<174)):
                whitematter[i,j] = 0
                
    
    #combine whitematter to 3D array
    
    
    whitematter3d[:,:,input1] = whitematter
    
    
    
    e = np.array((256,256))
    Cortial_whitematter = erosion(whitematter)
    Cortial_whitematter = dilation(Cortial_whitematter)
    Cortial_whitematter = dilation(Cortial_whitematter)
    Cortial_whitematter = dilation(Cortial_whitematter)
    Cortial_whitematter = dilation(Cortial_whitematter)
    Cortial_whitematter = dilation(Cortial_whitematter)
    Cortial_whitematter = dilation(Cortial_whitematter)
    whitematter = erosion(whitematter)
    Cortial = whitematter
    for i in range(len(whitematter)):
        for j in range(len(whitematter[i])):
            Cortial[i,j] =Cortial_whitematter[i,j]-whitematter[i,j]

    cortial_thick = np.zeros((256,256))



    for i in range(len(whitematter)):
        for j in range(len(whitematter[i])):
            if Cortial[i,j]>0 and test2[i,j]<140 and test2[i,j]>80:
                cortial_thick[i,j] = 1

    Thinkness_combined = np.zeros((256,256))    
    ersd_cortial = erosion(cortial_thick)
    for i in range(len(whitematter)):
        for j in range(len(whitematter[i])):
           Thinkness_combined[i,j] = ersd_cortial[i,j]+cortial_thick[i,j]


    Thinkness_combined[i,j] = 20

    Thinkness_combined[i,j] = 0
    
    
    #Combine Each layer of cortical to one 3D array
    cortical_thic3d[:,:,input1] = Thinkness_combined
    
#We can get nii file by converting 3d numpy array to nii file simply!!!
ni_img = nib.Nifti1Image(cortical_thic3d, affine=np.eye(4))
nib.save(ni_img, "thickness_map_subject_02.nii")
print("nii file exported successfully!")

    
    



    


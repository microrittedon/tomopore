#!/usr/bin/env python
# coding: utf-8

# In[1]:


from porespy import generators as ps
import matplotlib.pyplot as plt
import numpy as np
from skimage.transform import radon, iradon,iradon_sart
from skimage.feature import shape_index
from skimage import exposure
from skimage.feature import blob_dog 
from numpy import ma 
import os


# input parameters

# In[1]:


Pixel_size = 10 #microns
Eyeshot = 0 #microns  
N_mes = 0 #set of angels

Blobiness = 0 #6
Porosity = 0

#Gausian filter parameters
Max_Sigma = 0 #both for phantom and reconstructed one
Threshold=0 #0.4
Np=0

Rec_filter = ''
Rec_algorithm = ''

STD=0


# In[4]:


class reconstruction_parameters:
    def __init__(self, number_of_angles,threshold, rec_filter='shepp-logan',rec_algorithm='FBP', max_sigma=20):
        global N_mes, Blobiness, Porosity, Max_Sigma, Threshold, Np, Rec_filter, Rec_algorithm
        
        N_mes = number_of_angles
        
        
        Max_Sigma = max_sigma
        Threshold = threshold
        
        Rec_filter = rec_filter
        Rec_algorithm = rec_algorithm
        

    


# In[5]:


def create_phantom(blobiness,porosity,eyeshot=10000, pixel_size=10):
    global Blobiness,Porosity
    Blobiness = blobiness
    Porosity = porosity
    
    #Counting amount of pixels in side of square detector
    global Np, Eyeshot, Pixel_size
    Eyeshot = eyeshot
    Pixel_size = pixel_size
    
    Np=int(Eyeshot/Pixel_size)
    im = ps.blobs(shape=[Np, Np], porosity=Porosity, blobiness=Blobiness)
    return im


# In[5]:


def create_sinogram(im):
    theta2d = np.linspace(0,180,N_mes,endpoint = False)
    sin = radon(im, theta=theta2d, circle = False)
    return sin


# In[6]:


def reconstruct(sin):
    theta2d = np.linspace(0,180,N_mes,endpoint = False)
    if Rec_algorithm == 'SART':
        im_r = iradon_sart(sin,theta=theta2d) #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    else:
        im_r = iradon(sin,theta=theta2d,filter='shepp-logan') 
    return im_r


# In[7]:


def rescale(im_r):
    im_r_rescaled=exposure.equalize_hist(im_r)
    return im_r_rescaled


# In[8]:


original = "original"
reconstructed = "reconstructed"
blobs_dog_dict = {}
d_dict={
    original:[],
    reconstructed:[]
}


# In[9]:


def _select_blobs_original(im_original):
    blobs_dog_dict.update({
        original: blob_dog(im_original, max_sigma=Max_Sigma, threshold=0)
    })


# In[10]:


def _select_blobs_ReconstructedRescaled(im_r_rescaled):
    blobs_dog_dict.update({
        reconstructed: blob_dog(im_r_rescaled, max_sigma=Max_Sigma, threshold=Threshold)
    })


# In[11]:


def _select_blobs(im_original,im_r_rescaled):
    if len(blobs_dog_dict)<2:
        _select_blobs_ReconstructedRescaled(im_r_rescaled)
        _select_blobs_original(im_original)


# In[12]:


def _circles(ax,blobs_dog):
    for blob in blobs_dog:
        y, x, r = blob
        c=plt.Circle((x,y),r,color='y',linewidth=2,fill=False)
        ax.add_patch(c)


# In[13]:


def _count_lists_with_diametres(im_original, im_r_rescaled):
    _select_blobs(im_original,im_r_rescaled)
    for i in  [reconstructed, original]:
        for blob in blobs_dog_dict[i]:
            d_dict[i].append(2*blob[2]*Pixel_size*0.001)#mm
        d_dict[i]=sorted(d_dict[i])


# In[14]:


def _put_annotation():
    plt.figtext(1,0.5,'pixel size: %d microns\n'%Pixel_size+
                'amount of angels: %d\n'%N_mes
               )


# In[1]:


def distributions_of_diameters(im_original, im_r_rescaled,show=False, write_markdown = False):
    _count_lists_with_diametres(im_original, im_r_rescaled)
    plt.figure(figsize=(7,7))
    plt.subplot()
    y = np.array(d_dict[reconstructed])
    x = d_dict[reconstructed]
    plt.hist(y, x, label=reconstructed)

    y = np.array(d_dict[original])
    x = d_dict[original]
    plt.hist(y, x, histtype='step',linewidth=3, label=original)

    plt.legend()
    plt.title('Reconstruction, number of angles: %d'%N_mes)
    plt.xlabel('diametres, mm')
    plt.ylabel('quantaty')
    plt.xlabel('diametres, mm')
    plt.ylabel('count')
    plt.grid()
    
     
    _count_STD()
    
    if  write_markdown:
        current_path=os.getcwd()
        issue = "porosity"+str(Porosity)+"blobiness"+str(Blobiness)
        FolderPath = current_path+"\\"+issue+"\\report"
        if not os.path.isdir(FolderPath):
            os.makedirs(issue+"\\report")
        FigPath=current_path+"\\"+issue+"\\PixelSize{}NumberOfAngels{}".format(Pixel_size,N_mes)+".png"
        _put_annotation()
        plt.savefig(FigPath,quality=100)
        append_to_report("report", figPath=FigPath,folderPath=FolderPath)
        
    plt.figtext(1,0.3,"STD: {}".format(STD))
    ax = plt.gca()
    
    if not show:
        plt.close()
 
    return ax


# In[9]:


def _count_STD():
    global STD
    y1=y2=[]
    y1 = y_like_at_histogram(reconstructed)
    y2 = y_like_at_histogram(original)
    
    l1=len(y1)
    l2=len(y2)
    
    if not (l1==l2):
        for t in range(min(l1,l2),max(l1,l2)):
            if l1<l2:
                y1.append(0)
            else:
                y2.append(0)
    
    
    STDarray = np.array([y1,y2])
    STD = ma.std(STDarray)*np.sqrt(max(l1,l2)*2/len(y1)) #or /len(y2)/ Doesn't matter
    STD = round(STD,4)


# In[4]:


def y_like_at_histogram(type_of_image):
    count_array = []
    for x in _remove_duplicates(d_dict[type_of_image]):
        count = 0
        for ele in d_dict[type_of_image]: 
            if (ele == x): 
                count = count + 1
        count_array.append(count)
    return count_array 


# In[1]:


def _remove_duplicates(list_with_duplicates):
    res = [] 
    for i in list_with_duplicates: 
        if i not in res: 
            res.append(i)
    return res


# In[24]:


def append_to_report(reportName, figPath, folderPath):
    file = open(folderPath+"\\{}.md".format(reportName),"a")
    md_experiment_info = "##Pixel size:{}; Number of angles:{}; Detector size: {}\n".format(Pixel_size,N_mes,Eyeshot)
    md_porosity = "Blobiness: {}; Porosity: {}\n".format(Blobiness,Porosity)
    md_gausian = "\nMax sigma: {}; Threshold: {}\n".format(Max_Sigma,Threshold)
    md_reconstruction_parameters="\nReconstruction filte: {}; Reconstruction algorithm: {}\n".format(Rec_filter,Rec_algorithm)
    md_alt = "PixelSize:{} NumberOfAngles:{}".format(Pixel_size,N_mes)
    md_STD = "\n**STD**: {}\n".format(STD)
    md_image="\n![{}]({})\n".format(md_alt,figPath)
    file.write(md_experiment_info+
               md_porosity+
               md_gausian+
               md_reconstruction_parameters+
               md_STD+
               md_image+
              "<div style=""page-break-after: always;""></div> \n\n")
    file.write("********\n")
    file.close()
    


# In[3]:


class show:
    def show_phantom(im):
        plt.imshow(im)
        
    def show_sinogram(sin):
        plt.imshow(sin, cmap='gray', extent=(0,180,0,sin.shape[0]), aspect='auto')
   
    def show_reconstruction(im_r):
        plt.figure(figsize=(15,15))
        plt.subplot(111)
        plt.title('Reconstruction')
        plt.imshow(im_r,cmap = 'gray')
        
    def show_rescaled(im_r_rescaled):
        plt.figure(figsize=(7,7))
        plt.imshow(im_r_rescaled, cmap='gray')
        
    def show_circles_on_images(im_original,im_r_rescaled):
        _select_blobs_ReconstructedRescaled(im_r_rescaled)
        _select_blobs_original(im_original)

        fig, (ax1,ax2) = plt.subplots(ncols=2,nrows=1,figsize=(15,15))

        ax1.imshow(im_r_rescaled, cmap='gray')
        ax1.set_title(reconstructed)
        blobs_dog=blobs_dog_dict[reconstructed]
        blobs_dog[:, 2] = blobs_dog[:, 2]*np.math.sqrt(2)
        _circles(ax1,blobs_dog)

        ax2.imshow(im_original, cmap='gray')
        ax2.set_title(original)
        blobs_dog=blobs_dog_dict[original]
        blobs_dog[:, 2] = blobs_dog[:, 2]*np.math.sqrt(2)
        _circles(ax2,blobs_dog)
    
    def show_distributions_of_diameters(im_original, im_r_rescaled, WriteMarkdown = False):
        distributions_of_diameters(im_original, im_r_rescaled, write_markdown=WriteMarkdown,show = True)


# In[ ]:


count=0
X = []
Y = []
Z = []
Z_points = []


# In[12]:


class STDvisualisation:
    def __init__(self,x,y):
        global X, Y
        X=x
        Y=y      
            
    def add_point(z):
        global Z_points
        Z_points.append(z)
     
    def show_3d_plot(x_label,y_label,savefig = False):
        global X, Y, Z
        
        ##################
        for k in range(len(Y)):
            Z.append([]) 
        count=0
        i=0
        for y in Y:
            for x in X:
                Z[count].append(Z_points[i])
                i = i+1
            count = count+1
        ######################    
            
        Y.sort()
        X.sort()
        
        from mpl_toolkits.mplot3d import Axes3D
        import matplotlib.pyplot as plt
        from matplotlib import cm
        from matplotlib.ticker import LinearLocator, FormatStrFormatter
        import numpy as np

        fig = plt.figure(figsize=(20,20))
        ax = fig.gca(projection='3d')

        # Make data.
        X = np.asarray(X)
        Y = np.array(Y)

        X, Y = np.meshgrid(X, Y)
        Z = np.asarray(Z)
        surf = ax.plot_surface(X,Y,Z, cmap=cm.jet,
                               linewidth=0, antialiased=False)


        # Add a color bar which maps values to colors.
        fig.colorbar(surf, shrink=0.5, aspect=5)
        
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_zlabel('STD');
    
        if savefig:
            current_path=os.getcwd()
            issue = "porosity"+str(Porosity)+"blobiness"+str(Blobiness)
            FolderPath = current_path+"\\"+issue+"\\report"
            if not os.path.isdir(FolderPath):
                os.makedirs(issue+"\\report")
            FigPath=current_path+"\\"+issue+"\\STDplot"+".png"
            plt.savefig(FigPath,quality=100)
            
            file = open(FolderPath+"\\report.md","a") # make more flexible filename "report"
            md_alt = "STD plot"
            md_image="\n![{}]({})\n".format(md_alt,FigPath)
            file.write(
               md_image)
            file.close()
            
        plt.show()


# In[ ]:





#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import tomopore.pordiamdist as tp
import numpy as np


# In[ ]:


PS = [10,20,30] #X
NA = [50,100,150] #Y


# In[ ]:


im = tp.create_phantom(blobiness=6,porosity=0.1,pixel_size=PS[0])


# In[ ]:


tp.STDvisualisation(x=PS,y=NA)

for na in NA:
    sim = tp.create_sinogram(im,number_of_angles=na)
    for ps in PS:
        new_sim = tp.rescale_sim(sim, new_pixel_size=ps, origin_pixel_size=PS[0])
        tp.reconstruction_parameters(number_of_angles=na,threshold = 0.4)
        imr=tp.rescale(tp.reconstruct(new_sim))
        STD = tp.distributions_of_diameters(im,imr,write_markdown=False)
        tp.STDvisualisation.add_point(STD)


# In[ ]:


tp.STDvisualisation.show_3d_plot(x_label="Pixel Size",y_label="number of angles",savefig=True)


# In[ ]:





#!/usr/bin/env python
# coding: utf-8

# The whole process begins with raw ModelNet10 data(.OFF file)
# It only contains endpoints of the model(like those points at each corner).
# Inorder to get points that evenly spread across all surfacts of the model, we need PointCloudLibrary(PCL) to sample our model.
# but PCL only accept .PLY file so conversion is needed.

# First: convert .OFF file to .PLY file.

# In[1]:


import numpy as np
import pandas as pd
import h5py
import os
from sklearn import preprocessing
from functools import partial
from multiprocessing.dummy import Pool
from subprocess import call


categories = ['animal-bottle',
'animal-head',
'bowl',
'cone-vase',
'cuenco',
'flat-canteen',
'jar',
'lebrillo',
'olla',
'plate',
'statue',
'vase',
'vessel']

def create_folders(path):
	for i in range(len(categories)):
		command = " ".join(["mkdir",path+categories[i]])
		os.system(command)
		f_train_command = " ".join(["mkdir",path+categories[i]+"/train"])
		os.system(f_train_command)
		f_test_command = " ".join(["mkdir",path+categories[i]+"/test"])
		os.system(f_test_command)


path = './Peruvian-DB/'
path1 = './Peruvian-DB-OFF/'
#create_folders(path1)
path2 = './Peruvian-DB-PLY/'
#create_folders(path2)
path3 = './Peruvian-DB-Normalized/'
create_folders(path3)
#path2 = './ModelNet40PCD/'
#path3 = './Peruvian-DB-PLY/'
#create_folders(path3)


def PLYtoOBJ(path,categories,DataGroup):
	commands = []
	for cat  in categories:
		#deal with train first
		files = os.listdir(path + cat + '/'+DataGroup+'/')
		files = [x for x in files if x[-4:] == '.ply']
		for file_index,file in enumerate(files):
			fileName = file.split('.')[0]
			command = " ".join(["./meshconv",path + cat + '/'+DataGroup+'/' + file,'-c obj -tri  -o',path3 + cat +'/'+DataGroup+'/' + fileName])
			
			commands.append(command)
			print(command)			
			os.system(command)
			#print(command)
	"""
	pool = Pool(16) # two concurrent commands at a time
	for i, returncode in enumerate(pool.imap(partial(call, shell=True), commands)):
		if(returncode != 0):
			print("%d command failed: %d" % (i, returncode))   
	"""

PLYtoOBJ(path2,categories,'train')
PLYtoOBJ(path2,categories,'test')


def OBJtoOFF(path,categories,DataGroup):
	commands = []
	for cat  in categories:
		#deal with train first
		files = os.listdir(path + cat + '/'+DataGroup+'/')
		files = [x for x in files if x[-4:] == '.obj']
		for file_index,file in enumerate(files):
			fileName = file.split('.')[0]
			command = " ".join(["./meshconv",path + cat + '/'+DataGroup+'/' + file,'-c off -tri -o',path1 + cat +'/'+DataGroup+'/' + fileName])
			
			commands.append(command)
			#print(command)			
			#os.system(command)
			#print(command)
	pool = Pool(16) # two concurrent commands at a time
	for i, returncode in enumerate(pool.imap(partial(call, shell=True), commands)):
		if(returncode != 0):
			print("%d command failed: %d" % (i, returncode))   


#OBJtoOFF(path,categories,'train')
#OBJtoOFF(path,categories,'test')

def OFFtoPLY(path,categories,DataGroup):
	for cat  in categories:
		DataArray=[]
		#deal with train first
		files = os.listdir(path + cat + '/'+DataGroup+'/')
		files = [x for x in files if x[-4:] == '.off']
		for file_index,file in enumerate(files):
			fileName = file.split('.')[0]
			with open(path + cat + '/'+DataGroup+'/' + file, 'r') as f:

				tmp=f.readline().replace('\n','')
				line=''
				if tmp !='OFF':
					line = tmp[3:]
				else:
					line = f.readline().replace('\n','')
				print(fileName)
				#get number of points in the model
				point_count = line.split(' ')[0]
				face_count = line.split(' ')[1]

				data = []
				#fill ndarray with datapoints
				for index in range(0,int(point_count)):
					line = f.readline().rstrip().split()
					line[0] = float(line[0])
					line[1] = float(line[1])
					line[2] = float(line[2])
					data.append(line)
				data = np.array(data)
				#normalize data before conversion
				centroid = np.mean(data, axis=0)
				#data = data - centroid
				x_norm = np.copy(data - centroid)

				m = np.max(np.sqrt(np.sum(data**2, axis=1)))
				#data = data / m
				data = x_norm / m


				#create ply file,write in header first.
				with open(path2 + cat + '/'+DataGroup+'/' + fileName + ".ply",'w') as plyFile:
					plyFile.write('ply\nformat ascii 1.0\nelement vertex ')
					plyFile.write(point_count)
					plyFile.write('\nproperty float32 x\nproperty float32 y\nproperty float32 z\nelement face ')
					plyFile.write(face_count)
					plyFile.write('\nproperty list uint8 int32 vertex_indices\nend_header\n')
					for index in range(0,int(point_count)):
						plyFile.write(' '.join(map(str, data[index])))
						plyFile.write('\n')
					for index in range(0,int(face_count)):
						plyFile.write(f.readline())


# In[2]:


#OFFtoPLY(path1,categories,'train')
#OFFtoPLY(path1,categories,'test')


# Setp two: call tool "pcl_mesh_sampling_release.exe"(for pcl version higher than 1.9.1) to convert all .PLY data to .PCD

# In[3]:


import subprocess

def PLYtoPCD(path,categories,DataGroup):
	for cat  in categories:
		DataArray=[]
		#deal with train first
		files = os.listdir(path + cat + '/'+DataGroup+'/')
		files = [x for x in files if x[-4:] == '.ply']
		for file_index,file in enumerate(files):
			fileName = file.split('.')[0]
			command = " ".join(["/home/jeff/Downloads/Programs/pcl/pcl-pcl-1.11.0/build/bin/pcl_mesh_sampling",path + cat + '/'+DataGroup+'/' + file,path2 + cat +'/'+DataGroup+'/' + fileName + ".pcd",'-no_vis_result','-n_samples', '2200','-leaf_size', '0.01'])
			os.system(command)            


# In[4]:


#PLYtoPCD(path1,categories,'train')
#PLYtoPCD(path1,categories,'test')


# Step three: Merge converted PCD file to one .h5 file the shape of the data should be [n,2048,3]

# In[4]:


def PCDtoH5(path,categories,DataGroup):
    for cat  in categories:
        DataArray=[]    
        #deal with train first
        files = os.listdir(path + cat + '/'+DataGroup+'/')
        files = [x for x in files if x[-4:] == '.pcd']
        for file_index,file in enumerate(files):
            fileName = file.split('.')[0]
            with open(path + cat + '/'+DataGroup+'/' + file, 'r') as f:
                for y in range(9):
                    f.readline()
                #get number of points in the model
                line = f.readline().replace('\n','')
                point_count = line.split(' ')[1]
                #number of data less or more than 2048
                pad_count = 2048 - int(point_count)
                data = []
                f.readline()
                #fill ndarray with datapoints
                for index in range(0,int(point_count)):
                    line = f.readline().rstrip().split()
                    line[0] = float(line[0])
                    line[1] = float(line[1])
                    line[2] = float(line[2])
                    data.append(line)
                data = np.array(data)
                if pad_count > 0 :
                    idx = np.random.randint(point_count, size=pad_count)
                    data = np.append(data,data[idx],axis=0)
                elif  pad_count < 0 :
                    index_pool = np.arange(int(point_count))
                    np.random.shuffle(index_pool)
                    data = data[index_pool[:2048]]
                
                data = np.array([data])
            
                label = np.array(categories.index(cat)).reshape(1,1)
                if file_index == 0 and categories.index(cat) ==0:
                    with h5py.File(path + DataGroup +"_Relabel.h5", "w") as ff:
                        ff.create_dataset(name='data', data=data,maxshape=(None, 2048, 3), chunks=True)
                        ff.create_dataset(name='label', data=label,maxshape=(None, 1), chunks=True)
                else:
                    with h5py.File(path +DataGroup +"_Relabel.h5", "a") as hf:
                        hf['data'].resize((hf['data'].shape[0] + 1), axis=0)
                        hf['data'][-1:] = data
                        hf['label'].resize((hf['label'].shape[0] + 1), axis=0)
                        hf['label'][-1:] = label


# In[5]:


#PCDtoH5(path2,categories,'test')
#PCDtoH5(path2,categories,'train')


# Here is something to shuffle data shape.

# In[14]:


def ShuffleDataSet(path,DataGroup):
    with h5py.File(path +DataGroup+"_Relabel.h5", 'a') as hf:
        label = np.array(hf['label'])
        data = np.array(hf['data'])
        indices = np.arange(data.shape[0])
        np.random.shuffle(indices)

        label = label[indices]
        data = data[indices]
    
        with h5py.File(path + DataGroup +"Shuffled_Relabel.h5", "w") as ff:
            ff.create_dataset(name='data', data=data,shape=(data.shape[0], 2048, 3), chunks=True)
            ff.create_dataset(name='label', data=label,shape=(data.shape[0], 1), chunks=True)


# In[15]:


#ShuffleDataSet(path,'test')
#ShuffleDataSet(path,'train')


# In[ ]:





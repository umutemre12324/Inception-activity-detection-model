import numpy as np
import pandas as pd
import os 

def load_dataset(file_path,subset="train"):
   """
   file_path= UCI HAR dataset/train/Inertial Signals 
   subset: "train" or "test"
   """

   signal_names=[
        "body_acc_x_"+subset,
        "body_acc_y_"+subset,
        "body_acc_z_"+subset,
        "body_gyro_x_"+subset,
        "body_gyro_y_"+subset,
        "body_gyro_z_"+subset,
        "total_acc_x_"+subset,
        "total_acc_y_"+subset,
        "total_acc_z_"+subset 
   ]

   signal_data=[]
   for signal_name in signal_names:
      signal_file=os.path.join(file_path,signal_name+".txt")
      data = pd.read_csv(signal_file, header=None, sep=r"\s+")
      signal_data.append(data.values)
      
   stacked_data=np.stack(signal_data,axis=-1)  
   return stacked_data

x_train=load_dataset("UCI HAR dataset/train/Inertial Signals",subset="train")
y_train=np.loadtxt("UCI HAR dataset/train/y_train.txt").astype(int)-1

print("x_train shape:",x_train.shape)
print("y_train shape:",y_train.shape)

x_test=load_dataset("UCI HAR dataset/test/Inertial Signals",subset="test")
y_test=np.loadtxt("UCI HAR dataset/test/y_test.txt").astype(int)-1

print("x_test shape:",x_test.shape)
print("y_test shape:",y_test.shape)

np.save("x_train.npy",x_train)
np.save("y_train.npy",y_train)  
np.save("x_test.npy",x_test)
np.save("y_test.npy",y_test)    


      
import scipy.io as sio
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

## IMPORT MATLAB matrix as np array
data1='NPS_gp.mat'
gp_label=1 # 2nd label = 1
data2='NPS_ap.mat'
ap_label=3 # 4th label = 3
# data3='data_m.mat'
mat1 = sio.loadmat(data1) # insert your filename here
mat2 = sio.loadmat(data2)
# mat3 = sio.loadmat(data3)
mat1=np.array(mat1['data']) #'data' is name of variable when saved in matlab
mat2=np.array(mat2['data'])
# mat3=np.array(mat3['data'])
# mat=np.vstack((mat1,mat2,mat3))
mat=np.vstack((mat1,mat2))


## DETERMINE DIMENSIONS
num_runs = len(mat)
run1 = len(mat1) #number of category 1
run2 = len(mat2) #number of category 2
# run3 = len(mat2) #number of category 3
num_feat=len(mat[0][0][0][0])
time=[]
for run in range(num_runs):
    time.append(len(mat[run][0][0]))
max_time = max(time)
min_time = min(time)
print('num runs:',num_runs)
print('run1:',run1)
print('run2:',run2)
# print('run3:',run3)
print('num features:',num_feat)
print('time lengths:',time)
print('max time:',max_time)
print('min time:',min_time)


# CREATE PYTHON DATA ARRAY
# (MIN or PAD): pad did not work b/c introduces false data

# MIN TIME (truncate each run to "min time" length; prevents jagged array, all instances have same time length)
data=mat[0,0][:,:min_time] #first run
for run in range(1,num_runs): #stack subsuquent runs
    temp=mat[run,0][:,:min_time]
    data=np.vstack((data,temp))
print('data shape:',data.shape)


## CREATE LABELS
# TODO: make this "not hard coded"; # labels = # of matlab imports
# label = np.vstack((np.zeros((run1,1), dtype=int),np.ones((run2,1), dtype=int),2*np.ones((run3,1), dtype=int))) #np.ones default type is float64
label = np.vstack((gp_label*np.ones((run1,1), dtype=int),ap_label*np.ones((run2,1), dtype=int))) #np.ones default type is float64
# label = np.zeros((run1,1), dtype=int)
print('label shape', label.shape)
# print('label sample', label[0:5],'\n', label[run1:run1+5],'\n', label[run1+run2:run1+run2+5])
print('label sample', label,'\n')


## NORMALIZE DATA
# FIT to training data only, then transform both training and test data (p.70 HOML)
# Comment/Uncomment lines to switch between "Standard Scaler" and "MinMax Scaler"

# print('x train example:',x_train[0,0])
print('data example:',data[0,0])
scaler = StandardScaler()
# scaler = MinMaxScaler()
# x_train = scaler.fit_transform(x_train.reshape(-1, x_train.shape[-1])).reshape(x_train.shape)
data = scaler.fit_transform(data.reshape(-1, data.shape[-1])).reshape(data.shape)
# print('\nx train scaled example:',x_train[0,0])
print('\ndata scaled example:',data[0,0])

## Standard Scaler attributes
print('\nmean:',scaler.mean_)
print('\nvariance:',scaler.var_)

## MinMax Scaler attributes
# print('\nNum Features:',scaler.n_features_in_)
# print('\nData Min:',scaler.data_min_)
# print('\nData Max:',scaler.data_max_)

# x_test = scaler.transform(x_test.reshape(-1, x_test.shape[-1])).reshape(x_test.shape)
# print('\nx test sample scaled example:',x_test[0,0])


## SAVE DATASET
filename='data_7v10_r4s_nps.npz' # BvR: #blue v #red; gsm: greedy, smart, merge; r=# runs; s=scaled
np.savez(filename, x_test=data, y_test=label)
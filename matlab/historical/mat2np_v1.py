import scipy.io as sio
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

## IMPORT MATLAB matrix as np array
data1='data_g.mat'
data2='data_s.mat'
data3='data_m.mat'
mat1 = sio.loadmat(data1) # insert your filename here
mat2 = sio.loadmat(data2)
mat3 = sio.loadmat(data3)
mat1=np.array(mat1['data_g'])
mat2=np.array(mat2['data_s'])
mat3=np.array(mat3['data_m'])
mat=np.vstack((mat1,mat2,mat3))

## DETERMINE DIMENSIONS
num_runs = len(mat)
run1 = len(mat1) #number of category 1
run2 = len(mat2) #number of category 2
run3 = len(mat2) #number of category 3
num_feat=len(mat[0][0][0][0])
time=[]
for run in range(num_runs):
    time.append(len(mat[run][0][0]))
max_time = max(time)
min_time = min(time)
print('num runs:',num_runs)
print('run1:',run1)
print('run2:',run2)
print('run3:',run3)
print('num features:',num_feat)
print('time lengths:',time)
print('max time:',max_time)
print('min time:',min_time)

## CREATE PYTHON DATA ARRAY
# MIN TIME (truncate each run to "min time" length; prevents jagged array, all instances have same time length)
data=mat[0,0][:,:min_time] #first run
for run in range(1,num_runs): #stack subsuquent runs
    temp=mat[run,0][:,:min_time]
    data=np.vstack((data,temp))
print('data shape:',data.shape)

## CREATE LABELS
# TODO: make this "not hard coded"; # labels = # of matlab imports
label = np.vstack((np.zeros((run1,1), dtype=int),np.ones((run2,1), dtype=int),2*np.ones((run3,1), dtype=int))) #np.ones default type is float64
print('label shape', label.shape)
print('label sample', label[0:5],'\n', label[run1:run1+5],'\n', label[run1+run2:run1+run2+5])

## SPLIT DATA (TRAIN & TEST)
test_percentage=0.25
# data=data_s #comment out if want unscaled data
x_gtrain, x_gtest, y_gtrain, y_gtest = train_test_split(data[:run1], label[:run1], test_size=test_percentage, random_state=0) #split each category separately (equal representation during training and testing)
x_strain, x_stest, y_strain, y_stest = train_test_split(data[run1:run1+run2], label[run1:run1+run2], test_size=test_percentage, random_state=0)
x_mtrain, x_mtest, y_mtrain, y_mtest = train_test_split(data[run1+run2:], label[run1+run2:], test_size=test_percentage, random_state=0)

x_train = np.vstack((x_gtrain,x_strain,x_mtrain)) #recombine datasets
x_test = np.vstack((x_gtest,x_stest,x_mtest))
y_train = np.vstack((y_gtrain,y_strain,y_mtrain))
y_test = np.vstack((y_gtest,y_stest,y_mtest))

print('x train shape', x_train.shape)
print('y train shape', y_train.shape)
print('x test shape', x_test.shape)
print('y test shape', y_test.shape)

print('\nx train sample',x_train[0,0])
print('y train sample',y_train[0])

## SHUFFLE TRAIN DATA
x_train, y_train = shuffle(x_train, y_train, random_state=0)
print('\nSHUFFLE')
print('\nx train sample',x_train[0,0])
print('\ny train sample',y_train[0])

## NORMALIZE DATA
# FIT to training data only, then transform both training and test data (p.70 HOML)
# Comment/Uncomment lines to switch between "Standard Scaler" and "MinMax Scaler"

print('x train example:',x_train[0,0])
# scaler = StandardScaler()
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train.reshape(-1, x_train.shape[-1])).reshape(x_train.shape)
print('\nx train scaled example:',x_train[0,0])

## Standard Scaler attributes
# print('\nmean:',scaler.mean_)
# print('\nvariance:',scaler.var_)

## MinMax Scaler attributes
print('\nNum Features:',scaler.n_features_in_)
# print('\nData Min:',scaler.data_min_)
# print('\nData Max:',scaler.data_max_)

x_test = scaler.transform(x_test.reshape(-1, x_test.shape[-1])).reshape(x_test.shape)
print('\nx test sample scaled example:',x_test[0,0])

## SAVE DATASET
filename='data_7v10_r1200.npz' # BvR: #blue v #red; gsm: greedy, smart, merge; r=# runs; s=scaled
np.savez(filename, x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test)
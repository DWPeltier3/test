import scipy.io as sio
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

## IMPORT MATLAB matrix as np array
data1='data_g.mat'
data2='data_a.mat'
data3='data_ap.mat'
mat1=sio.loadmat(data1) # insert your filename here
mat2=sio.loadmat(data2)
mat3=sio.loadmat(data3)
mat1=np.array(mat1['data'])
mat2=np.array(mat2['data'])
mat3=np.array(mat3['data'])
mat=np.vstack((mat1,mat2,mat3))

## DETERMINE DIMENSIONS
num_runs = len(mat)
run1 = len(mat1) #number of class 1
run2 = len(mat2) #number of class 2
run3 = len(mat3) #number of class 3
#class starting index
c1si=0
c2si=run1
c3si=run1+run2
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
# print('time lengths:',time) #gets too long with large number of runs
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
label = np.vstack((np.zeros((run1,1),dtype=int), np.ones((run2,1),dtype=int), 2*np.ones((run3,1),dtype=int))) #np.ones default type is float64
print('label shape', label.shape)
print('label sample', label[c1si:c1si+5],'\n', label[c2si:c2si+5],'\n', label[c3si:c3si+5])

## SPLIT DATA (TRAIN & TEST)
test_percentage=0.25
x_c1train, x_c1test, y_c1train, y_c1test = train_test_split(data[:c2si], label[:c2si], test_size=test_percentage, random_state=0) #split each category separately (equal representation during training and testing)
x_c2train, x_c2test, y_c2train, y_c2test = train_test_split(data[c2si:c3si], label[c2si:c3si], test_size=test_percentage, random_state=0)
x_c3train, x_c3test, y_c3train, y_c3test = train_test_split(data[c3si:], label[c3si:], test_size=test_percentage, random_state=0)


x_train = np.vstack((x_c1train,x_c2train,x_c3train)) #recombine datasets
x_test = np.vstack((x_c1test,x_c2test,x_c3test))
y_train = np.vstack((y_c1train,y_c2train,y_c3train))
y_test = np.vstack((y_c1test,y_c2test,y_c3test))

print('x train shape', x_train.shape)
print('y train shape', y_train.shape)
print('x test shape', x_test.shape)
print('y test shape', y_test.shape)

print('\nx TRAIN sample (first instance, firt time step):\n',x_train[0,0])
print('y train sample:',y_train[0])

## SHUFFLE TRAIN DATA
x_train, y_train = shuffle(x_train, y_train, random_state=0)
print('\nSHUFFLE')
print('\nx train sample:\n',x_train[0,0])
print('\ny train sample:',y_train[0])

## NORMALIZE DATA
# FIT to training data only, then transform both training and test data (p.70 HOML)
# Comment/Uncomment lines to switch between "Standard Scaler" and "MinMax Scaler"

# print('\nx train example:',x_train[0,0]) #repeat from above
# scaler = StandardScaler()
scaler = MinMaxScaler(feature_range=(-1,1))
x_train = scaler.fit_transform(x_train.reshape(-1, x_train.shape[-1])).reshape(x_train.shape)
print('\nx TRAIN SCALED example (first instance, firt time step):\n',x_train[0,0])

## Standard Scaler attributes
# print('\nmean:',scaler.mean_)
# print('\nvariance:',scaler.var_)

## MinMax Scaler attributes
print('\nNum Features:',scaler.n_features_in_)
# print('\nData Min:',scaler.data_min_)
# print('\nData Max:',scaler.data_max_)

x_test = scaler.transform(x_test.reshape(-1, x_test.shape[-1])).reshape(x_test.shape)
print('\nx TEST SCALED example (first instance, firt time step):\n',x_test[0,0])

## SAVE DATASET
filename='data_10v10_r3600m11.npz' # BvR: #blue v #red; gsm: greedy, smart, merge; r=# runs; s=std scaled, m=minmax scaled
np.savez(filename, x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test)
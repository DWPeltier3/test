import numpy as np
from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

from timeit import default_timer as timer
from utils.elapse import elapse_time
from utils.resources import print_resources
import utils.params as params


## INTRO
start = timer() # start timer to calculate run time
GPUs=print_resources() # computation resources available
hparams = params.get_hparams() # parse BASH run-time hyperparameters (used throughout script below)
params.save_hparams(hparams) #create model folder and save hyperparameters list .txt


## IMPORT DATA
x_train, y_train, x_test, y_test, num_classes, cs_idx, input_shape, output_shape = import_data(hparams)

num_runs=x_train.shape[0]
time_steps=x_train.shape[1]
num_features=x_train.shape[2]
print('num runs:',num_runs)
print('time steps:',time_steps)
print('num features:',num_features)

# Assuming your data is in the format (batch, time, feature) and is stored in a numpy array
# For simplicity, we will create some dummy data
# Replace the following line with your actual data
# data = np.random.rand(1000, 10, 4)  # 1000 samples, 10 time steps, 4 features (2D position and 2D velocity)

# Let's say the labels are stored in a separate array
# Replace the following line with your actual labels
# labels = np.random.randint(0, 4, size=(1000,))  # 4 classes

# Reshape data for sklearn
# X = data.reshape(1000, -1)  # Reshape to (1000, 10*4)
x_train = x_train.reshape(num_runs, -1)  # Reshape to (batch, time*feature)
x_test = x_test.reshape(num_runs, -1)  # Reshape to (batch, time*feature)

# Split data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)

# Create and train the Random Forest classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(x_train, y_train)

# Get parameters for estimator
print(f'estimator parameters: {clf.get_params()}')

# Get feature importances
feature_importances = clf.feature_importances_
# Reshape feature importances to match the original data shape (excluding the batch size)
# This assumes that you still have your original data shape as (batch, time, feature)
reshaped_importances = feature_importances.reshape((time_steps, num_features))
# Create a bar plot for each time step
for i, importances in enumerate(reshaped_importances):
    plt.figure()
    plt.title(f'Feature importances for time step {i}')
    plt.bar(range(4), importances, align='center')
    plt.xticks(range(4), ['X Position', 'Y Position', 'X Velocity', 'Y Velocity'])
    plt.legend()
    plt.savefig(hparams.model_dir + "Feature_Importance.png")

# Evaluate the classifier
y_pred = clf.predict(x_test)
print('Accuracy:', accuracy_score(y_test, y_pred))
print('Confusion Matrix:\n', confusion_matrix(y_test, y_pred))

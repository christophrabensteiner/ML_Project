import os
from sklearn.model_selection import train_test_split
import data

#
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

training_dir = './data/fashion_png/training'
testing_dir = './data/fashion_png//testing'


imageWidth = 28
imageHeight = 28
imageSize = 32 * 32
NChannels = 1
NClasses = 10


BATCH_SIZE = 128


AddNoise = float(0)

#Load Data through pipeline

X_train, y_train = data.LoadTrainingData(training_dir, (imageWidth, imageHeight), AddNoise)
data.TrainingData = X_train
data.TrainingLables = y_train

X_test, y_test, NamesT, _, Paths = data.LoadTestingData(testing_dir, (imageWidth, imageHeight))
data.TestingData = X_test
data.TestingLables = y_test


# Training / Validation Split
X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=0.08333, random_state=1)

# Diese Daten werden im nächsten Schritt im CNN verarbeitet
# X_train, y_train ; X_validation, y_validation ; X_test, y_test
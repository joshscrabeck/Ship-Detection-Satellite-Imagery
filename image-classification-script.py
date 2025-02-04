
#Configuration environment
import os
data = {"username": "josh.scrabeck","key": "3cee02afa51f371fea201d506bbf147b"}
os.environ['KAGGLE_USERNAME'] = data["username"] # username from the json file
os.environ['KAGGLE_KEY'] = data["key"] # key from the json file

# importing the required libraries 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf 
from tensorflow import keras
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.densenet import DenseNet169 as PTModel, preprocess_input
import skimage
from skimage.io import imread
from skimage.util import montage as montage
from sklearn.model_selection import train_test_split
from kaggle.api.kaggle_api_extended import KaggleApi

# Initialize the Kaggle API
api = KaggleApi()
api.authenticate()

# Define the competition name
competition_name = "airbus-ship-detection"

# Define the directory to save the data
base_dir = "airbus_ship_detection_data"
os.makedirs(base_dir, exist_ok=True)

# Download the dataset files
print("Downloading dataset...")
api.competition_download_files(competition_name, path=base_dir)

# Unzip the downloaded files
import zipfile

with zipfile.ZipFile(os.path.join(base_dir, f"{competition_name}.zip"), 'r') as zip_ref:
    zip_ref.extractall(base_dir)

print("Dataset downloaded and extracted.")

# Define paths to the extracted files
train_csv_path = os.path.join(base_dir, "train_ship_segmentations_v2.csv")
train_image_dir = os.path.join(base_dir, "train_v2")
test_image_dir = os.path.join(base_dir, "test_v2")

# Load the CSV file
train_ship_segmentations = pd.read_csv(train_csv_path)

# List the images in the train and test directories
train_images = os.listdir(train_image_dir)
test_images = os.listdir(test_image_dir)

# Display some basic information
print(f"Number of rows in train_ship_segmentations: {len(train_ship_segmentations)}")
print(f"Number of images in train_v2: {len(train_images)}")
print(f"Number of images in test_v2: {len(test_images)}")

# Example: Display the first few rows of the CSV
print(train_ship_segmentations.head())

# Example: Display the first few image names in train_v2
print(train_images[:5])

# Example: Display the first few image names in test_v2
print(test_images[:5])

# Getting a count of number of trainign and testing images from the airbus ship detection dataset
train_data = os.listdir(train_path)
test_data = os.listdir(test_path)

print("Train Images : {} \nTest Images : {}".format(len(train_data), len(test_data)))
     
# Loading the map in for the labels
segmentation_data_df = pd.read_csv(base_path + "train_ship_segmentations_v2.csv")

# Filtering out data based on our available sample 
true_segment_df = segmentation_data_df[segmentation_data_df["ImageId"].isin(train_data)]

# Conducting a train and test split 
df = true_segment_df
total = len(df)

# splitting in a 70/30 ratio
train_ids, valid_ids = train_test_split(df.index, 
                 test_size = 0.3, 
                 stratify = df['HasShip'])

# Looking at the distribution of the labels
total = len(df)
ship = df['HasShip'].sum()
no_ship = total - ship
total_ships = int(df['TotalShips'].sum())
    
print(f"Images: {total} \nShips:  {total_ships}")
print(f"Images with ships:    {round(ship/total,2)} ({ship})")
print(f"Images with no ships: {round(no_ship/total,2)} ({no_ship})")
     
# Using keras ImageDataGenerator to preprocess the images
training_data = tf.keras.utils.image_dataset_from_directory(
    train_path, 
    labels = None, 
    color_mode = 'rgb',
    batch_size = 30,
    image_size = (256,256)
)

# Generate maps for the training and validation data
train_df = df[df.index.isin(train_ids)]
valid_df = df[df.index.isin(valid_ids)]


# Defininng required and optional parameters for the ImageDataGenerator for the training and testing 
# dataset
dg_args = dict(featurewise_center = False, 
                  samplewise_center = False,
                  rotation_range = 45, 
                  width_shift_range = 0.1, 
                  height_shift_range = 0.1, 
                  shear_range = 0.01,
                  zoom_range = [0.9, 1.25],  
                  brightness_range = [0.5, 1.5],
                  horizontal_flip = True, 
                  vertical_flip = True,
                  fill_mode = 'reflect',
                   data_format = 'channels_last',
              preprocessing_function = preprocess_input)

valid_args = dict(fill_mode = 'reflect',
                   data_format = 'channels_last',
                  preprocessing_function = preprocess_input)

core_idg = ImageDataGenerator(**dg_args)
valid_idg = ImageDataGenerator(**valid_args)

# Defininng required and optional parameters for the ImageDataGenerator for the training and testing 
# dataset
dg_args = dict(featurewise_center = False, 
                  samplewise_center = False,
                  rotation_range = 45, 
                  width_shift_range = 0.1, 
                  height_shift_range = 0.1, 
                  shear_range = 0.01,
                  zoom_range = [0.9, 1.25],  
                  brightness_range = [0.5, 1.5],
                  horizontal_flip = True, 
                  vertical_flip = True,
                  fill_mode = 'reflect',
                   data_format = 'channels_last',
              preprocessing_function = preprocess_input)

valid_args = dict(fill_mode = 'reflect',
                   data_format = 'channels_last',
                  preprocessing_function = preprocess_input)

core_idg = ImageDataGenerator(**dg_args)
valid_idg = ImageDataGenerator(**valid_args)

def flow_from_dataframe(img_data_gen, in_df, path_col, y_col, **dflow_args):
    """
    Map the images to their actual labels stored in the segmentation_df

    Args:
        img_data_gen (iterator): Generator object for the training dataset
        in_df (pd.DataFrame): Map of the images to their labels
        path_col (pd.Series): Column in the dataframe that contains the path to the image
        y_col (pd.Series): Label Column

    Returns:
        pd.Dataframe : Mapped images to their labels accessed using generators 
    """
    base_dir = base_path
    print('## Ignore next message from keras, values are replaced anyways')
    
    # Loading the images from the dataframe
    df_gen = img_data_gen.flow_from_directory(base_dir, 
                                     class_mode = 'sparse',
                                    **dflow_args)
    # Get file names
    df_gen.filenames = in_df.index
    # Get labels
    df_gen.classes = np.stack(in_df[y_col].values)
    # Sample size
    df_gen.samples = in_df.shape[0]
    df_gen.n = in_df.shape[0]
    df_gen._set_index_array()
    df_gen.directory = '' # since we have the full path
    print('Reinserting dataframe: {} images'.format(in_df.shape[0]))
    return df_gen

# getting the trainign and validation generators
train_gen = flow_from_dataframe(core_idg, train_df, 
                             path_col = 'ImageId',
                            y_col = 'HasShip', 
                            target_size = (128,128),
                             color_mode = 'rgb',
                            batch_size = 100)

# used a fixed dataset for evaluating the algorithm
valid_x, valid_y = next(flow_from_dataframe(valid_idg, 
                               valid_df, 
                             path_col = 'path',
                            y_col = 'HasShip', 
                            target_size = (128,128),
                             color_mode = 'rgb',
                            batch_size = 100)) # one big batch
print(valid_x.shape, valid_y.shape)

# Getting a montage of images
montage_rgb = lambda x: np.stack([montage(x[:, :, :, i]) for i in range(x.shape[3])], -1)
RGB_FLIP = 1

# Getting training and testing data
t_x, t_y = next(train_gen)
print('x', t_x.shape, t_x.dtype, t_x.min(), t_x.max())
print('y', t_y.shape, t_y.dtype, t_y.min(), t_y.max())
fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (20, 10))
ax1.imshow(montage_rgb((t_x - t_x.min()) / (t_x.max() - t_x.min()))[:, :, ::RGB_FLIP], cmap='gray')
ax1.set_title('images')
ax2.plot(t_y)
ax2.set_title('ships')

# Importing required packages
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.applications.densenet import DenseNet169, preprocess_input
from keras.applications.densenet import DenseNet121, preprocess_input

# Defining the required parameters for the model
GAUSSIAN_NOISE = 0.1
UPSAMPLE_MODE = 'SIMPLE'
# number of validation images to use
VALID_IMG_COUNT = 1000
# maximum number of training images
MAX_TRAIN_IMAGES = 8000 
IMG_SIZE = (224, 224) # [(224, 224), (384, 384), (512, 512), (640, 640)]
BATCH_SIZE = 64 # [1, 8, 16, 24]
DROPOUT = 0.5
DENSE_COUNT = 128
LEARN_RATE = 0.001
RGB_FLIP = 1 # should rgb be flipped when rendering images

# Getting the model
base_pretrained_model = VGG16(input_shape =  t_x.shape[1:], include_top = False, weights = 'imagenet')
base_pretrained_model.trainable = False

# Adding input and noise layers
img_in = keras.layers.Input(t_x.shape[1:], name='Image_RGB_In')
img_noise = keras.layers.GaussianNoise(GAUSSIAN_NOISE)(img_in)
pt_features = base_pretrained_model(img_noise)
pt_depth = base_pretrained_model.get_output_shape_at(0)[-1]
bn_features = keras.layers.BatchNormalization()(pt_features)
feature_dropout = keras.layers.SpatialDropout2D(DROPOUT)(bn_features)
gmp_dr = keras.layers.GlobalMaxPooling2D()(feature_dropout)
dr_steps = keras.layers.Dropout(DROPOUT)(keras.layers.Dense(DENSE_COUNT, activation = 'relu')(gmp_dr))
out_layer = keras.layers.Dense(1, activation = 'sigmoid')(dr_steps)

# Genetating the final model with the modifications 
ship_model = keras.models.Model(inputs = [img_in], outputs = [out_layer], name = 'full_model')

# Compiling the model with Adam optimizer and binary crossentropy loss
ship_model.compile(optimizer = keras.optimizers.Adam(learning_rate=LEARN_RATE), loss = 'binary_crossentropy', metrics = ['binary_accuracy'])

# Printing out the summary of the model 
ship_model.summary()

# Defining various callbacks
weight_path="{}_weights.best.hdf5".format('boat_detector')

checkpoint = ModelCheckpoint(weight_path, monitor='val_loss', verbose=1, 
                             save_best_only=True, mode='min', save_weights_only = True)

reduceLROnPlat = ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=10, verbose=1, mode='auto', epsilon=0.0001, cooldown=5, min_lr=0.0001)

early = EarlyStopping(monitor="val_loss", mode="min", patience=10) # probably needs to be more patient, but kaggle time is limited

callbacks_list = [checkpoint, early, reduceLROnPlat]

# Fit the model
train_gen.batch_size = BATCH_SIZE
VGG16_history = ship_model.fit_generator(train_gen, 
                         steps_per_epoch=train_gen.n//BATCH_SIZE,
                      validation_data=(valid_x, valid_y), 
                      epochs=10, 
                      callbacks=callbacks_list,
                      workers=3)

# Save the model and its history
os.mkdir(base_path + "VGG16/")
ship_model.save(base_path+ "VGG16/")

# Generating the plots 
plotting_history = VGG16_history.history

# Training and validation loss plot 
ax = plt.subplot(111)
ax.plot(plotting_history["loss"], label = "Training Loss")
ax.plot(plotting_history["val_loss"], label = "Validation Loss")
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_xticks(range(0,10))
ax.set_xlabel('Epochs', labelpad=15, color='#333333')
ax.set_ylabel('Loss', labelpad=15, color='#333333')
ax.set_title('Reduction of the Training and Validation Loss', pad=15, color='#333333',
             weight='bold')
ax.legend()
     

# Accuracy plot
ax = plt.subplot(111)
ax.plot(plotting_history["binary_accuracy"], linestyle = "--", marker = 'o', label = "Training Accuracy")
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_xticks(range(0,10))
ax.set_xlabel('Epochs', labelpad=15, color='#333333')
ax.set_ylabel('Accuracy', labelpad=15, color='#333333')
ax.set_title('Training and Validation Accuracy trend with increasing epochs', pad=15, color='#333333',
             weight='bold')

####Fitting of the VGG19 model

# import required packages
from keras.applications.vgg19 import VGG19, preprocess_input
from tensorflow.keras.optimizers import Adam
     

# Defining the required parameters for the model
base_model = VGG19(weights=None, include_top=False, input_shape=(128, 128, 3))

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.8)(x)
predictions = Dense(2, activation='softmax')(x)
model = Model(inputs = base_model.input, outputs = predictions)

adam = Adam(learning_rate=0.0001)
# Compiling the model with Adam optimizer and binary crossentropy loss
model.compile(optimizer=adam, loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

     

# Fitting the model
vgg19_history = model.fit_generator(train_gen, 
                         steps_per_epoch=train_gen.n//64,
                      validation_data=(valid_x, valid_y), 
                      epochs=10, 
                      callbacks=callbacks_list,
                      workers=3)

# Saving the model and its history
plotting_history = pd.DataFrame(vgg19_history.history)
plotting_history.to_csv(base_path + "vgg19.csv")

# Accuracy plot
epochs = np.arange(1, 11)
acc = [0.7776, 0.7814, 0.7793, 0.7799, 0.7791, 0.7795, 0.7811, 0.7803, 0.7806, 0.7804]

ax = plt.subplot(111)
ax.plot(epochs, acc, linestyle = "--", marker = 'o', label = "Training Accuracy")
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_xticks(range(0,10))
ax.set_xlabel('Epochs', labelpad=15, color='#333333')
ax.set_ylabel('Accuracy', labelpad=15, color='#333333')
ax.set_title('Training and Validation Accuracy trend with increasing epochs', pad=15, color='#333333',
             weight='bold')
plt.savefig('vgg19_acc.png')
os.files.download("vgg19_acc.png") 
plt.show()

# Training and validation loss plot
plt.figure(figsize=(7,5))
ax = plt.subplot(111)
ax.plot(plotting_history["loss"], label = "Training Loss")
ax.plot(plotting_history["val_loss"], label = "Validation Loss")
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_xticks(range(0,10))
ax.set_xlabel('Epochs', labelpad=15, color='#333333')
ax.set_ylabel('Loss', labelpad=15, color='#333333')
ax.set_title('Reduction of the Training and Validation Loss for VGG19', pad=15, color='#333333',
             weight='bold')
plt.savefig('vgg19_loss.png')
os.files.download("vgg19_loss.png") 
ax.legend()
plt.show()


# importing required packages
from keras.models import Model, load_model
from keras.layers import Input, Dropout, Flatten, Dense
from keras.layers.core import Lambda
from keras.layers import Conv2D, Conv2DTranspose, MaxPooling2D, GlobalAveragePooling2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import Sequential
from keras import backend as K
     
# Defining a sequential model
model = Sequential()
model.add(Conv2D(filters=16, kernel_size=2, padding='same', activation='relu', 
                        input_shape=(128, 128, 3)))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(filters=32, kernel_size=2, padding='same', activation='relu', 
                        input_shape=(128, 128, 3)))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(filters=64, kernel_size=2, padding='same', activation='relu', 
                        input_shape=(128, 128, 3)))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(500, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(133, activation='softmax'))

# Compiling the model with rmsprop optimizer and categorical crossentropy loss
model.compile(optimizer='rmsprop', loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

# Printing out the summary of the model
model.summary()

train_gen.batch_size = 64
# Fit the model
cnn_history = model.fit_generator(train_gen, 
                         steps_per_epoch=train_gen.n//64,
                      validation_data=(valid_x, valid_y), 
                      epochs=10, 
                      callbacks=callbacks_list,
                      workers=3)

# Save the model and its history
model.save(base_path+ "CNN/")
plotting_history = pd.DataFrame(cnn_history.history)
plotting_history.to_csv(base_path + "cnn.csv")

# Training and validation loss plot
ax = plt.subplot(111)
ax.plot(plotting_history["loss"], label = "Training Loss")
ax.plot(plotting_history["val_loss"], label = "Validation Loss")
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_xticks(range(0,10))
ax.set_xlabel('Epochs', labelpad=15, color='#333333')
ax.set_ylabel('Loss', labelpad=15, color='#333333')
ax.set_title('Reduction of the Training and Validation Loss', pad=15, color='#333333',
             weight='bold')
ax.legend()

# Accuracy plot
ax = plt.subplot(111)
ax.plot(plotting_history["accuracy"], linestyle = "--", marker = 'o', label = "Training Accuracy")
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_xticks(range(0,10))
ax.set_xlabel('Epochs', labelpad=15, color='#333333')
ax.set_ylabel('Accuracy', labelpad=15, color='#333333')
ax.set_title('Training and Validation Accuracy trend with increasing epochs', pad=15, color='#333333',
             weight='bold')

###Fitting of the ResNet50 model
# importing required packages
from tensorflow.keras.applications.resnet50 import ResNet50

# initializing the pretrained model
base_model =ResNet50(weights= None, include_top=False, input_shape= (128, 128, 3))

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.8)(x)
predictions = Dense(2, activation= 'softmax')(x)
model = Model(inputs = base_model.input, outputs = predictions)

adam = Adam(lr=0.0001)
model.compile(optimizer= adam, loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

# Fit the model
resnet_history = model.fit_generator(train_gen, 
                         steps_per_epoch=train_gen.n//64,
                      validation_data=(valid_x, valid_y), 
                      epochs=10, 
                      callbacks=callbacks_list,
                      workers=3)

# Saving the model and its history
model.save(base_path + "ResNet50/")
plotting_history = pd.DataFrame(resnet_history.history)
plotting_history.to_csv(base_path + "resnet50.csv")

# Training and validation loss plot
ax = plt.subplot(111)
ax.plot(plotting_history["loss"], label = "Training Loss")
ax.plot(plotting_history["val_loss"], label = "Validation Loss")
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_xticks(range(0,10))
ax.set_xlabel('Epochs', labelpad=15, color='#333333')
ax.set_ylabel('Loss', labelpad=15, color='#333333')
ax.set_title('Reduction of the Training and Validation Loss', pad=15, color='#333333',
             weight='bold')
ax.legend()

# Accuracy plot
ax = plt.subplot(111)
ax.plot(plotting_history["accuracy"], linestyle = "--", marker = 'o', label = "Training Accuracy")
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_xticks(range(0,10))
ax.set_xlabel('Epochs', labelpad=15, color='#333333')
ax.set_ylabel('Accuracy', labelpad=15, color='#333333')
ax.set_title('Training and Validation Accuracy trend with increasing epochs', pad=15, color='#333333',
             weight='bold')






















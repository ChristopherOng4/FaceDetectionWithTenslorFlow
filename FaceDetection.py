import os
import time
import uuid
import cv2

import tensorflow as tf
import json 
import numpy as np
from matplotlib import pyplot as plt
import albumentations as alb

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Dense, GlobalMaxPooling2D
from tensorflow.keras.applications import VGG16

from tensorflow.keras.models import load_model

# Caputres 30 images
IMAGES_PATH = os.path.join('data', 'images')
number_images = 30

# Caputres the images from the camera 
# cap = cv2.VideoCapture(0)
# for imgnum in range(number_images):
#     print('Collecting image {}'.format(imgnum))
#     ret, frame = cap.read()
#     imgname = os.path.join(IMAGES_PATH,f'{str(uuid.uuid1())}.jpg')
#     cv2.imwrite(imgname, frame)
#     cv2.imshow('frame', frame)
#     time.sleep(0.5)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
# cap.release()
# cv2.destroyAllWindows()

# Avoid OOM errors by setting GPU Memory Consumption Growth
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus: 
    tf.config.experimental.set_memory_growth(gpu, True)

tf.config.list_physical_devices('GPU')

images = tf.data.Dataset.list_files('data\\images\\*.jpg',shuffle = False)

images.as_numpy_iterator().next()

# Define a function to load images from file paths
def load_image(x): 
    byte_img = tf.io.read_file(x) # Read the image file as a byte string
    img = tf.io.decode_jpeg(byte_img) # Decode the byte string to an image tensor
    return img

images = images.map(load_image)

images.as_numpy_iterator().next()

type(images)

# View Raw Images with Matplotlib
# Create an iterator for the image dataset
image_generator = images.batch(4).as_numpy_iterator()

# Get the next batch of images
plot_images = image_generator.next()

# Create a subplot for each image in the batch
fig, ax = plt.subplots(ncols=4, figsize=(20,20))
for idx, image in enumerate(plot_images): # Display each image in the subplot
    ax[idx].imshow(image) 
plt.show()

# Iterate through the partitions
for folder in ['train','test','val']:
    for file in os.listdir(os.path.join('data', folder, 'images')): # Iterate through the image files in each partition
        
        # Extract the filename without extension
        filename = file.split('.')[0]+'.json'
        existing_filepath = os.path.join('data','labels', filename) # Define the path to the corresponding label file
        if os.path.exists(existing_filepath): # Check if the label file exists
            new_filepath = os.path.join('data',folder,'labels',filename)
            os.replace(existing_filepath, new_filepath)   

# Define augmentation pipeline
augmentor = alb.Compose([alb.RandomCrop(width=450, height=450), 
                         alb.HorizontalFlip(p=0.5), 
                         alb.RandomBrightnessContrast(p=0.2),
                         alb.RandomGamma(p=0.2), 
                         alb.RGBShift(p=0.2), 
                         alb.VerticalFlip(p=0.5)], 
                       bbox_params=alb.BboxParams(format='albumentations', 
                                                  label_fields=['class_labels']))

img = cv2.imread(os.path.join('data','train', 'images', 'f5ac1b32-ff8a-11ee-ad4a-9c6b00527e70.jpg'))
with open(os.path.join('data', 'train', 'labels', 'f5ac1b32-ff8a-11ee-ad4a-9c6b00527e70.json'), 'r') as f:
    label = json.load(f)

label['shapes'][0]['points']

# Extract Coordinates and Rescale to Match Image Resolution
coords = [0,0,0,0]
coords[0] = label['shapes'][0]['points'][0][0]
coords[1] = label['shapes'][0]['points'][0][1]
coords[2] = label['shapes'][0]['points'][1][0]
coords[3] = label['shapes'][0]['points'][1][1]

coords
coords = list(np.divide(coords, [640,480,640,480]))
coords

# Apply augmentations to the image and bounding box
augmented = augmentor(image=img, bboxes=[coords], class_labels=['face'])
augmented['bboxes'][0][2:]
augmented['bboxes']

# Display the augmented image with bounding box
cv2.rectangle(augmented['image'], 
              tuple(np.multiply(augmented['bboxes'][0][:2], [450,450]).astype(int)),
              tuple(np.multiply(augmented['bboxes'][0][2:], [450,450]).astype(int)), 
                    (255,0,0), 2)

plt.imshow(augmented['image'])

# for partition in ['train','test','val']: 
#     for image in os.listdir(os.path.join('data', partition, 'images')):
#         img = cv2.imread(os.path.join('data', partition, 'images', image))

#         coords = [0,0,0.00001,0.00001]
#         label_path = os.path.join('data', partition, 'labels', f'{image.split(".")[0]}.json')
#         if os.path.exists(label_path):
#             with open(label_path, 'r') as f:
#                 label = json.load(f)

#             coords[0] = label['shapes'][0]['points'][0][0]
#             coords[1] = label['shapes'][0]['points'][0][1]
#             coords[2] = label['shapes'][0]['points'][1][0]
#             coords[3] = label['shapes'][0]['points'][1][1]
#             coords = list(np.divide(coords, [640,480,640,480]))

#         try: 
#             for x in range(60):
#                 augmented = augmentor(image=img, bboxes=[coords], class_labels=['face'])
#                 cv2.imwrite(os.path.join('aug_data', partition, 'images', f'{image.split(".")[0]}.{x}.jpg'), augmented['image'])

#                 annotation = {}
#                 annotation['image'] = image

#                 if os.path.exists(label_path):
#                     if len(augmented['bboxes']) == 0: 
#                         annotation['bbox'] = [0,0,0,0]
#                         annotation['class'] = 0 
#                     else: 
#                         annotation['bbox'] = augmented['bboxes'][0]
#                         annotation['class'] = 1
#                 else: 
#                     annotation['bbox'] = [0,0,0,0]
#                     annotation['class'] = 0 


#                 with open(os.path.join('aug_data', partition, 'labels', f'{image.split(".")[0]}.{x}.json'), 'w') as f:
#                     json.dump(annotation, f)

#         except Exception as e:
#             print(e)

# Load augmented images for training
train_images = tf.data.Dataset.list_files('aug_data\\train\\images\\*.jpg', shuffle=False)
train_images = train_images.map(load_image)
train_images = train_images.map(lambda x: tf.image.resize(x, (120,120)))
train_images = train_images.map(lambda x: x/255)

# Load augmented images for testing
test_images = tf.data.Dataset.list_files('aug_data\\test\\images\\*.jpg', shuffle=False)
test_images = test_images.map(load_image)
test_images = test_images.map(lambda x: tf.image.resize(x, (120,120)))
test_images = test_images.map(lambda x: x/255)

# Load augmented images for validation
val_images = tf.data.Dataset.list_files('aug_data\\val\\images\\*.jpg', shuffle=False)
val_images = val_images.map(load_image)
val_images = val_images.map(lambda x: tf.image.resize(x, (120,120)))
val_images = val_images.map(lambda x: x/255)

images.as_numpy_iterator().next()

# Define a function to load labels from JSON files
def load_labels(label_path):
    with open(label_path.numpy(), 'r', encoding = "utf-8") as f:
        label = json.load(f)
        
    return [label['class']], label['bbox']

# Load labels for training dataset
train_labels = tf.data.Dataset.list_files('aug_data\\train\\labels\\*.json', shuffle=False)
train_labels = train_labels.map(lambda x: tf.py_function(load_labels, [x], [tf.uint8, tf.float16]))

# Load labels for testing dataset
test_labels = tf.data.Dataset.list_files('aug_data\\test\\labels\\*.json', shuffle=False)
test_labels = test_labels.map(lambda x: tf.py_function(load_labels, [x], [tf.uint8, tf.float16]))

# Load labels for validation dataset
val_labels = tf.data.Dataset.list_files('aug_data\\val\\labels\\*.json', shuffle=False)
val_labels = val_labels.map(lambda x: tf.py_function(load_labels, [x], [tf.uint8, tf.float16]))

# Get a sample label from the training dataset
train_labels.as_numpy_iterator().next()

# Calculate the lengths of different partitions
len(train_images), len(train_labels), len(test_images), len(test_labels), len(val_images), len(val_labels)

# Combine images and labels for training dataset
train = tf.data.Dataset.zip((train_images, train_labels))
train = train.shuffle(5000)
train = train.batch(8)
train = train.prefetch(4)

# Combine images and labels for testing dataset
test = tf.data.Dataset.zip((test_images, test_labels))
test = test.shuffle(1300)
test = test.batch(8)
test = test.prefetch(4)

# Combine images and labels for validation dataset
val = tf.data.Dataset.zip((val_images, val_labels))
val = val.shuffle(1000)
val = val.batch(8)
val = val.prefetch(4)

# Get a sample from the training dataset to visualize
train.as_numpy_iterator().next()[1]

# Get a sample batch from the dataset
data_samples = train.as_numpy_iterator()
res = data_samples.next()

fig, ax = plt.subplots(ncols=4, figsize=(20,20))
for idx in range(4):
    # Make a writable copy of the image from the result set
    sample_image = np.copy(res[0][idx])

    # Coordinates for the rectangle
    sample_coords = res[1][1][idx]

    # Convert coordinates for use with cv2.rectangle and apply the rectangle
    start_point = tuple(np.multiply(sample_coords[:2], [120,120]).astype(int))
    end_point = tuple(np.multiply(sample_coords[2:], [120,120]).astype(int))
    color = (255, 0, 0)  # Red color in BGR
    thickness = 2

    # Draw the rectangle on the writable copy of the image
    cv2.rectangle(sample_image, start_point, end_point, color, thickness)

    # Display the image
    ax[idx].imshow(cv2.cvtColor(sample_image, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB for correct color display in matplotlib
    ax[idx].axis('off')  # Hide axes

plt.show()


vgg = VGG16(include_top=False)
vgg.summary()

def build_model(): 
    input_layer = Input(shape=(120,120,3))
    
    vgg = VGG16(include_top=False)(input_layer) # Pass input through VGG16 model

    # Classification Model  
    f1 = GlobalMaxPooling2D()(vgg)
    class1 = Dense(2048, activation='relu')(f1)
    class2 = Dense(1, activation='sigmoid')(class1)
    
    # Bounding box model
    f2 = GlobalMaxPooling2D()(vgg)
    regress1 = Dense(2048, activation='relu')(f2)
    regress2 = Dense(4, activation='sigmoid')(regress1)
    
    facetracker = Model(inputs=input_layer, outputs=[class2, regress2])
    return facetracker

# Get a sample batch of images and make predictions using the model
facetracker = build_model()
facetracker.summary()
X, y = train.as_numpy_iterator().next()
X.shape
classes, coords = facetracker.predict(X)
classes, coords

# Calculate the number of batches per epoch
batches_per_epoch = len(train)
lr_decay = (1./0.75 -1)/batches_per_epoch # Calculate the learning rate decay

opt = tf.keras.optimizers.Adam(learning_rate=0.0001, decay=lr_decay) 

# Define the localization loss function
def localization_loss(y_true, yhat):            
    delta_coord = tf.reduce_sum(tf.square(y_true[:,:2] - yhat[:,:2]))
                  
    h_true = y_true[:,3] - y_true[:,1] 
    w_true = y_true[:,2] - y_true[:,0] 

    h_pred = yhat[:,3] - yhat[:,1] 
    w_pred = yhat[:,2] - yhat[:,0] 
    
    delta_size = tf.reduce_sum(tf.square(w_true - w_pred) + tf.square(h_true-h_pred))
    
    return delta_coord + delta_size

classloss = tf.keras.losses.BinaryCrossentropy()
regressloss = localization_loss

localization_loss(y[1], coords)
classloss(y[0], classes)
regressloss(y[1], coords)


# Define a custom model class inheriting from tf.keras.Model
class FaceTracker(Model): 
    def __init__(self, eyetracker, **kwargs): 
        super().__init__(**kwargs)
        self.model = eyetracker

    def compile(self, opt, classloss, localizationloss, **kwargs):
        super().compile(**kwargs)
        self.closs = classloss
        self.lloss = localizationloss
        self.opt = opt
    
    def train_step(self, batch, **kwargs): 
        X, y = batch

        with tf.GradientTape() as tape: 
            classes, coords = self.model(X, training=True)

            # Use reshaped tensors for loss computation
            y0_reshaped = tf.reshape(y[0], [-1])  # Ensure y[0] is a 1D tensor matching the batch size
            y1_reshaped = tf.cast(tf.reshape(y[1], [-1, tf.shape(y[1])[-1]]), tf.float32)  # Ensure proper shape for coordinates

            batch_classloss = self.closs(y0_reshaped, classes)
            batch_localizationloss = self.lloss(y1_reshaped, coords)
            
            total_loss = batch_localizationloss + 0.5 * batch_classloss
            
            grad = tape.gradient(total_loss, self.model.trainable_variables)
        
        self.opt.apply_gradients(zip(grad, self.model.trainable_variables))
        
        return {"total_loss": total_loss, "class_loss": batch_classloss, "regress_loss": batch_localizationloss}
    
    def test_step(self, batch, **kwargs): 
        X, y = batch
        
        classes, coords = self.model(X, training=False)
        
        # Use reshaped tensors for loss computation in the test phase
        y0_reshaped = tf.reshape(y[0], [-1])
        y1_reshaped = tf.cast(tf.reshape(y[1], [-1, tf.shape(y[1])[-1]]), tf.float32)
        
        batch_classloss = self.closs(y0_reshaped, classes)
        batch_localizationloss = self.lloss(y1_reshaped, coords)
        total_loss = batch_localizationloss + 0.5 * batch_classloss
        
        return {"total_loss": total_loss, "class_loss": batch_classloss, "regress_loss": batch_localizationloss}
        
    def call(self, X, **kwargs): 
        return self.model(X, **kwargs)
        
model = FaceTracker(facetracker)
model.compile(opt, classloss, regressloss)

logdir='logs'
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
hist = model.fit(train, epochs=10, validation_data=val, callbacks=[tensorboard_callback])

keys = hist.history.keys()
fig, ax = plt.subplots(ncols=3, figsize=(20,5))

# Plot the training history (losses)
if 'total_loss' in keys and 'val_total_loss' in keys:
    ax[0].plot(hist.history['total_loss'], color='teal', label='loss')
    ax[0].plot(hist.history['val_total_loss'], color='orange', label='val loss')
    ax[0].title.set_text('Loss')
    ax[0].legend()

if 'class_loss' in keys and 'val_class_loss' in keys:
    ax[1].plot(hist.history['class_loss'], color='teal', label='class loss')
    ax[1].plot(hist.history['val_class_loss'], color='orange', label='val class loss')
    ax[1].title.set_text('Classification Loss')
    ax[1].legend()

if 'regress_loss' in keys and 'val_regress_loss' in keys:
    ax[2].plot(hist.history['regress_loss'], color='teal', label='regress loss')
    ax[2].plot(hist.history['val_regress_loss'], color='orange', label='val regress loss')
    ax[2].title.set_text('Regression Loss')
    ax[2].legend()

plt.show()

# Make predictions on test set 
test_data = test.as_numpy_iterator()
test_sample = test_data.next()
yhat = facetracker.predict(test_sample[0])
fig, ax = plt.subplots(ncols=4, figsize=(20,20))

for idx in range(4): 
    sample_image = test_sample[0][idx].copy()  # Make a copy if necessary to avoid modifying original data
    sample_coords = yhat[1][idx]

    # Ensure the image data type is correct
    if sample_image.dtype != np.uint8:
        sample_image = (sample_image * 255).astype(np.uint8)  # Assuming the image is scaled between 0 and 1

    if yhat[0][idx] > 0.9:
        # Calculate coordinates and ensure they are within the image bounds and are integers
        start_point = np.multiply(sample_coords[:2], [120, 120]).astype(int)
        end_point = np.multiply(sample_coords[2:], [120, 120]).astype(int)
        
        # Ensure coordinates are within image bounds
        start_point = np.clip(start_point, 0, sample_image.shape[1::-1])
        end_point = np.clip(end_point, 0, sample_image.shape[1::-1])

        # Draw the rectangle
        cv2.rectangle(sample_image, 
                      tuple(start_point),  # Convert start_point to tuple
                      tuple(end_point),    # Convert end_point to tuple
                      (255, 0, 0), 2)
    
    # Convert image from BGR to RGB for correct display with matplotlib
    ax[idx].imshow(cv2.cvtColor(sample_image, cv2.COLOR_BGR2RGB))
    ax[idx].axis('off')  # Turn off axis

# Save the model
facetracker.save('facetracker.keras')
facetracker = load_model('facetracker.keras')

cap = cv2.VideoCapture(0)
while cap.isOpened():
    _ , frame = cap.read()
    frame = frame[50:500, 50:500,:]
    
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = tf.image.resize(rgb, (120,120))
    
    yhat = facetracker.predict(np.expand_dims(resized/255,0))
    sample_coords = yhat[1][0]
    
    if yhat[0] > 0.5: 
        # Controls the main rectangle
        cv2.rectangle(frame, 
                      tuple(np.multiply(sample_coords[:2], [450,450]).astype(int)),
                      tuple(np.multiply(sample_coords[2:], [450,450]).astype(int)), 
                            (255,0,0), 2)
        # Controls the label rectangle
        cv2.rectangle(frame, 
                      tuple(np.add(np.multiply(sample_coords[:2], [450,450]).astype(int), 
                                    [0,-30])),
                      tuple(np.add(np.multiply(sample_coords[:2], [450,450]).astype(int),
                                    [80,0])), 
                            (255,0,0), -1)
        
        # Controls the text rendered
        cv2.putText(frame, 'face', tuple(np.add(np.multiply(sample_coords[:2], [450,450]).astype(int),
                                               [0,-5])),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
    
    cv2.imshow('EyeTrack', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
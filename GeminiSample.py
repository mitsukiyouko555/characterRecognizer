# This is a reference for myself of the code that Gemini Generated after arduous promopting to get the results I want.

# the stuff I asked consists of the following logic that I came up on my own:

#1. Due to the fact that when you train a model, say you have two cells in Jupyter notebook.. If you trained the model in cell 1, and you want to train the model with a different architecture or settings, you would need to reset the model each time.. So I figured, if i can make a class of the model then every time i instantiate it it would be like new! so i wouldn't have to reset it everytime.

#2. I wanted to experiement with a bunch of different combinations but instead of writing out all the layers each time i want to make it so that it is a def or class that i can call upon and easily go through all the different options in just a few lines once everything has been set up properly.

#3. I also want to show a fancy graph of the training outcome as well as have it show me which combination is the best for each of the 4 phases.

#4 For the sake of clarity, I also asked Gemini to confirm which items I can reuse (classes, defs), vs which ones I would have to re-code for every phase.


# -*- coding: utf-8 -*-
"""
Multi-Phase Image Classification Experiment Notebook (Modern Data Loading)
"""

# ## 1. Import Libraries

import tensorflow as tf
import keras
from keras import layers, models, optimizers
from keras import callbacks 
from keras.src.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np
import matplotlib.pyplot as plt
import json
import os
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

# ## 2. Reusable Definitions and Classes

# ### 2.1. Model Attribute Classes (Reusable - instantiate with phase-specific dimensions)
class CNNModelAttributes1:
    def __init__(self, img_height, img_width, num_channels, num_classes):
        self.input_shape = (img_height, img_width, num_channels)
        self.conv1_filters = 32
        # ... other attributes ...
        self.num_classes = num_classes

class CNNModelAttributes2:
    def __init__(self, img_height, img_width, num_channels, num_classes):
        self.input_shape = (img_height, img_width, num_channels)
        self.conv1_filters = 64
        # ... different attributes ...
        self.num_classes = num_classes

# Add more attribute classes here

# ### 2.2. Early Stopping Configuration Classes (Reusable)
class EarlyStoppingConfig1:
    def __init__(self):
        self.monitor = 'val_loss'
        self.patience = 10
        self.restore_best_weights = True

class EarlyStoppingConfig2:
    def __init__(self):
        self.monitor = 'val_accuracy'
        self.patience = 5
        self.restore_best_weights = False

# Add more early stopping config classes here

# ### 2.3. Model Architecture Classes (Reusable - ensure they can handle varying input shapes and output units)
class LeNet5(models.Model):
    def __init__(self, model_attributes, early_stopping_config):
        super(LeNet5, self).__init__()
        self.attributes = model_attributes
        self.early_stopping_config = early_stopping_config
        self.conv1 = layers.Conv2D(6, (5, 5), activation='relu', input_shape=self.attributes.input_shape)
        self.pool1 = layers.MaxPooling2D((2, 2))
        self.conv2 = layers.Conv2D(16, (5, 5), activation='relu')
        self.pool2 = layers.MaxPooling2D((2, 2))
        self.flatten = layers.Flatten()
        self.fc1 = layers.Dense(120, activation='relu')
        self.fc2 = layers.Dense(84, activation='relu')
        self.output_layer = layers.Dense(self.attributes.num_classes, activation='softmax') # Adjust for multilabel

    def call(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return self.output_layer(x)

class SimpleAlexNet(models.Model):
    def __init__(self, model_attributes, early_stopping_config):
        super(SimpleAlexNet, self).__init__()
        self.attributes = model_attributes
        self.early_stopping_config = early_stopping_config
        # ... (as defined before - ensure output layer is flexible) ...
        pass

# Add more model architecture classes here

# ### 2.4. Model Compiler Class (Reusable)
class ModelCompiler:
    def __init__(self, optimizer='adam', learning_rate=0.001, loss='categorical_crossentropy', metrics=['accuracy']):
        self.optimizer_name = optimizer
        self.learning_rate = learning_rate
        self.loss = loss
        self.metrics = metrics
        self.optimizer_instance = self._get_optimizer()

    def _get_optimizer(self):
        if self.optimizer_name.lower() == 'adam':
            return optimizers.Adam(learning_rate=self.learning_rate)
        # ... other optimizers ...
        else:
            raise ValueError(f"Optimizer '{self.optimizer_name}' not supported.")

    def compile(self, model):
        model.compile(optimizer=self.optimizer_instance,
                      loss=self.loss,
                      metrics=self.metrics)
        return model

# ### 2.5. Data Loading and Preprocessing Function (Modern Approach)
def load_and_preprocess_data(train_dir, val_dir, img_height, img_width, batch_size, label_mode='categorical'):
    """Loads and preprocesses image data using tf.keras.utils.image_dataset_from_directory."""

    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        labels='inferred',
        label_mode=label_mode,
        image_size=(img_height, img_width),
        batch_size=batch_size,
        shuffle=True
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        val_dir,
        labels='inferred',
        label_mode=label_mode,
        image_size=(img_height, img_width),
        batch_size=batch_size,
        shuffle=False
    )

    # Normalize the images
    normalization_layer = layers.Rescaling(1./255)
    normalized_train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
    normalized_val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))

    return normalized_train_ds, normalized_val_ds, len(train_ds.class_names)

# ### 2.6. Data Augmentation Function (Using tf.image)
def augment_data(image, label):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_rotation(image, factor=0.2)
    image = tf.image.random_brightness(image, max_delta=0.2)
    image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
    return image, label

# ### 2.7. Apply Augmentation to the Training Dataset
def prepare_augmented_dataset(dataset, batch_size):
    augmented_ds = dataset.map(augment_data)
    augmented_ds = augmented_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    return augmented_ds

# ### 2.8. Reusable Training Loop Function (Adjusted for Dataset API)
def run_phase_training(train_ds_orig, train_ds_aug, val_ds, model_classes, attribute_classes, es_configs, epochs, checkpoint_base_path, compiler, phase_name, results_list):
    """Runs training and evaluation for a given phase across all model/attribute/ES combinations using tf.data.Dataset."""
    print(f"\n--- Starting Phase: {phase_name} ---")
    for model_class in model_classes:
        for attr_config in attribute_classes:
            for es_config in es_configs:
                result = train_and_evaluate_model(
                    train_ds_orig,
                    train_ds_aug,
                    val_ds,
                    model_class,
                    attr_config,
                    es_config,
                    epochs,
                    checkpoint_base_path,
                    compiler
                )
                results_list.append({**result, 'phase': phase_name})

def train_and_evaluate_model(train_ds, train_ds_augmented, val_ds, model_class, attr_config, es_config, epochs, checkpoint_base_path, compiler):
    """Trains and evaluates a model with specified tf.data.Dataset objects."""
    print(f"\n--- Training: {model_class.__name__} with Attributes: {attr_config.__class__.__name__}, ES: {es_config.__class__.__name__} ---")

    # Instantiate Model
    model = model_class(attr_config, es_config)

    # Compile Model
    model = compiler.compile(model)

    # Define Callbacks
    early_stopping = EarlyStopping(**es_config.__dict__)
    checkpoint_path = f'{checkpoint_base_path}_{model_class.__name__}_{attr_config.__class__.__name__}_{es_config.__class__.__name__}_best.h5'
    model_checkpoint = ModelCheckpoint(checkpoint_path, save_best_only=True, monitor=es_config.__dict__.get('monitor', 'val_loss'), mode='min')
    callbacks = [early_stopping, model_checkpoint]

    # Train Model (Epoch-wise control for augmentation)
    history = None
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        if epoch == 0:
            history = model.fit(
                train_ds,
                epochs=1,
                validation_data=val_ds,
                callbacks=callbacks,
                verbose=1
            )
        else:
            history = model.fit(
                train_ds_augmented,
                epochs=1,
                validation_data=val_ds,
                callbacks=callbacks,
                verbose=1
            )

    # Load the best weights
    model.load_weights(checkpoint_path)

    # Evaluate the best model
    loss, accuracy = model.evaluate(val_ds, verbose=0) # Adjust metrics for multilabel
    print(f"Best Validation Loss: {loss:.4f}, Best Validation Accuracy: {accuracy:.4f}") # Adjust metrics for multilabel

    return {
        'model_architecture': model_class.__name__,
        'attributes': attr_config.__class__.__name__,
        'early_stopping': es_config.__class__.__name__,
        'best_val_loss': loss,
        'best_val_accuracy': accuracy, # Adjust metrics for multilabel
        'checkpoint_path': checkpoint_path
    }

# ## 3. Phase-Specific Configurations and Training Calls

experiment_results = []

# ### 3.1. Phase 1: Eyes Only
print("\n--- Configuring Phase 1: Eyes Only ---")
train_dir_phase1 = 'data/eyes_train'
val_dir_phase1 = 'data/eyes_val'
img_height_phase1 = 64
img_width_phase1 = 64
batch_size_phase1 = 32
epochs_phase1 = 10

train_ds_orig_phase1, val_ds_phase1, num_classes_phase1 = load_and_preprocess_data(
    train_dir_phase1, val_dir_phase1, img_height_phase1, img_width_phase1, batch_size_phase1
)
train_ds_aug_phase1 = prepare_augmented_dataset(train_ds_orig_phase1, batch_size_phase1)
attribute_classes_phase1 = [CNNModelAttributes1(img_height_phase1, img_width_phase1, 3, num_classes_phase1),
                            CNNModelAttributes2(img_height_phase1, img_width_phase1, 3, num_classes_phase1)]
model_classes_phase1 = [LeNet5, SimpleAlexNet]
es_configs_phase1 = [EarlyStoppingConfig1(), EarlyStoppingConfig2()]
compiler_phase1 = ModelCompiler()

run_phase_training(
    train_ds_orig_phase1,
    train_ds_aug_phase1,
    val_ds_phase1,
    model_classes_phase1,
    attribute_classes_phase1,
    es_configs_phase1,
    epochs_phase1,
    'checkpoints/phase1',
    compiler_phase1,
    'phase1',
    experiment_results
)

# ### 3.2. Phase 2: Headshots
print("\n--- Configuring Phase 2: Headshots ---")
train_dir_phase2 = 'data/headshots_train'
val_dir_phase2 = 'data/headshots_val'
img_height_phase2 = 128
img_width_phase2 = 128
batch_size_phase2 = 32
epochs_phase2 = 10

train_ds_orig_phase2, val_ds_phase2, num_classes_phase2 = load_and_preprocess_data(
    train_dir_phase2, val_dir_phase2, img_height_phase2, img_width_phase2, batch_size_phase2
)
train_ds_aug_phase2 = prepare_augmented_dataset(train_ds_orig_phase2, batch_size_phase2)
attribute_classes_phase2 = [CNNModelAttributes1(img_height_phase2, img_width_phase2, 3, num_classes_phase2),
                            CNNModelAttributes2(img_height_phase2, img_width_phase2, 3, num_classes_phase2)]
model_classes_phase2 = [LeNet5, SimpleAlexNet]
es_configs_phase2 = [EarlyStoppingConfig1(), EarlyStoppingConfig2()]
compiler_phase2 = ModelCompiler()

run_phase_training(
    train_ds_orig_phase2,
    train_ds_aug_phase2,
    val_ds_phase2,
    model_classes_phase2,
    attribute_classes_phase2,
    es_configs_phase2,
    epochs_phase2,
    'checkpoints/phase2',
    compiler_phase2,
    'phase2',
    experiment_results
)

# ### 3.3. Phase 3: Full Body
print("\n--- Configuring Phase 3: Full Body ---")
train_dir_phase3 = 'data/fullbody_train' # Adjust
val_dir_phase3 = 'data/fullbody_val'     # Adjust
img_height_phase3 = 224 # Adjust
img_width_phase3 = 224   # Adjust
batch_size_phase3 = 32  # Adjust
epochs_phase3 = 10     # Adjust

train_ds_orig_phase3, val_ds_phase3, num_classes_phase3 = load_and_preprocess_data(
    train_dir_phase3, val_dir_phase3, img_height_phase3, img_width_phase3, batch_size_phase3
)
train_ds_aug_phase3 = prepare_augmented_dataset(train_ds_orig_phase3, batch_size_phase3)
attribute_classes_phase3 = [CNNModelAttributes1(img_height_phase3, img_width_phase3, 3, num_classes_phase3),
                            CNNModelAttributes2(img_height_phase3, img_width_phase3, 3, num_classes_phase3)]
model_classes_phase3 = [LeNet5, SimpleAlexNet] # Adjust
es_configs_phase3 = [EarlyStoppingConfig1(), EarlyStoppingConfig2()] # Adjust
compiler_phase3 = ModelCompiler() # Adjust

run_phase_training(
    train_ds_orig_phase3,
    train_ds_aug_phase3,
    val_ds_phase3,
    model_classes_phase3,
    attribute_classes_phase3,
    es_configs_phase3,
    epochs_phase3,
    'checkpoints/phase3',
    compiler_phase3,
    'phase3',
    experiment_results
)

# ### 3.4. Phase 4: Multilabel / Multi Character
print("\n--- Configuring Phase 4: Multilabel / Multi Character ---")
train_dir_phase4 = 'data/multilabel_train' # Adjust
val_dir_phase4 = 'data/multilabel_val'     # Adjust
img_height_phase4 = 128
img_width_phase4 = 128
batch_size_phase4 = 32
epochs_phase4 = 15 # Adjust
label_mode_phase4 = 'binary' # Or None if you have a custom label format

train_ds_orig_phase4, val_ds_phase4, num_classes_phase4 = load_and_preprocess_data(
    train_dir_phase4, val_dir_phase4, img_height_phase4, img_width_phase4, batch_size_phase4, label_mode=label_mode_phase4
)
#train_ds_aug_phase4
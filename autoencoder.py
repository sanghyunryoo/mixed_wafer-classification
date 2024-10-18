from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import os
import math
from tensorflow.keras.utils import Sequence
from keras.callbacks import ModelCheckpoint
from tensorflow_addons.optimizers import AdamW
from tensorflow import keras
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, Dense, Flatten, Reshape, Lambda

# Define image dimensions
input_shape = (52, 52, 1)  # Binary image, so 1 channel
latent_dim = 64
size = 52
batch = 16
channel = 1

# Encoder
def build_ae():
    input_img = Input(shape=input_shape)

    # Flatten the image for the Dense layers
    x = Flatten()(input_img)

    # Encoder layers
    encoded = Dense(256, activation='gelu')(x)
    encoded = Dense(128, activation='gelu')(encoded)
    encoded = Dense(64, activation='gelu')(encoded)  # Bottleneck (compressed representation)

    # Decoder layers
    decoded = Dense(128, activation='gelu')(encoded)
    decoded = Dense(256, activation='gelu')(decoded)
    decoded = Dense(np.prod(input_shape), activation='sigmoid')(decoded)  # Output layer

    # Reshape back to image shape
    decoded = Reshape(input_shape)(decoded)

    # Build the autoencoder model
    autoencoder = Model(input_img, decoded)
    
    # Compile the model with a binary cross-entropy loss since the images are binary
    autoencoder.compile(AdamW(learning_rate=0.001, weight_decay=False),
                loss="binary_crossentropy",
                metrics=['accuracy'])
    return autoencoder

def build_conv_ae():
    input_img = Input(shape=(52, 52, 1))  # Update input shape for 2D image with 1 channel (grayscale)

    # Encoder: Convolutional Layers
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(input_img)  # 1st conv layer
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)  # 2nd conv layer
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)  # 3rd conv layer

    # Bottleneck (compressed representation)
    encoded = Conv2D(8, (3, 3), activation='relu', padding='same')(x)

    # Decoder: Deconvolutional (Transpose Convolution) Layers
    x = Conv2DTranspose(16, (3, 3), activation='relu', padding='same')(encoded)  # 1st deconv layer
    x = Conv2DTranspose(32, (3, 3), activation='relu', padding='same')(x)  # 2nd deconv layer
    x = Conv2DTranspose(64, (3, 3), activation='relu', padding='same')(x)  # 3rd deconv layer

    # Final decoder layer to return to original image shape (52, 52, 1)
    decoded = Conv2DTranspose(1, (3, 3), activation='sigmoid', padding='same')(x)

    # Build the autoencoder model
    autoencoder = Model(input_img, decoded)
    
    # Compile the model with AdamW optimizer and binary cross-entropy loss
    autoencoder.compile(optimizer=AdamW(learning_rate=0.001, weight_decay=False),
                        loss='binary_crossentropy',
                        metrics=['accuracy'])
    
    return autoencoder

def sampling(args):
    mu, log_var = args
    epsilon = K.random_normal(shape=(K.shape(mu)[0], latent_dim), mean=0., stddev=1.0)
    return mu + K.exp(0.5 * log_var) * epsilon  # Reparameterization trick

def build_encoder():
    input_img = Input(shape=input_shape)

    # Flatten the image
    x = Flatten()(input_img)

    # Encoder layers
    x = Dense(256, activation='relu')(x)
    x = Dense(128, activation='relu')(x)

    # Output mean and log variance for latent space
    mu = Dense(latent_dim)(x)
    log_var = Dense(latent_dim)(x)

    # Latent space sampling (reparameterization trick)
    z = Lambda(sampling, output_shape=(latent_dim,))([mu, log_var])

    # Encoder model
    encoder = Model(input_img, [mu, log_var, z])
    return encoder, mu, log_var

# Decoder network
def build_decoder():
    latent_inputs = Input(shape=(latent_dim,))

    # Decoder layers
    x = Dense(128, activation='relu', name="latent_space")(latent_inputs)
    x = Dense(256, activation='relu')(x)
    x = Dense(np.prod(input_shape), name='decoder_output', activation='sigmoid')(x)

    # Reshape back to the original image shape
    decoded = Reshape(input_shape)(x)

    # Decoder model
    decoder = Model(latent_inputs, decoded)
    return decoder

def vae_loss(input_img, reconstructed, mu, log_var):
    # Reconstruction loss (binary cross-entropy)
    reconstruction_loss = binary_crossentropy(K.flatten(input_img), K.flatten(reconstructed))
    reconstruction_loss *= np.prod(input_shape)

    # KL divergence loss
    kl_loss = 1 + log_var - K.square(mu) - K.exp(log_var)
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss *= -0.5

    # Total loss: reconstruction + KL divergence
    return K.mean(reconstruction_loss + kl_loss)

# Build the full VAE model
def build_vae(encoder, decoder, mu, log_var):
    # Input image
    input_img = Input(shape=input_shape)

    # Get the encoder's outputs
    mu, log_var, z = encoder(input_img)

    # Get the reconstructed image from the decoder
    reconstructed = decoder(z)

    # Define the VAE model
    vae = Model(input_img, reconstructed)
    vae.add_loss(vae_loss(input_img, reconstructed, mu, log_var))

    # Compile the VAE
    vae.compile(optimizer=AdamW(learning_rate=0.001, weight_decay=False))

    return vae


# Dummy data for training (replace with your actual data)
# Assuming your dataset is binary images of shape (52, 52)
train_data = np.load('data/mixedwm38/mixed/orig/mixed_train_data.npy').reshape(-1, size, size, channel)
val_data = np.load('data/mixedwm38/mixed/orig/mixed_val_data.npy').reshape(-1, size, size, channel)
test_data = np.load('data/mixedwm38/mixed/orig/mixed_test_data.npy').reshape(-1, size, size, channel)
test_label = np.load('data/mixedwm38/mixed/label/mixed_test_label.npy')  

# train_data = tf.image.per_image_standardization(train_data) * 0.23 + 0.49
# val_data = tf.image.per_image_standardization(val_data) * 0.23 + 0.49
# test_data = tf.image.per_image_standardization(test_data) * 0.23 + 0.49
    
# train_loader = Dataloader(train_data, batch, shuffle=True)
# val_loader = Dataloader(val_data, batch, shuffle=True)
# test_loader = Dataloader(test_data, batch, shuffle=True)        

# Train the autoencoder
print('Training -------------------------------------------------\n')
MODEL_SAVE_FOLDER_PATH = 'mixedwm38_test/cae/'

os.makedirs(MODEL_SAVE_FOLDER_PATH, exist_ok=True)                                                                                

model_path = MODEL_SAVE_FOLDER_PATH + 'best.h5'
    
cb_checkpoint = ModelCheckpoint(filepath=model_path, monitor='val_loss', # val_predictions_loss # val_loss
                                    verbose=1, save_best_only=True)
        
# Create the autoencoder
autoencoder = build_conv_ae()
autoencoder.fit(train_data, train_data, validation_data=(val_data, val_data), epochs=100, callbacks=[cb_checkpoint])

# Generate predictions (reconstructed images) on test data

autoencoder = keras.models.load_model(model_path)
# autoencoder = keras.models.load_model(model_path, custom_objects={'vae_loss': vae_loss})
sample_index = np.random.choice(len(test_data), size=5, replace=True)
sampled_data = test_data[sample_index]
sampled_label = test_label[sample_index]

reconstructed_imgs = autoencoder.predict(sampled_data)
binary_reconstructed_imgs = (reconstructed_imgs >= 0.5).astype(np.float32)  # Convert to 0 or 1

# Function to visualize the original and reconstructed images
def visualize_reconstruction(original, reconstructed, n=5):
    plt.figure(figsize=(10, 4))
    
    for i in range(n):
        # Original images
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(original[i].reshape(52, 52))
        plt.title("Original")
        plt.axis('off')
        
        # Reconstructed images
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(reconstructed[i].reshape(52, 52))
        plt.title("Reconstructed")
        plt.axis('off')
        print(sampled_label[i])
    plt.tight_layout()
    plt.show()

# Visualize the first 5 test images and their reconstructions
visualize_reconstruction(sampled_data, binary_reconstructed_imgs, n=5)

# # Define the dimension of the latent space (same as during training)
# latent_dim = 64

# # Function to generate and visualize new samples
# def generate_new_samples(decoder, n=5):
#     # Sample random latent vectors from a standard normal distribution
#     latent_samples = np.random.normal(size=(n, latent_dim))
    
#     # Generate new images from the decoder
#     generated_imgs = decoder.predict(latent_samples)

#     # Visualize the generated images
#     plt.figure(figsize=(10, 4))
#     for i in range(n):
#         ax = plt.subplot(1, n, i + 1)
#         plt.imshow(generated_imgs[i].reshape(52, 52), cmap='gray')  # Reshape based on your input image dimensions
#         plt.title(f"Sample {i + 1}")
#         plt.axis('off')
    
#     plt.tight_layout()
#     plt.show()

# # Assuming 'decoder' is already built and trained
# latent_inputs = Input(shape=(latent_dim,))  # Latent space input
# decoder = Model(inputs=autoencoder.get_layer('latent_space').input, 
#                 outputs=autoencoder.get_layer('decoder_output').output)

# generate_new_samples(decoder, n=5)
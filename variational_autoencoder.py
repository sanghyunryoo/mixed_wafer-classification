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


def sampling(args):
    mu, log_var = args
    epsilon = K.random_normal(shape=(K.shape(mu)[0], latent_dim), mean=0., stddev=1.0)
    return mu + K.exp(0.5 * log_var) * epsilon  # Reparameterization trick

def build_encoder():
    input_img = Input(shape=input_shape)

    # Flatten the image
    x = Flatten()(input_img)

    # Encoder layers
    x = Dense(256, activation='gelu')(x)
    x = Dense(128, activation='gelu')(x)
    
    # Output mean and log variance for latent space
    mu = Dense(latent_dim)(x)
    log_var = Dense(latent_dim)(x)

    # Latent space sampling (reparameterization trick)
    z = Lambda(sampling, output_shape=(latent_dim,))([mu, log_var])

    # Encoder model
    encoder = Model(input_img, [mu, log_var, z])
    return encoder, mu, log_var

def build_conv_encoder():
    input_img = Input(shape=input_shape)

    # Encoder: Convolutional layers
    x = Conv2D(32, (3, 3), activation='gelu', padding='same')(input_img)  # 1st conv layer
    x = Conv2D(64, (3, 3), activation='gelu', padding='same')(x)  # 2nd conv layer
    x = Conv2D(128, (3, 3), activation='gelu', padding='same')(x)  # 3rd conv layer
    x = Flatten()(x)  # Flatten the output

    # Latent space: Mean and log variance for the Gaussian distribution
    mu = Dense(latent_dim)(x)
    log_var = Dense(latent_dim)(x)

    # Latent space sampling (reparameterization trick)
    z = Lambda(sampling, output_shape=(latent_dim,))([mu, log_var])

    # Encoder model
    encoder = Model(input_img, [mu, log_var, z])
    return encoder, mu, log_var

def build_hybrid_encoder():
    input_img = Input(shape=input_shape)

    # Encoder: Convolutional layers for feature extraction
    x = Conv2D(32, (3, 3), activation='gelu', padding='same')(input_img)  # 1st conv layer
    x = Conv2D(64, (3, 3), activation='gelu', padding='same')(x)  # 2nd conv layer
    x = Conv2D(128, (3, 3), activation='gelu', padding='same')(x)  # 3rd conv layer
    x = Flatten()(x)  # Flatten the output for dense layers

    # Dense layers for latent space representation
    x = Dense(256, activation='gelu')(x)
    mu = Dense(latent_dim)(x)  # Mean of the latent space
    log_var = Dense(latent_dim)(x)  # Log variance of the latent space

    # Latent space sampling (reparameterization trick)
    z = Lambda(sampling, output_shape=(latent_dim,))([mu, log_var])

    # Encoder model
    encoder = Model(input_img, [mu, log_var, z])
    return encoder, mu, log_var

# Decoder network
def build_decoder():
    latent_inputs = Input(shape=(latent_dim,))

    # Decoder layers
    x = Dense(128, activation='gelu', name="latent_space")(latent_inputs)
    x = Dense(256, activation='gelu')(x)
    x = Dense(np.prod(input_shape), name='decoder_output', activation='sigmoid')(x)

    # Reshape back to the original image shape
    decoded = Reshape(input_shape)(x)

    # Decoder model
    decoder = Model(latent_inputs, decoded)
    return decoder

def build_conv_decoder():
    latent_inputs = Input(shape=(latent_dim,))

    # Decoder: Fully connected layers followed by transposed convolution layers
    x = Dense(128 * 52 * 52, activation='gelu')(latent_inputs)  # Fully connected layer
    x = Reshape((52, 52, 128))(x)  # Reshape into feature map
    
    x = Conv2DTranspose(128, (3, 3), activation='gelu', padding='same')(x)  # 1st deconv layer
    x = Conv2DTranspose(64, (3, 3), activation='gelu', padding='same')(x)  # 2nd deconv layer
    x = Conv2DTranspose(32, (3, 3), activation='gelu', padding='same')(x)  # 3rd deconv layer

    # Final output layer with sigmoid activation to reconstruct the image
    decoded = Conv2DTranspose(1, (3, 3), activation='sigmoid', padding='same')(x)

    # Decoder model
    decoder = Model(latent_inputs, decoded)
    return decoder

def build_hybrid_decoder():
    latent_inputs = Input(shape=(latent_dim,))

    # Dense layer to upscale the latent vector
    x = Dense(128 * 52 * 52, activation='gelu')(latent_inputs)  # Upscale to match the feature map size
    x = Reshape((52, 52, 128))(x)  # Reshape to feature map

    # Decoder: Transposed convolutional layers to reconstruct the image
    x = Conv2DTranspose(128, (3, 3), activation='gelu', padding='same')(x)  # 1st deconv layer
    x = Conv2DTranspose(64, (3, 3), activation='gelu', padding='same')(x)  # 2nd deconv layer
    x = Conv2DTranspose(32, (3, 3), activation='gelu', padding='same')(x)  # 3rd deconv layer

    # Final output layer with sigmoid activation to reconstruct the image
    decoded = Conv2DTranspose(1, (3, 3), activation='sigmoid', padding='same')(x)

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
    vae.compile(optimizer=AdamW(learning_rate=0.0003, weight_decay=False))

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
    
    
# Train the autoencoder
print('Training -------------------------------------------------\n')
MODEL_SAVE_FOLDER_PATH = 'mixedwm38_test/cvae/'

os.makedirs(MODEL_SAVE_FOLDER_PATH, exist_ok=True)                                                                                

model_path = MODEL_SAVE_FOLDER_PATH + 'best.h5'
    
cb_checkpoint = ModelCheckpoint(filepath=model_path, monitor='val_loss', # val_predictions_loss # val_loss
                                    verbose=1, save_best_only=True)
        

# encoder, mu, log_var = build_conv_encoder()
# decoder = build_conv_decoder()
# autoencoder = build_vae(encoder, decoder, mu, log_var)
# autoencoder.fit(train_data, 
#                 epochs=100, 
#                 validation_data=(val_data, None),  # validation_data is only input data as well
#                 callbacks=[cb_checkpoint])



# Generate predictions (reconstructed images) on test data
ae = keras.models.load_model('mixedwm38_test/ae/best.h5')
cae = keras.models.load_model('mixedwm38_test/cae/best.h5')

vae = keras.models.load_model('mixedwm38_test/vae/best.h5')
cvae = keras.models.load_model('mixedwm38_test/cvae/best.h5')
hvae = keras.models.load_model('mixedwm38_test/hvae/best.h5')

# autoencoder = keras.models.load_model(model_path, custom_objects={'vae_loss': vae_loss})
sample_index = np.random.choice(len(test_data), size=5, replace=True)
sampled_data = test_data[sample_index]
sampled_label = test_label[sample_index]

ae_reconstructed_imgs = ae.predict(sampled_data)
ae_binary_reconstructed_imgs = (ae_reconstructed_imgs >= 0.5).astype(np.float32) 

cae_reconstructed_imgs = cae.predict(sampled_data)
cae_binary_reconstructed_imgs = (cae_reconstructed_imgs >= 0.5).astype(np.float32)

vae_reconstructed_imgs = vae.predict(sampled_data)
vae_binary_reconstructed_imgs = (vae_reconstructed_imgs >= 0.5).astype(np.float32) 

cvae_reconstructed_imgs = cvae.predict(sampled_data)
cvae_binary_reconstructed_imgs = (cvae_reconstructed_imgs >= 0.5).astype(np.float32)

hvae_reconstructed_imgs = hvae.predict(sampled_data)
hvae_binary_reconstructed_imgs = (hvae_reconstructed_imgs >= 0.5).astype(np.float32)

# Function to visualize the original and reconstructed images
def visualize_reconstruction(original, reconstructed1, reconstructed2, reconstructed3, reconstructed4, reconstructed5, n=5):
    plt.figure(figsize=(10, 4))
    
    for i in range(n):
        # Original images
        ax = plt.subplot(5, n, i + 1)
        plt.imshow(original[i].reshape(52, 52))
        # plt.title("Original")
        plt.axis('off')     
        
        ax = plt.subplot(5, n, i + 1 + 1*n)
        plt.imshow(reconstructed1[i].reshape(52, 52))
        # plt.title("Reconstructed")
        plt.axis('off')
                
        ax = plt.subplot(5, n, i + 1 + 2*n)
        plt.imshow(reconstructed3[i].reshape(52, 52))
        # plt.title("Reconstructed")
        plt.axis('off')
        
        ax = plt.subplot(5, n, i + 1 + 3*n)
        plt.imshow(reconstructed4[i].reshape(52, 52))
        # plt.title("Reconstructed")
        plt.axis('off')

        ax = plt.subplot(5, n, i + 1 + 4*n)
        plt.imshow(reconstructed5[i].reshape(52, 52))
        # plt.title("Reconstructed")
        plt.axis('off')
                
        print(sampled_label[i])
    plt.tight_layout()
    plt.show()

# Visualize the first 5 test images and their reconstructions
visualize_reconstruction(sampled_data, ae_reconstructed_imgs, cae_reconstructed_imgs, vae_reconstructed_imgs, cvae_reconstructed_imgs, hvae_reconstructed_imgs, n=5)
visualize_reconstruction(sampled_data, ae_binary_reconstructed_imgs, cae_binary_reconstructed_imgs, vae_binary_reconstructed_imgs, cvae_binary_reconstructed_imgs, hvae_binary_reconstructed_imgs, n=5)

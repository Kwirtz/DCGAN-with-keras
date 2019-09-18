import time
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

from keras.models import Sequential
from keras.layers import Conv2D, Conv2DTranspose, Reshape
from keras.layers import Flatten, BatchNormalization, Dense, Activation
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
import keras.backend as K

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # not being spammed by tf warnings

def wasserstein_loss(y_true,y_pred):
    return K.mean(y_true*y_pred)

def load_dataset(dataset_path, batch_size, image_shape):
    dataset_generator = ImageDataGenerator()
    dataset_generator = dataset_generator.flow_from_directory( # Note that flow_from_directory require a folder inside the dataset_path
        dataset_path, target_size=(image_shape[0], image_shape[1]),
        batch_size=batch_size,
        class_mode=None) #Unsupervised so we do not care about class

    return dataset_generator

# Displays a figure of the generated images and saves them in as .png image
def save_generated_images(generated_images, epoch):

    plt.figure(figsize=(5, 5))  # define the size of the whole plot

    for i in range(4): # iterate through the number of image we want 4 here just to check mode collapse and everything is fine
        ax = plt.subplot(2,2,i+1) # 2 rows, 2 cols
        image = generated_images[i, :, :, :] # select the i image
        image += 1 
        image *= 127.5 # denormalize (to train we normalize the data so to create the input we need to do decode the data)
        im = ax.imshow(image.astype(np.uint8)) # convert the image in a readable form for matplotlib (float32 [0,1] RGB)
        plt.axis('off') # dont show the axis

    plt.tight_layout() 
    save_name = 'generated_images/generatedSamples_epoch' + str(
        epoch + 1) + '.png' 

    plt.savefig(save_name, bbox_inches='tight', pad_inches=0) # every given number of epoch, save this plot to see the evolution at the end
    plt.pause(0.0000000001)
    plt.show()

def save_loss(batches, adversarial_loss, discriminator_loss, epoch):
        plt.figure(1)
        plt.plot(batches, adversarial_loss, color='green',
                 label='Generator Loss')
        plt.plot(batches, discriminator_loss, color='blue',
                 label='Discriminator Loss')
        plt.title("DCGAN Train")
        plt.xlabel("Batch Iteration")
        plt.ylabel("Loss")
        if epoch == 0:
            plt.legend()
        plt.pause(0.0000000001)
        plt.show()
        plt.savefig('trainingLossPlot.png')

# Creates the discriminator model. fake vs real, the image shape does not really matter in the discriminator
# if you have any doubt with the dimensions we added the summary of the model so check that when running the code (same for generator)
# usually a bloc looks like this conv_layer-BatchNormalization-activation(LeakyReLU)-repeat
def construct_discriminator(image_shape):

    discriminator = Sequential()
    discriminator.add(Conv2D(filters=64, kernel_size=(5, 5),
                             strides=(2, 2), padding='same',
                             data_format='channels_last',
                             kernel_initializer='glorot_uniform',
                             input_shape=(image_shape)))
    discriminator.add(LeakyReLU(0.2))

    discriminator.add(Conv2D(filters=128, kernel_size=(5, 5),
                             strides=(2, 2), padding='same',
                             data_format='channels_last',
                             kernel_initializer='glorot_uniform'))
    discriminator.add(BatchNormalization(momentum=0.5))
    discriminator.add(LeakyReLU(0.2))

    discriminator.add(Conv2D(filters=256, kernel_size=(5, 5),
                             strides=(2, 2), padding='same',
                             data_format='channels_last',
                             kernel_initializer='glorot_uniform'))
    discriminator.add(BatchNormalization(momentum=0.5))
    discriminator.add(LeakyReLU(0.2))

    discriminator.add(Flatten())
    discriminator.add(Dense(1,activation="sigmoid"))
    discriminator.summary()

    optimizer = Adam(lr=0.0002, beta_1=0.5)
    discriminator.compile(loss="binary_crossentropy", # If you use binary cross-entropy (fake vs real) add a sigmoid activation at the end
                          optimizer=optimizer,
                          metrics=None)

    return discriminator


# Create the generator model. Here the input shape (units=16*16*256) is really important since you want to have the same shape as a real image
# even though you started with 100 value, randomly generated.
# In this case every transposed convolution double the size of the image. You could also use upsampling2D but transposed convolution worked better with my case

def construct_generator():

    generator = Sequential()

    generator.add(Dense(units=16 * 16 * 256, # can be considered as creating a 16*16 image with 256 filters
                        kernel_initializer='glorot_uniform',
                        input_shape=(1, 1, 100)))
    generator.add(Reshape(target_shape=(16, 16, 256)))
    generator.add(BatchNormalization(momentum=0.5))
    generator.add(Activation('relu'))


    generator.add(Conv2DTranspose(filters=128, kernel_size=(5, 5),
                                  strides=(2, 2), padding='same',
                                  data_format='channels_last',
                                  kernel_initializer='glorot_uniform')) # becomes (32*32*128)
    generator.add(BatchNormalization(momentum=0.5))
    generator.add(Activation('relu'))

    generator.add(Conv2DTranspose(filters=64, kernel_size=(5, 5),
                                  strides=(2, 2), padding='same',
                                  data_format='channels_last',
                                  kernel_initializer='glorot_uniform')) # becomes(64*64*64)
    generator.add(BatchNormalization(momentum=0.5))
    generator.add(Activation('relu'))

    generator.add(Conv2DTranspose(filters=3, kernel_size=(5, 5),
                                  strides=(2, 2), padding='same', 
                                  data_format='channels_last',
                                  kernel_initializer='glorot_uniform')) # becomes (128*128*3) just like the image shape we use
    generator.add(Activation('tanh'))
    generator.summary()
    return generator



# Main train function

def train_dcgan(batch_size, epochs, image_shape, dataset_path, n_samples, previous=False):

    #Initialize the models with the functions previously seen

    generator = construct_generator()
    discriminator = construct_discriminator(image_shape)

    # If you already trained this model once and want to keep training you can use previous weight
    # The saved weights depend on the number of epochs, make sure to look at your names in the checkpoint folder

    if previous == True:
        print("Previous model used")
        generator.load_weights('checkpoint/gen_200_scaled_images.h5')
        discriminator.load_weights('checkpoint/dis_200_scaled_images.h5')
    else:
        print("New model created")

    # Combining the generation and discriminator (for fake input you need to go through both)

    gan = Sequential()
    
    # The objective function of a gan is a minmax function and is solve by alternatively fix the discriminator and optimize the generator
    # and then do the inverse

    discriminator.trainable = False
    gan.add(generator)
    gan.add(discriminator)

    optimizer = Adam(lr=0.00015, beta_1=0.5)
    gan.compile(loss="binary_crossentropy", optimizer=optimizer,
                metrics=None)

    # Create the dataset
    
    dataset_generator = load_dataset(dataset_path, batch_size, image_shape)

    # Number of iterations (reminder number_of_batches*batch_size = 1 epoch)
    number_of_batches = int(n_samples / batch_size)

    # Variables used for the loss plot
    adversarial_loss = np.empty(shape=1)
    discriminator_loss = np.empty(shape=1)
    batches = np.empty(shape=1)

    # Dynamically change the plot
    plt.ion()

    current_batch = 0

    # Let's train the DCGAN for n epochs
    for epoch in range(epochs):

        print("Epoch " + str(epoch+1) + "/" + str(epochs) + " :")

        for batch_number in range(number_of_batches):

            start_time = time.time()

            # Get the current batch and normalize the images between -1 and 1
            real_images = dataset_generator.next()
            real_images /= 127.5
            real_images -= 1

            # The last batch is smaller than the other ones, so we need to
            # take that into account

            current_batch_size = real_images.shape[0]

            # Generate noise with a latent dim of 100
            
            noise = np.random.normal(0, 1, size=(current_batch_size,) + (1, 1, 100))

            # Generate images
            generated_images = generator.predict(noise)

            # Add some noise to the labels that will be
            # fed to the discriminator
            real_y = (np.ones(current_batch_size) -
                      np.random.random_sample(current_batch_size) * 0.2)
            fake_y = np.random.random_sample(current_batch_size) * 0.2

            # Let's train the discriminator
            discriminator.trainable = True

            d_loss = discriminator.train_on_batch(real_images, real_y)
            d_loss += discriminator.train_on_batch(generated_images, fake_y)

            discriminator_loss = np.append(discriminator_loss, d_loss)

            # Now it's time to train the generator
            discriminator.trainable = False

            noise = np.random.normal(0, 1,
                                     size=(current_batch_size * 2,) +
                                     (1, 1, 100))

            # We try to mislead the discriminator by giving the opposite labels
            fake_y = (np.ones(current_batch_size * 2) -
                      np.random.random_sample(current_batch_size * 2) * 0.2)

            g_loss = gan.train_on_batch(noise, fake_y)
            adversarial_loss = np.append(adversarial_loss, g_loss)
            batches = np.append(batches, current_batch)


            time_elapsed = time.time() - start_time

            # Display and plot the results
            print("     Batch " + str(batch_number + 1) + "/" +
                  str(number_of_batches) +
                  " generator loss | discriminator loss : " +
                  str(g_loss) + " | " + str(d_loss) + ' - batch took ' +
                  str(time_elapsed) + ' s.')

            current_batch += 1
        


        # Each epoch update the loss graphs and save the generated img after 25 epochs
        save_loss(batches,adversarial_loss,discriminator_loss,epoch)
        if epoch % 25 == 0:
            save_generated_images(generated_images, epoch)    
            #save model, weights for retraining purposes and model for generation purposes 
            generator.save_weights('checkpoint/gen_'+ str(epochs) +'.h5')
            discriminator.save_weights('checkpoint/dis_'+ str(epochs) +'.h5')
            generator.save('checkpoint/gen_model_'+ str(epochs) +'.h5')


def main():
    dataset_path = './resizedData/' # the folder containing the folder with the images (loading with keras)
    batch_size = 32 # Depends on memory
    image_shape = (128, 128, 3) # rows,cols,channels
    epochs = 200
    n_samples = 718 # Total number of samples
    train_dcgan(batch_size, epochs, image_shape, dataset_path, n_samples, previous = False) # Previous = True if loading weights

if __name__ == "__main__":
  main()
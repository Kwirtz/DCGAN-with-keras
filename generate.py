import matplotlib.pyplot as plt
import numpy as np
import math
from keras.models import load_model


generator = load_model('checkpoint/gen_model_200.h5')



def generate(num_images):

    noise = np.random.normal(0.0, 1.0,size=(num_images,) + (1, 1, 100))
    generated_images = generator.predict(noise)
    dim = math.ceil(math.sqrt(num_images))
    fig = plt.figure(figsize=(5,5))
    for i in range(num_images):
        ax = plt.subplot(dim,dim,i+1)
        image = generated_images[i, :, :, :]
        image += 1
        image *= 127.5
        im = ax.imshow(image.astype(np.uint8))
        plt.axis('off')
    plt.tight_layout()
    save_name = 'generated.png' 

    plt.savefig(save_name, bbox_inches='tight', pad_inches=0)
    plt.show()
    
generate(7)




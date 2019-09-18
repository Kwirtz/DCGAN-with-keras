# DCGAN-with-keras

<p> Deep convolutional neural network on pokemons using keras functional API.</p>



<p>First download the data from https://github.com/PokeAPI/sprites or run</p>

```
git clone https://github.com/PokeAPI/sprites.git
```

<p>Get the requirements (You can change tensorflow to tensorflow-gpu if you wish)</p>

```
pip install -r requirements.txt
```

<p> Run resize.py to homogenize the shape of data (Note that you could also use the functionality of keras to resize) </p>

```
python resize.py
```

<p>Once all this is done you shoulda have a resizedData folder and you can proceed to run dcgan.py</p>

```
python dcgan.py
```

<p> The results appear in generated_images and you can generate a new image (generated.png) using generate.py </p>

<p> Results : </p>
<img src="https://raw.githubusercontent.com/Kwirtz/DCGAN-with-keras/master/generated_images/generatedSamples_epoch176.png" width="400" height="200" />

<p> Ressources i used to create this repository: </p>

-https://github.com/eriklindernoren/Keras-GAN <br>
-https://github.com/keras-team/keras-contrib/blob/master/examples/improved_wgan.py <br>
-https://github.com/jacobgil/keras-dcgan/blob/master/dcgan.py <br>
-https://github.com/Neerajj9/DCGAN-Keras <br>
-https://github.com/davidreiman/mnist-wgan <br>


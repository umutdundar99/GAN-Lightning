# GAN-Lightning
Basic to Advanced Implementation of Generative Adversarial Networks module,  powered by PyTorch Lightning

## Dependencies
  -  `PyTorch Lightning`
  -  `PyTorch`
  -   `OpenCV`
  -   `Numpy`
  -   `Hydra`
  -    `Albumentation`


## How To Run

### Creating a Virtual Environment

    python3 -m venv virtual_env

### Activate the Enviorenment

    source path/to/virtual_env/bin/activate

### Download the Dependencies

    pip install -r requirements.txt
> Note: Editable package install option will be added soon.

### Run the Module
Run the gan_lightning python module.

    python3 -m gan_lightning

## 1- MNIST DATASET

### 1.1 SimpleGAN
  SimpleGAN consists of a SimpleGenerator and SimpleDiscriminator which contain linear layers.

  ![generated_images](https://github.com/umutdundar99/GAN-Lightning/assets/77459948/f363074b-4520-43da-8f3c-8dd0bb396955)

  Here are the results of the SimpleGAN

### 1.2. SimpleGANv2

  SimpleGANv2 has not been developed yet, but it will consist of Convolutional Neural Networks.
  The results will be updated.

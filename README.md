# Data-efficient GANs with Adaptive Discriminator Augmentation

Official tutorial: [click here](https://keras.io/examples/generative/gan_ada/).

Although the official Keras tutorial is very detailed, there will be a download error of tensorflow_data when I run the relevant program. This situation is very distressing. So I clicked [the download link](https://drive.google.com/uc?export=download&id=1hbzc_P1FuxMkcabkgn9ZKinBwW683j45) of the raw [caltech_birds2011](http://www.vision.caltech.edu/visipedia/CUB-200.html) dataset and customized the dataset processing file [cub_data_processor.py](./cub_data_processor.py).

## 1. Data preprocessing

Download the [CUB_200_2011.tgz](https://drive.google.com/uc?export=download&id=1hbzc_P1FuxMkcabkgn9ZKinBwW683j45) file and move it to the current project directory.

Then unzip it as follows.

```bash
tar -zxvf CUB_200_2011.tgz
```

## 2. Run the training program

 Run the training program as follows.

```bash
python gan_ada.py
```
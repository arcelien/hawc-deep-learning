# HAWC Deep Learning

Deep learning models on HAWC simulation dataset

## Prerequisites
- A HAWC simulation dataset should be downloaded and placed in main directory, `$HAWC`
- XCDF from https://github.com/jimbraun/XCDF should be complied
    - contents of the compiled `lib/` folder, `libxcdf.so` and `xcdf/`, should be placed in the main directory
    - If getting errors with `libxcdf.so`, add it's location to `$LD_LIBRARY_PATH`
    - Make sure you're using python 2 and cython 2 with XCDF
- Install python packages with `pip install -r requirements.txt` (preferably in a Conda or virtualenv environment)

## Overview
### Data Processing
`parse_hawc.py` reads in data from the `$HAWC` folder and generates the training and testing datasets for our experiments.

To generate the dataset, run 
``` bash
python parse_hawc.py $HAWC
``` 
The dataset will be stored in `$HAWC/data`

For PMT data from grid hit events, we use a mapping of PMTs to specific coordinates of a 40x40 grid, defined in `squaremapping.py`

Note: because there are more coordinates (1600) than PMTs (<1200), some coordinates will always have 0 value if they don't correspond to a PMT.

### Plotting
`plot.py` contains many visualization functions. It can visualize data from XCDF files (called in `parse_hawc.py`).

We can visualize the actual structure of the HAWC grid of PMTs. It's clear that some data is unintialized and we clean it during processing.

<img src="./plots/pmt_vis_1.png" width="400px"/>
<img src="./plots/pmt_vis_2.png" width="400px"/>

Here are some visualizations of data from our dataset:

<img src="./plots/ground_truth/ground_truth_mapping_gamma_log.png" width="600px"/>
<img src="./plots/ground_truth/ground_truth_mapping_gamma_pmts.png" width="600px"/>

## Deep learning Models and Experiments
### What is a GAN? A Physicists Primer Into Generative Models
GAN is short for Generative Adversarial Neural Network, and it is part of a family of deep learning models called "Generative Models".  A GAN is composed of 3 parts; the generator G, the discriminator D, and the expert sample X.  Traditionally, the generator and discriminator are neural networks, while the expert sample is a sample of the distribution you are trying to mimic.  The generator network takes in a vector of random noise z (which it needs as an entropy source), and produces an output that has the same form as that of the expert sample, G(z).  The Discriminator takes in a sample (either from the generator, G(z), or the expert sample x), and returns the probability that the sample is a forgery from the generator.  The model parameters of the generator are tuned to minimize D(G(z)) while the parameters of the generator are turned to maximize D(G(z)) and minimize D(x).  The end result is a mathematical arms race between G and D (they are "adversaries"), which slowly moves the distribution G(z) closer and closer to the expert sample X.

**diagram here**

This is a fundamentally different approach to distribution sampling compared that employed by physicists, as no knowledge of the mathematical form of the expert distribution is needed.  The only requirement is that the expert sample is "large enough" to express the distribution.  What is and isn't "large enough" is a question that is the subject of ongoing research, so developing a functional GAN model requires some amount of numerical experimental work to be done.

A question that a physicist might ask is this; "Why are GANs useful?  How can a GAN help us learn something about physics?".  While it is true that a GAN can be expressed mathematically, the expression is not useful to a physicist.  Such an expression would take the form of something akin to a Fourier series, which doesn't nescissarily tell you anything useful about what's actually being described by the model.  To a physicist, knowing that some distribution is Gaussian or Poissonian is much more useful that being able to draw samples from it.  So why use GANs at all?

The key is this;  physical simulations can be thought of as generative models
### 1D Distribution Generation with Non-Conditional GANs
Using GANs we can feed in a noise vector into a generator and get a output vector where each element is a sample from some distribution. 

To run param-gen/parameterGAN.py, run gen_gamma_params("/path/to/gamma") in parse_hawc and specifcy paramters to collect in the function. Then run in `param-gen/`: 

```bash
python parameterGAN.py
```
Histograms for the generated and actual distributions will be written to the paramGANplots folder in the same directory. An example of a generated histogram is shown below.

<img src="./plots/1DGAN/1Dhist.png" width="600px"/>

### 1D Distribution Generation with Conditional GANs
### 2D Distribution Generation with WGANs
### Generative Model with Pixel-cnn
We can use a pixel-cnn model to generate very realistic PMT grid hit data.

To run pixelcnn on the 40x40 images generated from above, run
```bash
cd pixel-cnn
sh scripts/train.sh
```
Checkpoints and output from pixelcnn will be located in `$HAWC/saves`, which can then be visualized with

```bash
python plot.py [epoch number of checkpoint]
```
Here is an example of generated samples from pixel-cnn. From inspection, it seems as if the pixel-cnn model learns to generate a distribution of samples that is representative of the varying sparsity between hits, and the smooth falloff of charge from a specific point indicative of gamma data.

<img src="./plots/pixelcnn/pixelcnn_pmt_hit_logcharge_40x40.png" width="600px"/>
<img src="./plots/pixelcnn/pixelcnn_pmt_hits_logcharge_pmts.png" width="600px"/>

# HAWC Deep Learning

Deep learning models on HAWC simulation dataset

## Install conda if necessary
```shell
wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
sh ./Miniconda3-latest-Linux-x86_64.sh -b -p
export PATH="$HOME/miniconda/bin:$PATH"

conda create --name hawc python=2.7
source activate hawc
conda install cython matplotlib numpy scipy imageio pytorch torchvision cuda90 -c pytorch
pip install tensorflow-gpu==1.8.0
```

## Prerequisites
- A HAWC simulation dataset should be downloaded and placed in `$HAWC`
- XCDF from https://github.com/jimbraun/XCDF should be complied to `$XCDF`
    - contents of the compiled `$XCDF/lib/` folder should be placed in the root directory, or link in `$LD_LIBRARY_PATH`
    - Make sure you're using python 2 and cython 2 with XCDF

To download this repository, run 
```shell
git clone --recurse-submodules https://github.com/arcelien/hawc-deep-learning.git
cd hawc-deep-learning
```

## Overview
### Data Processing
`parse_hawc.py` reads in data from the `$HAWC` folder and generates the training and testing datasets for our experiments.

To generate the dataset, run 
```shell
# generate the coordinates of the PMT by reading a XCDF file (for visualization later)
python parse_hawc.py --hawc-dir $HAWC --save-dir [dir to save data in] --gen layout
# generate the specified data files
python parse_hawc.py --hawc-dir $HAWC --save-dir [dir to save data in] --gen [one-channel-map or two-channel-map or one-dim]
``` 

For PMT data from grid hit events, we use a mapping of PMTs to specific coordinates of a 40x40 grid, defined in `squaremapping.py`

Note: because there are more coordinates (1600) than PMTs (<1200), some coordinates will always have 0 value if they don't correspond to a PMT.

### Plotting
`plot.py` contains many visualization functions. It can visualize data from XCDF files (called in `parse_hawc.py`).

We can visualize the actual structure of the HAWC grid of PMTs. It's clear that some data is unintialized and we clean it during processing.

<img src="./plots/pmt_vis_1.png" width="400px"/> <img src="./plots/pmt_vis_2.png" width="400px"/>

Here are some visualizations of data from our dataset:

<img src="./plots/ground_truth/ground_truth_mapping_gamma_log.png" width="600px"/>
<img src="./plots/ground_truth/ground_truth_mapping_gamma_pmts.png" width="600px"/>

## Deep learning Models and Experiments

### 1D Distribution Generation with Non-Conditional GANs
This GAN is a simple MLP (Multi Layer Perceptron) model that has been trained exclusively on the gamma ray simulation.  It accepts a vector of random draws from an 8D standard spherical gaussian.  The output of this model is  rec.logNPE, log(rec.nHit), rec.nTankHit, rec.zenith, rec.azimuth, rec.coreX, rec.coreY, and rec.CxPE40.

To run param-gen/parameterGAN.py, run gen_gamma_params("/path/to/gamma") in parse_hawc and specifcy paramters to collect in the function. Then run in `param-gen/`: 

```bash
python parameterGAN.py
```
Histograms for the generated and actual distributions will be written to the paramGANplots folder in the same directory. An example of a generated histogram is shown below.

<img src="./plots/1DGAN/1Dhist.png" width="600px"/>

This model trains to near completion in less than an hour on a GeForce GTX 1080 Ti.

### 1D Distribution Generation with Conditional GANs
The 1D parameter GAN can be modified slightly, allowing for conditional inputs.  Along with the 8D entropy vector, another set of input parameters can be appended.  In this case, these parameters are log(SimEvent.energyTrue), SimEvent.thetaTrue, and SimEvent.phiTrue.  These values are also included as an input to the discriminator.  During training, the GAN samples the input params from the real simulation, and generates an output.  The sampled params and the generated output are passed to the discriminator.  The output of this model is as before; rec.logNPE, log(rec.nHit), rec.nTankHit, rec.zenith, rec.azimuth, rec.coreX, rec.coreY, and rec.CxPE40.  

To run this model, you should ... **to be added**

The effects of the conditional inputs can be seen in the following tests.  First, we left the conditional variables free and sampled the inputs from uniform distributions.  This shows how much variation the generative model can express.

<img src="./plots/1DGAN/1dgan_uniform labels.png" width="600px"/>

We also passed input values from the simulation directly to the generative model, which illustrates just how well the model is able to capture the source distribution.

<img src="./plots/1DGAN/1dgan_reallabels.png" width="600px"/>

Differences here, especially in the distributions with hard cutoffs, comes from a combination of using only gaussians as the input entropy source, and training time.  

This model was trained to near completion in less than an hour on a GTX 1080 Ti

### 2D Distribution Generation with WGANs
WIP

### Generative Model with Pixel-cnn
We can use a pixel-cnn (https://arxiv.org/abs/1601.06759) model to generate very realistic PMT grid hit data. Because we can map each simulation event to a 40x40 array, we can use a generative model like PixelCNN to sample images corresponding to events.

To run PixelCNN on the 40x40 images generated from above, run
```shell
cd pixel-cnn
python train.py --save_dir [dir to save pixel-cnn output] --data_dir [where processed data was saved to] --save_interval 3 --dataset [hawc1 or hawc2] (--nosample if no matplotlib)
```
Checkpoints and output from PixelCNN will be located in `$HAWC/saves`, which can then be visualized with

```shell
python plot.py --num [epoch number of checkpoint] --chs [1 or 2] --data-path [dir of processed HAWC data (layout.npy)] --save-path [dir of pixel-cnn output]
```
Here is an example of generated samples from pixel-cnn. From inspection, it seems as if the pixel-cnn model learns to generate a distribution of samples that is representative of the varying sparsity between hits, and the smooth falloff of charge from a specific point indicative of gamma data.

<img src="./plots/pixelcnn/pixelcnn_pmt_hit_logcharge_40x40.png" width="800px"/>
<img src="./plots/pixelcnn/pixelcnn_pmt_hits_logcharge_pmts.png" width="800px"/>

### Two channel generation
We then extend our pixel-cnn model to generate a simulation event including both the charge and hit time recorded at each PMT.

We ran the model on a NVIDIA Tesla V100 16GB GPU; training takes 2170 seconds (36 minutes) per epoch (entire set of gamma images). 

Because the model converges and can generate very realistic iamges after only 3-6 epochs of training, it takes less than two hours to train this model. A small additional improvement in the metric (bits per dimension) can be realized after training for a full day, but images look equally realistic to the human eye.

Unfortunately, PixelCNN is not a fast model for sampling, and it takes 272 seconds to generate a batch of 16 images (17 seconds per image)

Here is a visualization where the first channel is log charge, and the second is hit time (normalized).
<img src="./plots/pixelcnn/pixelcnn_pmt_hit_two_dim.png" width="1000px"/>

### Fast PixelCNN Sampling
We can apply a method to cache results while generating images (https://github.com/PrajitR/fast-pixel-cnn) to speed up the two channel image sampling process significantly. This new sampling technique gets an initial, signficiant speedup over the original while using the same batch size, and it scales with batch scale in near constant time - increasing the batch size does not increase sampling time.

On a GTX 1080 ti, it took less than 82 seconds to generate a batch of 512 images (0.16 seconds / image). Compared to the 272 seconds to generate a batch of 16 on a Teslta V100 from before (17 seconds / image), this generation technique is over 100 times faster.

The number of images per batch is limited by the ammount of GPU memory, and we ran out of memory when trying a batch of size 1024. On a GPU with 32 GBs of memory (new Telsa V100), it shoould be able to generate a batch of size `32/11*512 ~ 1500` in the same amount of time (0.05 seconds / image). 

Run the fast generation with the following command:
```shell
# generate new images
cd fast-pixel-cnn
python generate.py --checkpoint [PixelCNN save directory]/params_hawc2.ckpt --save_dir [location to save images] --batch-size [max size that fits on GPU]

# now visualize results
cd ..
python plot.py --num [iteration number to view] --chs 2 --data-path [dir of processed HAWC data] --save-path [save dir from above]
```


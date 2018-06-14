# HAWC Deep Learning

Deep learning models on HAWC simulation dataset

## Prerequisites
- HAWC dataset should be downloaded and placed in main directory, `$HAWC`
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
### Generative Model with Pixel-cnn
We can use a pixel-cnn model to generate very realistic PMT grid hit data.

To run pixelcnn on the 40x40 images generated from above, run
``` bash
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

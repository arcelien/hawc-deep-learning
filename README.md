# HAWC Gen

Generative Model on HAWC sim dataset

### Requirements
- HAWC dataset should be downloaded from dagon.nvidia.com and located in $HAWC
- XCDF should be complied
    - contents of the compiled `lib/` folder placed in the main folder
    - Instructions are in the `INSTALL` file at https://github.com/jimbraun/XCDF
    - `libxcdf.so` and `xcdf/`
    - If getting errors with `libxcdf.so`, add it's location to `$LD_LIBRARY_PATH`
    - XCDF currently only supports python 2 and cython 2
- Clone pixel_cnn from `https://github.com/arcelien/pixel-cnn` to the main folder

## Generative Model Instructions
To generate the dataset, run 
``` bash
python parse_hawc.py $HAWC
``` 
The dataset will be stored in `$HAWC/data`

To run pixelcnn on the 40x40 images generated from above, run
``` bash
cd pixel-cnn-hawc
sh train.sh
```
Checkpoints and output from pixelcnn will be located in `$HAWC/save`

### Low Dimensional MLP Gan
We try to learn the distribution of the parameters: `Theta, Azimuth, nPE, nHit`

Use the `gen_gamma_4` function in `parse_hawc.py` to generate this dataset.

Then use one of the GAN functions to learn the distribution of parameters.
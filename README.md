# SPARC: Sparse Render-and-Compare for CAD model alignment from a single RGB image.

Official implementation of the paper

> **SPARC: Sparse Render-and-Compare for CAD model alignment from a single RGB image** \
> British Machine Vision Conference 2022\
> [Florian Langer][flo], [Gwangbin Bae][gb], [Ignas Budvytis][ignas], [Roberto Cipolla][roberto] \
> [arXiv][1]

Code will be released soon!

## Installation Requirements
- [PyTorch][torch]
- [PyTorch3D][py3d]

We recommend installing via conda.
```
conda create -n sparc python=3.9
conda activate sparc
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
conda install -c bottler nvidiacub
conda install pytorch3d -c pytorch3d
```

After installing the packages above install additional dependencies by
```
pip install -r requirements.txt
```
To install this repo
```
git clone https://github.com/florianlanger/SPARC
cd SPARC && pip install -e .
```

## Data

Download data from this link:



* sparc_release
  * data
    * data_3d
    * data_scannet
    * train
    * val
  * main_experiment
    * results
      * per_frame_predictions.json
      * raw_results.csv
      * results_scannet.txt
    * visualisation
    * config.json
    * network.pth

The folder `data` contains all necessary data for training and evaluating SPARC on ScanNet25k.
We also directly release our results. The `results` folder contains per frame predictions as well as the predictions transformed into ScanNet world coodinates `raw_results.csv`. The accuracies we obtain are provided in `results_scannet.txt`. We also visualise predictions for all images in `visualisation`.

## Training/Evaluating
To train open the config file in the SPARC code and replace the tags ["general"]["output_dir"] and ["general"]["dataset_dir"] with the intended output dir path and the path to the downloaded and unzipped SPARC data.
For trainig run `python main.py`. For evaluating run `bash eval.sh`. This will evaluate the provided model by first selecting one of four rotation initialisations for each image and then iteratively improving the pose for the best initialisation. 



## Citations
If you find our work helpful for your research please consider citing the following publication:
```
@inproceedings{sparc,
               author = {Langer, F. and Bae, G. and Budvytis, I. and Cipolla, R.},
               title = {SPARC: Sparse Render-and-Compare for CAD model alignment in a single RGB image},
               booktitle = {Proc. British Machine Vision Conference},
               month = {November},
               year = {2022},
               address={London}
}
```

[1]: https://arxiv.org/
[flo]: https://www.florianlanger.co.uk
[roberto]: https://mi.eng.cam.ac.uk/~cipolla/
[ignas]: http://mi.eng.cam.ac.uk/~ib255/
[gb]: https://www.baegwangbin.com/about
[py3d]: https://github.com/facebookresearch/pytorch3d
[torch]: https://pytorch.org

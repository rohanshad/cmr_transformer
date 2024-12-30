## A Generalizable Deep Learning System for Cardiac MRI
###### Rohan Shad, Cyril Zakka, Dhamanpreet Kaur, Robyn Fong, Ross Warren Filice, John Mongan, Kimberly Kalianos, Nishith Khandwala, David Eng, Matthew Leipzig, Walter Witschey, Alejandro de Feria, Victor Ferrari, Euan Ashley,  Michael A. Acker, Curtis Langlotz, William Hiesinger

[![arXiv](https://img.shields.io/badge/arXiv-2312.00357-b31b1b.svg)](https://arxiv.org/abs/2312.00357)

#### Project overview: 

![input_mri](https://github.com/rohanshad/cmr_transformer/blob/760cd4a200155dd95c30f3900594b3127785b001/media/mri_inputs.gif)
Here we describe a transformer-based vision system that learns complex pathophysiological visual representations from a large multi-institutional dataset of 19,041 CMR scans, guided by natural language supervision from the text reports accompanying each CMR study. We use a large language model to help ‘teach’ a vision network to generate meaningful low-dimensional representations of CMR studies, by showing examples of how radiologists describe what they see while drafting their reports. We utilize a contrastive learning objective using the InfoNCE objective. The video encoder used is an implementation of [MVIT](https://arxiv.org/abs/2104.11227) (Multi-scale vision transformers) initialzed using Kinetics-400 pre-trained weights. The text encoder used is an implementation of [BERT](http://arxiv.org/abs/1810.04805) (Bidirectional encoder representations with transformers) pretrained on [pubmed abstracts](http://arxiv.org/abs/2007.15779) with a custom vocabulary. Please see our [paper](https://arxiv.org/abs/2312.00357) for more. 


#### Video Dataset Structure

MRI cine sequences are stored within hdf5 files for portability and performance. Pixel information are stored as arrays under a top level directory for each unique patient. Certain views may have more than one 'video' taken at multiple parallel sections (eg: SAX view typically has numerous sequences taken from base to apex of the heart). Attributes such as 'slice frame index' demarcate when each unique video begins and ends. Please see the paper and supplementary appendix for additional details on how we prepare and structure our augmentations. 

```
patient_id
  ├── accession_number_1.h5
  ├── accession_number_2.h5
    ├── 4CH	{data: 4d array (c, f, h, w)} {attr: total images, slice frame index}
    ├── SAX	{data: 4d array (c, f, h, w)} {attr: total images, slice frame index}
    ├── 2CH	{data: 4d array (c, f, h, w)} {attr: total images, slice frame index}
    ├── 3CH	{data: 4d array (c, f, h, w)} {attr: total images, slice frame index}
```

#### Repository Contents
This repository contains template code for finetuning and evaluation, in addition contains all model classes required to load our weights for use in your own projects. To use this repository as is for finetuning on your own datasets, you will need to use [Wandb](https://wandb.ai/site/) for experiment tracking, or make appropriate changes to use [Tensorboard](https://www.tensorflow.org/tensorboard). The repository also relies on a `local_config.yaml` (not included) file to set some variables (eg: `PRETRAIN_WEIGHTS` or `ATTN_DIR`. This format of this file is as follows: 

```
# Device specific options
your_computer_name:
  tmp_dir: 'some/temp/dir'
  pretrain_dir: '/some/pretrain/dir'
  attn_dir: 'some/dir/attention_maps'
```

Using the repo without a `local_config.yaml` file is possible if you hard code those variables in `model_factory.py` and `mri_trainer.py`


#### Download Weights

Weights for our pretrained CMR encoders are available on Huggingface for non-commercial use (CC-BY-NC 4.0): https://huggingface.co/rohanshad/cmr_c0.1. 

#### Install Dependencies 

Tested with CUDA on Ubuntu 20.02, 24.04 and CentOS7. 

1. Create [new conda environment](https://anaconda.org) using python version 3.9

    ```
    conda create -n mri_torch python=3.9
    conda activate mri_torch
    ```
  
2. Install dependencies
    ```
    cat requirements.txt | xargs -n 1 pip install --force-reinstall
    pip install torch==1.11.0+cu113 --extra-index-url https://download.pytorch.org/whl/cu113 --force
    ```

    Specific torch download links for cuda enabled pytorch versions. Codebase not tested with torch > 1.11

3. Download example data: Please reach out to me over email if you wish to test your models on the University of Pennsylvania Cardiac MRI dataset. [Kaggle](https://www.kaggle.com/c/second-annual-data-science-bowl) and [ACDC](https://www.creatis.insa-lyon.fr/Challenge/acdc/databases.html) Datasets are publicly avaialble. Kaggle data will require conversion to hdf5 via preprocessing scripts supplied, ACDC datasets directly usable in native nifti format. 

#### Preprocessing UK BioBank Data

Researchers may have access to the UK BioBank and wish to use our models on CMR data from the UK BioBank. We use scripts avaialble in our [cmr_toolkit](https://github.com/rohanshad/cmr_toolkit) to prepare and pre-process the data. We first run the entire UK BioBank data directory through `tar_compress.py` to restructure data from each scan into a single parent level tarfile unique for each scan. We use the `preprocess_mri.py` and then `build_dataset.py` scripts to build the final hdf5 datastore. Instructions in the `cmr_toolkit` repository. 

#### Evaluate on ACDC Data
```
python mri_trainer.py validate --config configs/acdc_evaluation.yaml
```

#### Example Finetune from pretrained weights on downstream evaluation task

```
python mri_trainer.py fit --config configs/finetune_config.yaml
```

#### Example Evaluate pretrained weights on downstream evaluation task (test dataset)

```
python mri_trainer.py test --config configs/eval_config.yaml
```


### Citation
If you use this codebase, or otherwise found our work valuable, please cite:
```
@misc{shad2023generalizabledeeplearningcardiac,
      title={A Generalizable Deep Learning System for Cardiac MRI}, 
      author={Rohan Shad and Cyril Zakka and Dhamanpreet Kaur and Robyn Fong and Ross Warren Filice and John Mongan and Kimberly Kalianos and Nishith Khandwala and David Eng and Matthew Leipzig and Walter Witschey and Alejandro de Feria and Victor Ferrari and Euan Ashley and Michael A. Acker and Curtis Langlotz and William Hiesinger},
      year={2023},
      eprint={2312.00357},
      archivePrefix={arXiv},
      primaryClass={eess.IV},
      url={https://arxiv.org/abs/2312.00357}, 
}
```




 
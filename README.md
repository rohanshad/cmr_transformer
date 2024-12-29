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

#### Preprocessing UK BioBank Data

Researchers may have access to the UK BioBank and wish to use our models on CMR data from the UK BioBank. We use scripts avaialble in our [cmr_toolkit](https://github.com/rohanshad/cmr_toolkit) to prepare and pre-process the data. We first run the entire UK BioBank data directory through `tar_compress.py` to restructure data from each scan into a single parent level tarfile unique for each scan. We use the `preprocess_mri.py` and then `build_dataset.py` scripts to build the final hdf5 datastore. Instructions in the `cmr_toolkit` repository. 

#### Download Weights

Weights for the contrastive pretrained vision encoders are available here: [placeholder link]()

#### Install Dependencies 

Tested on Ubuntu 20.02 and CentOS7 

1. Create [new conda environment](https://anaconda.org) using python version 3.9

    ```
    conda create -n mri_torch python=3.9
    conda activate mri_torch
    ```
  
2. Install dependencies

    ```
    cat requirements.txt | xargs -n 1 pip install
    ```

3. Download example data: Please reach out to me over email if you wish to test your models on the University of Pennsylvania Cardiac MRI dataset. [Kaggle](https://www.kaggle.com/c/second-annual-data-science-bowl) and [ACDC](https://www.creatis.insa-lyon.fr/Challenge/acdc/databases.html) Datasets are publicly avaialble. Kaggle data will require conversion to hdf5 via preprocessing scripts supplied, ACDC datasets directly usable in native nifti format. 


#### Finetune from pretrained weights on downstream evaluation task

```
python mri_trainer.py fit --config finetune_config.yaml
```

#### Evaluate pretrained weights on downstream evaluation task (test dataset)

```
python mri_trainer.py test --config eval_config.yaml
```

 
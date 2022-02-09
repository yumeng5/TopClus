# TopClus

The source code used for [**Topic Discovery via Latent Space Clustering of Pretrained Language Model Representations**](), published in WWW 2022.

## Requirements

At least one GPU is required to run the code.

Before running, you need to first install the required packages by typing following commands (Using a virtual environment is recommended):

```
$ pip3 install -r requirements.txt
```

You need to also download the following resources in NLTK:
```
import nltk
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('universal_tagset')
```

## Running Topic Discovery


## Running Document Clustering


## Citations

Please cite the following paper if you find the code helpful for your research.
```
@inproceedings{meng2022topic,
  title={Topic Discovery via Latent Space Clustering of Pretrained Language Model Representations},
  author={Meng, Yu and Zhang, Yunyi and Huang, Jiaxin and Zhang, Yu and Han, Jiawei},
  booktitle={The Web Conference},
  year={2022},
}
```

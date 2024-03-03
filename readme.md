# Multi-modal emotion recognition via unified granularity contrastive learning and similar negative discrimination



## Modalities

Audio & Video



## Datasets

[CREMA-D](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4313618/)

[IEMOCAP](https://sail.usc.edu/iemocap/)



## Dependencies

- Python 3.8.12
- Pytorch 1.7.1



## Pre-processing

For audio data, no preprocessing is applied. For video data, we first detect the facial regions using **VGG-Face**, then extract feature vectors using a pre-trained **Vision-Transformer**.



## Running

```
`bash sh.sh`
```




# ConvCRF - TF2
Re-implementation of ConvCRF for Tensorflow 2
Based on original pytorch implementation: https://github.com/MarvinTeichmann/ConvCRF and out-dated tensorflow implementation: https://github.com/feixiang7701/ConvCRF

Method described in the paper ["Convolutional CRFs for Semantic Segmentation" writed by Marvin T. T. Teichmann and Roberto Cipolla.](https://arxiv.org/abs/1805.04777)

To test:
```
python convcrf.py
```
Tests a forward and backwards pass through the layer. Expect no warnings or errors.

Demo run:
Bilateral kernel, spatial kernel, compatibility matrix initialized from [CRFRNN checkpoint](https://github.com/sadeepj/crfasrnn_keras/tree/master)
```
python demo.py data/2007_000033_0img.png ./data/2007_000033_5labels.png" --output ./data/output/2007_000033_crf.png
```

# Multidimensional LSTM (2D)
This is an implementation of the MDLSTM layer used for [Adaptive Optics Image Segmentation](https://www.nature.com/articles/s41598-018-26350-3). And is obviously based on the original [paper](https://arxiv.org/abs/0705.2011) with some modification described below. 


## Faster MDLSTM
To speed up training and inference, the MDLSTM layer is implemented as described [here](https://ieeexplore.ieee.org/document/7814068), which simply says that we can go diagonal by diagonal rather than pixel by pixel, to take greater advantage of parallelism.

## Cells 
There are currently a choice of two cell state calculations. One from the original paper:
<p align="center"><img alt="$$ci + c_{\text{up}}f_1 + c_{\text{left}}f_2,$$" src="./svgs/b4654fa7580147c03705b6983106e382.svg" align="middle" width="137.5412346pt" height="16.1187015pt"/></p>
and one which attempts to be more stable taken from [here](https://dl.acm.org/citation.cfm?id=2946645.3007050)
<p align="center"><img alt="$$ci + \dfrac{c_{\text{up}}f_1 + c_{\text{left}}f_2}{f_1 + f_2}(1 - i).$$" src="./svgs/b0e14f6eb8534349faaad143e2205eb9.svg" align="middle" width="188.2454706pt" height="37.0084374pt"/></p>
The later tries to keep the resulting cell state bounded (if <img alt="$|c_{\text{up}}|, |c_{\text{left}}| &lt; 1 \implies |c_\text{new}| &lt; 1$" src="./svgs/473ac41d69bcef9a3cbfe143d5c5897b.svg" align="middle" width="219.32059214999995pt" height="24.65753399999998pt"/>).

## Of note
- Gradients are clipped by norm, within the recurrent steps
- Layer runs 4 MDLSTM blocks 4 directions as in original paper

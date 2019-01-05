<style TYPE="text/css">
code.has-jax {font: inherit; font-size: 100%; background: inherit; border: inherit;}
</style>
<script type="text/x-mathjax-config">
MathJax.Hub.Config({
    tex2jax: {
        inlineMath: [['$','$'], ['\\(','\\)']],
        skipTags: ['script', 'noscript', 'style', 'textarea', 'pre'] // removed 'code' entry
    }
});
MathJax.Hub.Queue(function() {
    var all = MathJax.Hub.getAllJax(), i;
    for(i = 0; i < all.length; i += 1) {
        all[i].SourceElement().parentNode.className += ' has-jax';
    }
});
</script>
<script type="text/javascript" src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.4/MathJax.js?config=TeX-AMS_HTML-full"></script>

# Multidimensional LSTM (2D)
This is an implementation of the MDLSTM layer used for [Adaptive Optics Image Segmentation](https://www.nature.com/articles/s41598-018-26350-3). And is obviously based on the original [paper](https://arxiv.org/abs/0705.2011) with some modification described below. 


## Faster MDLSTM
To speed up training and inference, the MDLSTM layer is implemented as described [here](https://ieeexplore.ieee.org/document/7814068), which simply says that we can go diagonal by diagonal rather than pixel by pixel, to take greater advantage of parallelism.

## Cells 
There are currently a choice of two cell state calculations. One from the original paper:
$$ci + c_{\text{up}}f_1 + c_{\text{left}}f_2,$$
and one which attempts to be more stable taken from [here](https://dl.acm.org/citation.cfm?id=2946645.3007050)
$$ci + \dfrac{c_{\text{up}}f_1 + c_{\text{left}}f_2}{f_1 + f_2}(1 - i).$$
The later tries to keep the resulting cell state bounded (if $|c_x| < 1 \implies |new_c| < 1$).

## Of note
- Gradients are clipped by norm, within the recurrent steps
- Layer runs 4 MDLSTM blocks 4 directions as in original paper

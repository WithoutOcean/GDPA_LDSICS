# GDPA_LDSICS

GDPA_LDSICS: graph and double pyramid attention network based on linear discrimination of spectral interclass slices for hyperspectral image classification

ABSTRACT:In recent years, convolution neural networks (CNNs) and graph convolution networks (GCNs) have been widely used in hyperspectral image classification (HSIC). In HSIC, CNNs can effectively extract the spatial spectral features of hyperspectral images (HSIs), while GCNs can quickly capture the structural features of HSIs. The proper combination of CNNs and GCNs is beneficial to improve classification performance of HSIs. However, the high redundancy of feature information and the problem of small sample are still the major challenges of HSIC. In order to alleviate these problems, in this paper, a new graph and double pyramid attention network based on linear discrimination of spectral interclass slices (GDPA_LDSICS) is proposed. First, a linear discrimination of spectral inter class slices (LDSICS) module is designed. The LDSICS module can effectively eliminate a lot of redundancy in spectral dimension, which is conducive to subsequent feature extraction. Then, the spatial spectral deformation (SSD) module is constructed, which can effectively correlate the spatial spectral information closely. Finally, in order to alleviate the problem of small sample, a double branch structure of CNN and GCN is developed. On the CNN branch, a double pyramid attention (DPA) structure is designed to model context semantics to avoid information loss caused by long-distance feature extraction. On the GCN branch, an adaptive dynamic encoding (ADE) method is proposed, which can more effectively capture the topological structure of spatial spectral features. Experiments on four open datasets show that the GDPA_LDSICS can provide better classification performance and generalization performance than other most advanced methods.

Environment: 
Python 3.7
PyTorch 1.10

How to use it?
---------------------
Here an example experiment is given by using **Indian Pines hyperspectral data**. Here is an example experiment using hyperspectral data of Indian pine trees. Please note that due to the randomness of parameter initialization, the experimental results may differ slightly from the results reported in the paper.

If you want to run the code in your own data, you can accordingly change the input (e.g., data, labels) and tune the parameters. For more details, please refer to the paper.

If you encounter the bugs while using this code, please do not hesitate to contact us.

If this work is helpful to you, please cite our paper: 
Haiyang Wu, Cuiping Shi & Liguo Wang (2023) GDPA_LDSICS: graph and double pyramid attention network based on linear discrimination of spectral interclass slices for hyperspectral image classification, International Journal of Remote Sensing, 44:17, 5283-5312, DOI: 10.1080/01431161.2023.2247523

If emergency, you can also add my email: 17398893383@163.com or QQ: 1752436056.


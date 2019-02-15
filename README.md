# FusDeepSeg
Fusion semantic segmantation with deeplabv3+ and AdapNet.

We proposed a separate deep-learning segmentation network for RGB and depth(Deeplabv3+ for RGB, AdapNet for Depth map). Then we use a fusion approach for the result of individual expert network to improve the accuracy of image segmentation. This fusion approach is based on Bayes and Dirichlet distributions. The background of image can be classified more accurately with Bayes fusion in this approach.

![alt text](https://github.com/wangyidong3/FusDeepSeg/blob/master/xview/our_solution.png)



![alt text](https://github.com/wangyidong3/FusDeepSeg/blob/master/xview/official_result.png)
![alt text](https://github.com/wangyidong3/FusDeepSeg/blob/master/xview/myplot2.png)


The background segmentation is only correctly recognized with our approach when bayes
fusion is used.

这是一份我自己学习cs336的笔记 ,仅供参考

# 1.Overview

##### 小规模的模型展现的特质不一定能外推到大规模的模型上  

1.参数占比随着规模变大发生改变

![image-20260506180700970](./readme.assets/image-20260506180700970.png)

对于不同规模的模型 , 不同layer所占参数可能不同 , 因此可能对于attention的优化在小模型上效果更好 , 但在更大规模的模型上因为参数占比的缘故导致效果降低

2.涌现(emergence of behavior with scale)

这个就不赘述了

##### 可迁移的知识

1. how work(how to build)怎么构建
2. how to squeeze hardware, taking scaling seriously(调用硬件, 正确提升规模)
3. which data modeling decision yield good accuracy 哪些数据建模可以带来好的性能
4. profile and benchmark everying (对所有东西进行性能分析和基准测试)



可规模化扩展的算法更重要Right interpretation:algorithms that scale are what matter.

`accuracy = efficiency x resources`



Resource accounting:memory and compute characteristics(显存和算力分配 , 决定了能训练多大的模型) FLOPs

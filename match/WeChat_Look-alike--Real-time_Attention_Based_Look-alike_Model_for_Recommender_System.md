Real-time Attention Based Look-alike Model (RALM) for Recommender System

RALM is a similarity based look-alike model, which consists of user representation learning and look-alike learning.



很多DNN的推荐模型，都会有严重的热门偏好，使得推荐系统的马太效应（Matthew effect）显著，很多长尾 的优质内容得不到有效的流量支持。



![image-20201030172711916](pics/image-20201030172711916.png)

seeds-to-user similarity calculation

- User Representation: 用attention merge layer代替concatenation

- Seeds Representation: global representation and local representation of seeds based on global and local attention units.



​	look-alike算法可以分成两类：similarity based methods and regression based methods.

![image-20201103155214641](pics/image-20201103155214641.png)

流程：







---

![image-20201103163530075](pics/image-20201103163530075.png)



local attention & global attention

<img src="pics/image-20201104155126058.png" alt="image-20201104155126058" style="zoom:50%;" />



 






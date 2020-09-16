[FwFM-Field-weighted Factorization Machines for Click-Through Rate Prediction in Display Advertising](FwFM-Field-weighted Factorization Machines for Click-Through Rate Prediction in Display Advertising.pdf)



FFM的参数量巨大一直饱受诟病。Field-weighted Factorization Machines (FwFMs) 参数量比FFM少很多，但是也可以达到和FFM差不多的效果。

不同域的特征交叉带来的效果有很大的差异，我们可以用互信息来进行验证。Mutual Information（互信息）可用于衡量和对比不同特征交叉的强度。

![image-20200916113332407](pics/image-20200916113332407.png)

正是由于不同特征域的组合重要性不一样，由此提出了Field-weighted Factorization Machines(FwFMs)模型，直接对不同域差异化的组合强度建模。特征交叉为：
$$
x_{i} x_{j}\left\langle\boldsymbol{v}_{i}, \boldsymbol{v}_{j}\right\rangle r_{F(i), F(j)}
$$
其中$r_{F(i), F(j)}$是$F(i)$和$F(j)$特征域交叉的权重，来显式得表征不同特征域交叉的重要性是有差异的（FFM可以隐式表达这种差异）。FwFMs完整公式如下：
$$
\Phi_{F w F M s}((\boldsymbol{w}, \boldsymbol{v}), \boldsymbol{x})=w_{0}+\sum_{i=1}^{m} x_{i} w_{i}+\sum_{i=1}^{m} \sum_{j=i+1}^{m} x_{i} x_{j}\left\langle\boldsymbol{v}_{i}, \boldsymbol{v}_{j}\right\rangle r_{F(i), F(j)}
$$
FMs/FFMs/FwFMs的对比：

![image-20200916141032448](pics/image-20200916141032448.png)

参数比较：

![image-20200916141257050](pics/image-20200916141257050.png)

由于通常n<<m，FwFM的参数量接近FM，但是远小于FFM。

不同模型的公式表达：

LR
$$
\Phi_{L R}(\boldsymbol{w}, \boldsymbol{x})=w_{0}+\sum_{i=1}^{m} x_{i} w_{i}
$$

Poly2
$$
\Phi_{P o l y 2}(w, x)=w_{0}+\sum_{i=1}^{m} x_{i} w_{i}+\sum_{i=1}^{m} \sum_{j=i+1}^{m} x_{i} x_{j} w_{h(i, j)}
$$
FM
$$
\Phi_{F M s}((\boldsymbol{w}, \boldsymbol{v}), \boldsymbol{x})=w_{0}+\sum_{i=1}^{m} x_{i} w_{i}+\sum_{i=1}^{m} \sum_{j=i+1}^{m} x_{i} x_{j}\left\langle\boldsymbol{v}_{i}, \boldsymbol{v}_{j}\right\rangle
$$
FFM
$$
\Phi_{F F M s}((\boldsymbol{w}, \boldsymbol{v}), \boldsymbol{x})=w_{0}+\sum_{i=1}^{m} x_{i} w_{i}+\sum_{i=1}^{m} \sum_{j=i+1}^{m} x_{i} x_{j}\left\langle\boldsymbol{v}_{i, F(j)}, \boldsymbol{v}_{j, F(i)}\right\rangle
$$
FwFM
$$
\Phi_{F w F M s}((\boldsymbol{w}, \boldsymbol{v}), \boldsymbol{x})=w_{0}+\sum_{i=1}^{m} x_{i} w_{i}+\sum_{i=1}^{m} \sum_{j=i+1}^{m} x_{i} x_{j}\left\langle\boldsymbol{v}_{i}, \boldsymbol{v}_{j}\right\rangle r_{F(i), F(j)}
$$



FwFM在线性部分的不同形式：

1. 原始的形式：$\sum_{i=1}^{m} x_{i} w_{i}$
2. 将特征的embedding加入线性项：$\sum_{i=1}^{m} x_{i}\left\langle\boldsymbol{v}_{i}, \boldsymbol{w}_{i}\right\rangle$

此时参数量为mK，m是特征数

3. 如果将参数从feature-wise改成field-wise，同一个特征域共享一个权重系数，则：$\sum_{i=1}^{m} x_{i}\left\langle\boldsymbol{v}_{i}, \boldsymbol{w}_{F(i)}\right\rangle$

此时参数量为nK，n是特征域的数









































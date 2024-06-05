1. 梯度下降算法的正确步骤是什么  
  a. 用随机值初始化权重和偏差  
  b. 把输入传入网络，得到输出值  
  c. 计算预测值和真实值之间的误差  
  d. 对每一个产生误差的神经元，调整相应的权重值以减少误差  
  e. 重复迭代，直至得到网络权重的最佳值  

2. 已知：

  - 大脑是有很多个叫做神经元的东西构成，神经网络是对大脑的简单的数学表达。
  - 每一个神经元都有输入、处理函数和输出。
  - 神经元组合起来形成了网络，可以拟合任何函数。
  - 为了得到最佳的神经网络，我们用梯度下降方法不断更新模型

  给定上述关于神经网络的描述，什么情况下神经网络模型被称为深度学习模型？

  A. **加入更多层，使神经网络的深度增加**

  B. 有维度更高的数据

  C. 当这是一个图形识别的问题时

  D. 以上都不正确

  ```
  解析：正确答案A，更多层意味着网络更深。没有严格的定义多少层的模型才叫深度模型，目前如果有超过2层的隐层，那么也可以及叫做深度模型。
  ```

3. 训练CNN时，可以对输入进行旋转、平移、缩放等预处理提高模型泛化能力。这么说是对，还是不对？

   A.**对**   B.不对

   ```
   解析：对。训练CNN时，可以进行这些操作。当然也不一定是必须的，只是data augmentation扩充数据后，模型有更多数据训练，泛化能力可能会变强。
   ```

4. 下面哪项操作能实现跟神经网络中Dropout的类似效果?

   A.Boosting **B.Bagging** C.Stacking D.Mapping

   ```
   解析：正确答案B。Dropout可以认为是一种极端的Bagging，每一个模型都在单独的数据上训练，同时，通过和其他模型对应参数的共享，从而实现模型参数的高度正则化。
   ```

5. 下列哪一项在神经网络中引入了非线性？

   A.随机梯度下降

   **B.修正线性单元（ReLU）**

   C.卷积函数

   D.以上都不正确

6. CNN的卷积核是单层的还是多层的？

   ```
   一般而言，深度卷积网络是一层又一层的。层的本质是特征图, 存贮输入数据或其中间表示值。一组卷积核则是联系前后两层的网络参数表达体, 训练的目标就是每个卷积核的权重参数组。
   描述网络模型中某层的厚度，通常用名词通道channel数或者特征图feature map数。不过人们更习惯把作为数据输入的前层的厚度称之为通道数（比如RGB三色图层称为输入通道数为3），把作为卷积输出的后层的厚度称之为特征图数。
   卷积核(filter)一般是3D多层的，除了面积参数, 比如3x3之外, 还有厚度参数H（2D的视为厚度1). 还有一个属性是卷积核的个数N。
   卷积核的厚度H, 一般等于前层厚度M(输入通道数或feature map数). 特殊情况M > H。
   卷积核的个数N, 一般等于后层厚度(后层feature maps数，因为相等所以也用N表示)。
   卷积核通常从属于后层，为后层提供了各种查看前层特征的视角，这个视角是自动形成的。
   卷积核厚度等于1时为2D卷积，对应平面点相乘然后把结果加起来，相当于点积运算；
   卷积核厚度大于1时为3D卷积，每片分别平面点求卷积，然后把每片结果加起来，作为3D卷积结果；1x1卷积属于3D卷积的一个特例，有厚度无面积, 直接把每片单个点乘以权重再相加。
   归纳之，卷积的意思就是把一个区域，不管是一维线段，二维方阵，还是三维长方块，全部按照卷积核的维度形状，对应逐点相乘再求和，浓缩成一个标量值也就是降到零维度，作为下一层的一个feature map的一个点的值！
   
   
   可以比喻一群渔夫坐一个渔船撒网打鱼，鱼塘是多层水域，每层鱼儿不同。
   船每次移位一个stride到一个地方，每个渔夫撒一网，得到收获，然后换一个距离stride再撒，如此重复直到遍历鱼塘。
   A渔夫盯着鱼的品种，遍历鱼塘后该渔夫描绘了鱼塘的鱼品种分布；
   B渔夫盯着鱼的重量，遍历鱼塘后该渔夫描绘了鱼塘的鱼重量分布；
   还有N-2个渔夫，各自兴趣各干各的；
   最后得到N个特征图，描述了鱼塘的一切！
   
   2D卷积表示渔夫的网就是带一圈浮标的渔网，只打上面一层水体的鱼；
   3D卷积表示渔夫的网是多层嵌套的渔网，上中下层水体的鱼儿都跑不掉；
   1x1卷积可以视为每次移位stride，甩钩钓鱼代替了撒网；
   
   实际上，除了输入数据的通道数比较少之外，中间层的feature map数很多，这样中间层算卷积会累死计算机（鱼塘太深，每层鱼都打，需要的鱼网太重了）。所以很多深度卷积网络把全部通道/特征图划分一下，每个卷积核只看其中一部分（渔夫A的渔网只打捞深水段，渔夫B的渔网只打捞浅水段）。这样整个深度网络架构是横向开始分道扬镳了，到最后才又融合。这样看来，很多网络模型的架构不完全是突发奇想，而是是被参数计算量逼得。特别是现在需要在移动设备上进行AI应用计算(也叫推断), 模型参数规模必须更小, 所以出现很多减少握手规模的卷积形式, 现在主流网络架构大都如此。
   ```

7. 什么是卷积？

   ```
   对图像（不同的数据窗口数据）和滤波矩阵（一组固定的权重：因为每个神经元的多个权重固定，所以又可以看做一个恒定的滤波器filter）做内积（逐个元素相乘再求和）的操作就是所谓的『卷积』操作，也是卷积神经网络的名字来源。
   非严格意义上来讲，下图中红框框起来的部分便可以理解为一个滤波器，即带着一组固定权重的神经元。多个滤波器叠加便成了卷积层。
   ```

   ![](https://ask.julyedu.com/uploads/article/20180709/d878fe44d36031f8d7a45224f7a622a3.png)

   OK，举个具体的例子。比如下图中，图中左边部分是原始输入数据，图中中间部分是滤波器filter，图中右边是输出的新的二维数据。

   ![](https://ask.julyedu.com/uploads/article/20180709/398d40ebfd4510eae029a2858eb98569.png)

   分解下上图

   ![](https://ask.julyedu.com/uploads/article/20180709/f95b781eea1585af694b2d68e6c4b907.jpg)

   中间滤波器filter与数据窗口做内积，其具体计算过程则是：4*0 + 0*0 + 0*0 + 0*0 + 0*1 + 0*1 + 0*0 + 0*1 + -4*2 = -8

8. 什么是CNN的池化pool层？

   池化，简言之，即取区域平均或最大，如下图所示（图引自cs231n）

   ![](https://ask.julyedu.com/uploads/article/20180709/7949d0925f3c74a2dac7e4e71b14f678.jpg)

   上图所展示的是取区域最大，即上图左边部分中左上角2x2的矩阵中6最大，右上角2x2的矩阵中8最大，左下角2x2的矩阵中3最大，右下角2x2的矩阵中4最大，所以得到上图右边部分的结果：6，8，3，4

9. 简述下什么是生成对抗网络

   ```
   GAN之所以是对抗的，是因为GAN的内部是竞争关系，一方叫generator，它的主要工作是生成图片，并且尽量使得其看上去是来自于训练样本的。另一方面是 discriminator，其目标是判断输入图片是否属于真实训练样本。
   
   更直白的讲，将generator想象成假币制造商，而discriminator是警察。generator目的是尽可能把假币造的跟真的一样，从而能够骗过discriminator，即生成样本并使它看上去好像来自于真实训练样本一样。
   
   生成对抗网络的一个简单解释如下：假设有两个模型，一个是生成模型（Generative Model，下文简写G），一个是判别模型（Discriminative Model，下文简写D），判别模型（D）的任务就是判断一个实例是真实的还是由模型生成的，生成模型（G）的任务是生成一个实例来骗过判别模型（D），两个模型互相对抗，发展下去就会达到一个平衡，生成模型生成的实例与真实的没有区别，判别模型无法区分自然的还是模型生成的。以赝品商人为例，赝品商人一直提升他的高仿水平来区分行家，行家也一直学习真的假的毕加索画作来提升自己的辨识能力，两个人一直博弈，最后赝品商人高仿的毕加索画作达到了以假乱真的水平，行家最后也很难区分正品和赝品了。
   ```

   如下图中的左右两个场景：

   ![](https://ask.julyedu.com/uploads/article/20180709/a741f2156acfcae53f020a66f47c5aef.png)

10. 学梵高作画的原理是什么

    http://www.julyedu.com/video/play/42/523


11. **请简要介绍下tensorflow的计算图。**

     ```
    tensorflow是一个通过计算图的形式来表述计算的编程系统，计算图也叫做数据流图，可以把计算图看做是一种有向图，Tensorflow中的每一个节点都是计算图上的一个Tensor，也就是张量，而节点之间的边描述了计算之间的依赖关系（定义时）和数学操作（运算时）
     ```

    如下图所示：

    a=x*y; b=a+z; c=tf.reduce_sum(b);

    ![](http://5b0988e595225.cdn.sohucs.com/images/20180710/724b10e2eb9741388a8e91f36b678ced.jpg)

    ![](http://5b0988e595225.cdn.sohucs.com/images/20180710/ef01099ff4614bb395a8b5a684713599.gif)

12. 你有哪些deep learning(run, cnn)调参的经验

    ```
    a. 参数初始化
    	下面几种方式,随便选一个,结果基本都差不多。但是一定要做。否则可能会减慢收敛速度，影响收敛结果，甚至造成Nan等一系列问题。
    	下面的n_in为网络的输入大小，n_out为网络的输出大小，n为n_in或(n_in+n_out)*0.5
    	
    	Xavier初始法论文：http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf
    	He初始化论文：https://arxiv.org/abs/1502.01852
    	uniform均匀分布初始化：w = np.random.uniform(low=-scale, high=scale, size=[n_in,n_out])
    	Xavier初始法，适用于普通激活函数(tanh,sigmoid)：scale = np.sqrt(3/n)
    	He初始化，适用于ReLU：scale = np.sqrt(6/n)
    	normal高斯分布初始化：w = np.random.randn(n_in,n_out) * stdev # stdev为高斯分布的标准差，均值设为0
    	Xavier初始法，适用于普通激活函数 (tanh,sigmoid)：stdev = np.sqrt(n)
    	He初始化，适用于ReLU：stdev = np.sqrt(2/n)
    	svd初始化：对RNN有比较好的效果。
    	
     b. 数据预处理方式
     	zero-center ,这个挺常用的.X -= np.mean(X, axis = 0) # zero-centerX /= np.std(X, axis = 0) # normalize
     	PCA whitening,这个用的比较少.
     	
     c. 训练技巧
     	要做梯度归一化,即算出来的梯度除以minibatch size
     	clip c(梯度裁剪): 限制最大梯度,其实是value = sqrt(w1^2+w2^2….),如果value超过了阈值,就算一个衰减系系数,让value的值等于阈值: 5,10,15
     	dropout对小数据防止过拟合有很好的效果,值一般设为0.5,小数据上dropout+sgd在我的大部分实验中，效果提升都非常明显.因此可能的话，建议一定要尝试一下。 dropout的位置比较有讲究, 对于RNN,建议放到输入->RNN与RNN->输出的位置.关于RNN如何用dropout,可以参考这篇论文:http://arxiv.org/abs/1409.2329
     	adam,adadelta等,在小数据上,我这里实验的效果不如sgd, sgd收敛速度会慢一些，但是最终收敛后的结果，一般都比较好。如果使用sgd的话,可以选择从1.0或者0.1的学习率开始,隔一段时间,在验证集上检查一下,如果cost没有下降,就对学习率减半. 我看过很多论文都这么搞,我自己实验的结果也很好. 当然,也可以先用ada系列先跑,最后快收敛的时候,更换成sgd继续训练.同样也会有提升.据说adadelta一般在分类问题上效果比较好，adam在生成问题上效果比较好。
     	除了gate之类的地方,需要把输出限制成0-1之外,尽量不要用sigmoid,可以用tanh或者relu之类的激活函数.
     	1. sigmoid函数在-4到4的区间里，才有较大的梯度。之外的区间，梯度接近0，很容易造成梯度消失问题。
     	2. 输入0均值，sigmoid函数的输出不是0均值的。
     	rnn的dim和embdding size,一般从128上下开始调整. batch size,一般从128左右开始调整.batch size合适最重要,并不是越大越好。
     	word2vec初始化,在小数据上,不仅可以有效提高收敛速度,也可以可以提高结果。
    
    d. 尽量对数据做shuffle
    	LSTM 的forget gate的bias,用1.0或者更大的值做初始化,可以取得更好的结果,来自这篇论文:http://jmlr.org/proceedings/papers/v37/jozefowicz15.pdf, 我这里实验设成1.0,可以提高收敛速度.实际使用中,不同的任务,可能需要尝试不同的值.
    	Batch Normalization据说可以提升效果，不过我没有尝试过，建议作为最后提升模型的手段，参考论文：Accelerating Deep Network Training by Reducing Internal Covariate Shift
    	如果你的模型包含全连接层（MLP），并且输入和输出大小一样，可以考虑将MLP替换成Highway Network,我尝试对结果有一点提升，建议作为最后提升模型的手段，原理很简单，就是给输出加了一个gate来控制信息的流动，详细介绍请参考论文: http://arxiv.org/abs/1505.00387
    	来自@张馨宇的技巧：一轮加正则，一轮不加正则，反复进行。（没有太明白）
    	
    e. ensemble
    	Ensemble是论文刷结果的终极核武器,深度学习中一般有以下几种方式
    	* 同样的参数,不同的初始化方式
    	* 不同的参数,通过cross-validation,选取最好的几组
    	* 同样的参数,模型训练的不同阶段，即不同迭代次数的模型。
    	* 不同的模型,进行线性融合. 例如RNN和传统模型。
    ```

13. **CNN最成功的应用是在CV，那为什么NLP和Speech的很多问题也可以用CNN解出来？为什么AlphaGo里也用了CNN？这几个不相关的问题的相似性在哪里？CNN通过什么手段抓住了这个共性？**

    ```
    Deep Learning -Yann LeCun, Yoshua Bengio & Geoffrey Hinton
    Learn TensorFlow and deep learning, without a Ph.D.
    The Unreasonable Effectiveness of Deep Learning -LeCun 16 NIPS Keynote
    ```

    以上几个不相关问题的相关性在于，都存在局部与整体的关系，由低层次的特征经过组合，组成高层次的特征，并且得到不同特征之间的空间相关性。如下图：低层次的直线／曲线等特征，组合成为不同的形状，最后得到汽车的表示。

    ![](http://5b0988e595225.cdn.sohucs.com/images/20180710/89726c9e34f9469d9bd192c3cbb85c0c.jpg)

    CNN抓住此共性的手段主要有四个：局部连接 / 权值共享 / 池化操作 / 多层次结构

    局部连接使网络可以提取数据的局部特征：权值共享大大降低了网络的训练难度，一个Filter只提取一个特征，在整个图片（或者语音/文本）中进行卷积；池化操作与多层次结构一起，实现了数据的降维，将低层次的局部特征组合成为较高层次的特征，从而对整个图片进行表示。如下图

    ![](http://5b0988e595225.cdn.sohucs.com/images/20180710/84f926b9aced4cf09763979b0cda5329.jpg)

    上图中，如果每一个点的处理使用相同的Filter，则为全卷积，如果使用不同的Filter，则为Local-Conv。

14. **LSTM结构推导，为什么比RNN好？**

    ```
    推导forget gate，input gate，cell state， hidden information等的变化；因为LSTM有进有出且当前的cell informaton是通过input gate控制之后叠加的，RNN是叠乘，因此LSTM可以防止梯度消失或者爆炸。
    ```

15. **Sigmoid、Tanh、ReLu这三个激活函数有什么缺点或不足，有没改进的激活函数。**

    ```
    sigmoid、Tanh、ReLU的缺点在121问题中已有说明，为了解决ReLU的dead cell的情况，发明了Leaky Relu， 即在输入小于0时不让输出为0，而是乘以一个较小的系数，从而保证有导数存在。同样的目的，还有一个ELU，函数示意图如下。
    ```

    ![](http://5b0988e595225.cdn.sohucs.com/images/20180710/e9fc81364eed48a8b4342a6a25ade06c.jpg)

    还有一个激活函数是Maxout，即使用两套w,b参数，输出较大值。本质上Maxout可以看做Relu的泛化版本，因为如果一套w,b全都是0的话，那么就是普通的ReLU。Maxout可以克服Relu的缺点，但是参数数目翻倍。

    ![](http://5b0988e595225.cdn.sohucs.com/images/20180710/7e61a7057cbe41aba95391dd7aaa15f2.jpg)

16. **为什么引入非线性激励函数？**

    ```
    a. 对于神经网络来说，网络的每一层相当于f(wx+b)=f(w'x)，对于线性函数，其实相当于f(x)=x，那么在线性激活函数下，每一层相当于用一个矩阵去乘以x，那么多层就是反复的用矩阵去乘以输入。根据矩阵的乘法法则，多个矩阵相乘得到一个大矩阵。所以线性激励函数下，多层网络与一层网络相当。比如，两层的网络f(W1*f(W2x))=W1W2x=Wx。
    b. 非线性变换是深度学习有效的原因之一。原因在于非线性相当于对空间进行变换，变换完成后相当于对问题空间进行简化，原来线性不可解的问题现在变得可以解了。
    ```

    下图可以很形象的解释这个问题，左图用一根线是无法划分的。经过一系列变换后，就变成线性可解的问题了。

    ![](https://ss.csdn.net/p?https://mmbiz.qpic.cn/mmbiz_png/pu7ghYhibpS88nnX6ISHnwUWV8u74m6tGyp7ZYAZL0xL1rNwBcaic7J86JLXT5QjjZNmmNFZoKUiclYQCdh88zBcQ/640?wx_fmt=png)

    如果不用激励函数（其实相当于激励函数是f(x) = x），在这种情况下你每一层输出都是上层输入的线性函数，很容易验证，无论你神经网络有多少层，输出都是输入的线性组合，与没有隐藏层效果相当，这种情况就是最原始的感知机（Perceptron）了。

    正因为上面的原因，我们决定引入非线性函数作为激励函数，这样深层神经网络就有意义了（不再是输入的线性组合，可以逼近任意函数）。最早的想法是sigmoid函数或者tanh函数，输出有界，很容易充当下一层输入（以及一些人的生物解释）。

17. **请问人工神经网络中为什么ReLu要好过于tanh和sigmoid function？**

    先看sigmoid、tanh和RelU的函数图

    ![](https://ss.csdn.net/p?https://mmbiz.qpic.cn/mmbiz_png/pu7ghYhibpS88nnX6ISHnwUWV8u74m6tGhCYNQs2QeQ4cwon1iaOjJ3QP47pczd1umicfw0aRGibTJwGCs7uia2XZ3A/640?wx_fmt=png)

    第一，采用sigmoid等函数，算激活函数时（指数运算），计算量大，反向传播求误差梯度时，求导涉及除法和指数运算，计算量相对大，而采用Relu激活函数，整个过程的计算量节省很多。

    第二，对于深层网络，sigmoid函数反向传播时，很容易就会出现梯度消失的情况（在sigmoid接近饱和区时，变换太缓慢，导数趋于0，这种情况会造成信息丢失），这种现象称为饱和，从而无法完成深层网络的训练。而ReLU就不会有饱和倾向，不会有特别小的梯度出现。

    第三，Relu会使一部分神经元的输出为0，这样就造成了网络的稀疏性，并且减少了参数的相互依存关系，缓解了过拟合问题的发生（以及一些人的生物解释balabala）。当然现在也有一些对relu的改进，比如prelu，random relu等，在不同的数据集上会有一些训练速度上或者准确率上的改进，具体的大家可以找相关的paper看。

    多加一句，现在主流的做法，会多做一步batch normalization，尽可能保证每一层网络的输入具有相同的分布[1]。而最新的paper[2]，他们在加入bypass connection之后，发现改变batch normalization的位置会有更好的效果。大家有兴趣可以看下。

18. **为什么LSTM模型中既存在sigmoid又存在tanh两种激活函数，而不是选择统一一种sigmoid或者tanh？这样做的目的是什么？**

    ![](https://ss.csdn.net/p?https://mmbiz.qpic.cn/mmbiz_png/pu7ghYhibpS88nnX6ISHnwUWV8u74m6tGj6qclADW2xz4Nicd6qPpOZ91rmHOUhpeSia223huOJMYybxNfAhppqdQ/640?wx_fmt=png)

    sigmoid用在了各种gate上，产生0~1之间的值，这个一般只有sigmoid最直接了

    tanh用在了状态和输出上，是对数据的处理，这个用其他激活函数或许也可以。

    二者目的不一样

19. **如何解决RNN梯度爆炸和弥散的问题？**

    为了解决梯度爆炸问题，Thomas Mikolov首先提出了一个简单的启发性的解决方案，就是当梯度大于一定阈值的的时候，将它截断为一个较小的数。具体如算法1所述：

    算法：当梯度爆炸时截断梯度（伪代码）

    ![](https://ss.csdn.net/p?https://mmbiz.qpic.cn/mmbiz_jpg/pu7ghYhibpS88nnX6ISHnwUWV8u74m6tGFyqytElKQFpv6tzIKiaK2akW1Nib0Z9XZib9hUXdiaLw44p6vhg7q7ibUXw/640?wx_fmt=jpeg)

    下图可视化了梯度截断的效果。它展示了一个小的rnn（其中W为权值矩阵，b为bias项）的决策面。这个模型是一个一小段时间的rnn单元组成；实心箭头表明每步梯度下降的训练过程。当梯度下降过程中，模型的目标函数取得了较高的误差时，梯度将被送到远离决策面的位置。截断模型产生了一个虚线，它将误差梯度拉回到离原始梯度接近的位置。

    ![](https://ss.csdn.net/p?https://mmbiz.qpic.cn/mmbiz_png/pu7ghYhibpS88nnX6ISHnwUWV8u74m6tGnRzc9OnzF8JaZq3XnSiaiahMl2QjA8ktuhPk2UjGC0sF2786xrPaHX3Q/640?wx_fmt=png)

    梯度爆炸，梯度截断可视化 

    为了解决梯度弥散的问题，我们介绍了两种方法。

    第一种方法是将随机初始化![640?wx_fmt=jpeg](https://ss.csdn.net/p?https://mmbiz.qpic.cn/mmbiz_jpg/pu7ghYhibpS88nnX6ISHnwUWV8u74m6tGS1CJtZp4Kt1UZer9Tic2H4COoibJfhgQxjXVTfuvTH6Oho79Ic08IdRA/640?wx_fmt=jpeg)改为一个有关联的矩阵初始化。

    第二种方法是使用ReLU（Rectified Linear Units）代替sigmoid函数。ReLU的导数不是0就是1.因此，神经元的梯度将始终为1，而不会当梯度传播了一定时间之后变小。

20. **什麽样的数据集不适合用深度学习？**

    1）数据集太小，数据样本不足时，深度学习相对其它机器学习算法，没有明显优势。

    2）数据集没有局部相关特性，目前深度学习表现比较好的领域主要是图像／语音／自然语言处理等领域，这些领域的一个共性是局部相关性。图像中像素组成物体，语音信号中音位组合成单词，文本数据中单词组合成句子，这些特征元素的组合一旦被打乱，表示的含义同时也被改变。对于没有这样的局部相关性的数据集，不适于使用深度学习算法进行处理。举个例子：预测一个人的健康状况，相关的参数会有年龄、职业、收入、家庭状况等各种元素，将这些元素打乱，并不会影响相关的结果。

21. **广义线性模型是怎被应用在深度学习中？**

    A Statistical View of Deep Learning (I): Recursive GLMs

    深度学习从统计学角度，可以看做递归的广义线性模型。

    广义线性模型相对于经典的线性模型(y=wx+b)，核心在于引入了连接函数g(.)，形式变为：y=g−1(wx+b)。

    深度学习时递归的广义线性模型，神经元的激活函数，即为广义线性模型的链接函数。逻辑回归（广义线性模型的一种）的Logistic函数即为神经元激活函数中的Sigmoid函数，很多类似的方法在统计学和神经网络中的名称不一样，容易引起初学者（这里主要指我）的困惑。

    下图是一个对照表

    ![](https://ss.csdn.net/p?https://mmbiz.qpic.cn/mmbiz_png/pu7ghYhibpSibBsaR9xWNdic022dsmWOeFSRqlmPUmMiaOjiagIbkNrEK8pOnQmYMGS7Cla6XNQ2H7Siad209PuicsgIg/640?wx_fmt=png)

22. **如何解决梯度消失和梯度膨胀？**

    （1）梯度消失：

    根据链式法则，如果每一层神经元对上一层的输出的偏导乘上权重结果都小于1的话，那么即使这个结果是0.99，在经过足够多层传播之后，误差对输入层的偏导会趋于0

    可以采用ReLU激活函数有效的解决梯度消失的情况，也可以用Batch Normalization解决这个问题。

    （2）梯度膨胀

    根据链式法则，如果每一层神经元对上一层的输出的偏导乘上权重结果都大于1的话，在经过足够多层传播之后，误差对输入层的偏导会趋于无穷大

    可以通过激活函数来解决，或用Batch Normalization解决这个问题。

23. **简述神经网络的发展历史。**

    ```
    1949年Hebb提出了神经心理学学习范式——Hebbian学习理论
    1952年，IBM的Arthur Samuel写出了西洋棋程序
    1957年，Rosenblatt的感知器算法是第二个有着神经系统科学背景的机器学习模型.
    3年之后，Widrow因发明Delta学习规则而载入ML史册，该规则马上就很好的应用到了感知器的训练中
    感知器的热度在1969被Minskey一盆冷水泼灭了。他提出了著名的XOR问题，论证了感知器在类似XOR问题的线性不可分数据的无力。
    尽管BP的思想在70年代就被Linnainmaa以“自动微分的翻转模式”被提出来，但直到1981年才被Werbos应用到多层感知器(MLP)中，NN新的大繁荣。
    1991年的Hochreiter和2001年的Hochreiter的工作，都表明在使用BP算法时，NN单元饱和之后会发生梯度损失。又发生停滞。
    时间终于走到了当下，随着计算资源的增长和数据量的增长。一个新的NN领域——深度学习出现了。
    简言之，MP模型+sgn—->单层感知机（只能线性）+sgn— Minsky 低谷 —>多层感知机+BP+sigmoid—- (低谷) —>深度学习+pre-training+ReLU/sigmoid
    ```

24. **深度学习常用方法。**

    全连接DNN（相邻层相互连接、层内无连接）

    AutoEncoder(尽可能还原输入)、Sparse Coding（在AE上加入L1规范）、RBM（解决概率问题）—–>特征探测器——>栈式叠加 贪心训练 

    RBM—->DBN 

    解决全连接DNN的全连接问题—–>CNN 

    解决全连接DNN的无法对时间序列上变化进行建模的问题—–>RNN—解决时间轴上的梯度消失问题——->LSTM

    DNN是传统的全连接网络，可以用于广告点击率预估，推荐等。其使用embedding的方式将很多离散的特征编码到神经网络中，可以很大的提升结果。

    CNN主要用于计算机视觉(Computer Vision)领域，CNN的出现主要解决了DNN在图像领域中参数过多的问题。同时，CNN特有的卷积、池化、batch normalization、Inception、ResNet、DeepNet等一系列的发展也使得在分类、物体检测、人脸识别、图像分割等众多领域有了长足的进步。同时，CNN不仅在图像上应用很多，在自然语言处理上也颇有进展，现在已经有基于CNN的语言模型能够达到比LSTM更好的效果。在最新的AlphaZero中，CNN中的ResNet也是两种基本算法之一。

    GAN是一种应用在生成模型的训练方法，现在有很多在CV方面的应用，例如图像翻译，图像超清化、图像修复等等。

    RNN主要用于自然语言处理(Natural Language Processing)领域，用于处理序列到序列的问题。普通RNN会遇到梯度爆炸和梯度消失的问题。所以现在在NLP领域，一般会使用LSTM模型。在最近的机器翻译领域，Attention作为一种新的手段，也被引入进来。

    除了DNN、RNN和CNN外， 自动编码器(AutoEncoder)、稀疏编码(Sparse Coding)、深度信念网络(DBM)、限制玻尔兹曼机(RBM)也都有相应的研究。

25. **请简述过拟合还是激活函数？的发展史。**

    sigmoid会饱和，造成梯度消失。于是有了ReLU。

    ReLU负半轴是死区，造成梯度变0。于是有了LeakyReLU，PReLU。

    强调梯度和权值分布的稳定性，由此有了ELU，以及较新的SELU。

    太深了，梯度传不下去，于是有了highway。

    干脆连highway的参数都不要，直接变残差，于是有了ResNet。

    强行稳定参数的均值和方差，于是有了BatchNorm。

    在梯度流中增加噪声，于是有了 Dropout。

    RNN梯度不稳定，于是加几个通路和门控，于是有了LSTM。

    LSTM简化一下，有了GRU。

    GAN的JS散度有问题，会导致梯度消失或无效，于是有了WGAN。

    WGAN对梯度的clip有问题，于是有了WGAN-GP。

26. **神经网络中激活函数的真正意义？一个激活函数需要具有哪些必要的属性？还有哪些属性是好的属性但不必要的？**

    （1）非线性：即导数不是常数。这个条件是多层神经网络的基础，保证多层网络不退化成单层线性网络。这也是激活函数的意义所在。

    （2）几乎处处可微：可微性保证了在优化中梯度的可计算性。传统的激活函数如sigmoid等满足处处可微。对于分段线性函数比如ReLU，只满足几乎处处可微（即仅在有限个点处不可微）。对于SGD算法来说，由于几乎不可能收敛到梯度接近零的位置，有限的不可微点对于优化结果不会有很大影响[1]。

    （3）计算简单：非线性函数有很多。极端的说，一个多层神经网络也可以作为一个非线性函数，类似于Network In Network[2]中把它当做卷积操作的做法。但激活函数在神经网络前向的计算次数与神经元的个数成正比，因此简单的非线性函数自然更适合用作激活函数。这也是ReLU之流比其它使用Exp等操作的激活函数更受欢迎的其中一个原因。

    （4）非饱和性（saturation）：饱和指的是在某些区间梯度接近于零（即梯度消失），使得参数无法继续更新的问题。最经典的例子是Sigmoid，它的导数在x为比较大的正值和比较小的负值时都会接近于0。更极端的例子是阶跃函数，由于它在几乎所有位置的梯度都为0，因此处处饱和，无法作为激活函数。ReLU在x>0时导数恒为1，因此对于再大的正值也不会饱和。但同时对于x<0，其梯度恒为0，这时候它也会出现饱和的现象（在这种情况下通常称为dying ReLU）。Leaky ReLU[3]和PReLU[4]的提出正是为了解决这一问题。

    （5）单调性（monotonic）：即导数符号不变。这个性质大部分激活函数都有，除了诸如sin、cos等。个人理解，单调性使得在激活函数处的梯度方向不会经常改变，从而让训练更容易收敛。

    （6）输出范围有限：有限的输出范围使得网络对于一些比较大的输入也会比较稳定，这也是为什么早期的激活函数都以此类函数为主，如Sigmoid、TanH。但这导致了前面提到的梯度消失问题，而且强行让每一层的输出限制到固定范围会限制其表达能力。因此现在这类函数仅用于某些需要特定输出范围的场合，比如概率输出（此时loss函数中的log操作能够抵消其梯度消失的影响[1]）、LSTM里的gate函数。

    （7）接近恒等变换（identity）：即约等于x。这样的好处是使得输出的幅值不会随着深度的增加而发生显著的增加，从而使网络更为稳定，同时梯度也能够更容易地回传。这个与非线性是有点矛盾的，因此激活函数基本只是部分满足这个条件，比如TanH只在原点附近有线性区（在原点为0且在原点的导数为1），而ReLU只在x>0时为线性。这个性质也让初始化参数范围的推导更为简单。额外提一句，这种恒等变换的性质也被其他一些网络结构设计所借鉴，比如CNN中的ResNet[6]和RNN中的LSTM。

    （8）参数少：大部分激活函数都是没有参数的。像PReLU带单个参数会略微增加网络的大小。还有一个例外是Maxout[7]，尽管本身没有参数，但在同样输出通道数下k路Maxout需要的输入通道数是其它函数的k倍，这意味着神经元数目也需要变为k倍；但如果不考虑维持输出通道数的情况下，该激活函数又能将参数个数减少为原来的k倍。

    （9）归一化（normalization）：这个是最近才出来的概念，对应的激活函数是SELU[8]，主要思想是使样本分布自动归一化到零均值、单位方差的分布，从而稳定训练。在这之前，这种归一化的思想也被用于网络结构的设计，比如Batch Normalization[9]。

27. **梯度下降法的神经网络容易收敛到局部最优，为什么应用广泛？**

    深度神经网络“容易收敛到局部最优”，很可能是一种想象，实际情况是，我们可能从来没有找到过“局部最优”，更别说全局最优了。

    很多人都有一种看法，就是“局部最优是神经网络优化的主要难点”。这来源于一维优化问题的直观想象。在单变量的情形下，优化问题最直观的困难就是有很多局部极值，如

    ![](https://ss.csdn.net/p?https://mmbiz.qpic.cn/mmbiz_jpg/pu7ghYhibpS86e4frRmTJick1ZheWNiaWicQ6qa6fGaEWqSqjribqJl9F0oXqia9UHYRgy51TkQMicKttR8KNXuy1Tk5A/640?wx_fmt=jpeg)

    人们直观的想象，高维的时候这样的局部极值会更多，指数级的增加，于是优化到全局最优就更难了。然而单变量到多变量一个重要差异是，单变量的时候，Hessian矩阵只有一个特征值，于是无论这个特征值的符号正负，一个临界点都是局部极值。但是在多变量的时候，Hessian有多个不同的特征值，这时候各个特征值就可能会有更复杂的分布，如有正有负的不定型和有多个退化特征值（零特征值）的半定型

    ![](https://ss.csdn.net/p?https://mmbiz.qpic.cn/mmbiz_jpg/pu7ghYhibpS86e4frRmTJick1ZheWNiaWicQicB87LArQNZSu6WwatVOJhH3hOibmARW1ibETicgnibib5AdBflISfD9ibPoA/640?wx_fmt=jpeg)

    在后两种情况下，是很难找到局部极值的，更别说全局最优了。

    　　现在看来，神经网络的训练的困难主要是鞍点的问题。在实际中，我们很可能也从来没有真的遇到过局部极值。Bengio组这篇文章Eigenvalues of the Hessian in Deep Learning（https://arxiv.org/abs/1611.07476）里面的实验研究给出以下的结论：

    Training stops at a point that has a small gradient. The norm of the gradient is not zero, therefore it does not, technically speaking, converge to a critical point.

    There are still negative eigenvalues even when they are small in magnitude.

    另一方面，一个好消息是，即使有局部极值，具有较差的loss的局部极值的吸引域也是很小的Towards Understanding Generalization of Deep Learning: Perspective of Loss Landscapes。（https://arxiv.org/abs/1706.10239）

    For the landscape of loss function for deep networks, the volume of basin of attraction of good minima dominates over that of poor minima, which guarantees optimization methods with random initialization to converge to good minima.

    所以，很可能我们实际上是在“什么也没找到”的情况下就停止了训练，然后拿到测试集上试试，“咦，效果还不错”。

    补充说明，这些都是实验研究结果。理论方面，各种假设下，深度神经网络的Landscape 的鞍点数目指数增加，而具有较差loss的局部极值非常少。

28. **简单说说CNN常用的几个模型。**

    ![](https://ss.csdn.net/p?https://mmbiz.qpic.cn/mmbiz_png/pu7ghYhibpS86e4frRmTJick1ZheWNiaWicQxecAsXwXNoibNjDVibSmphoON3ODaw1kxIeUg9oKU00AxNj6R5H5qBzg/640?wx_fmt=png)

29. **为什么很多做人脸的Paper会最后加入一个Local Connected Conv？**

    以FaceBook DeepFace 为例：

    ![](https://ss.csdn.net/p?https://mmbiz.qpic.cn/mmbiz_jpg/pu7ghYhibpS86e4frRmTJick1ZheWNiaWicQJUVN2zicBibnsegBM0sc63Ntwu2RHmP7XwXIicbaD1nUNHyr9VhiceW8jw/640?wx_fmt=jpeg)

    DeepFace 先进行了两次全卷积＋一次池化，提取了低层次的边缘／纹理等特征。后接了3个Local-Conv层，这里是用Local-Conv的原因是，人脸在不同的区域存在不同的特征（眼睛／鼻子／嘴的分布位置相对固定），当不存在全局的局部特征分布时，Local-Conv更适合特征的提取。

30. **什么是梯度爆炸？**

    误差梯度是神经网络训练过程中计算的方向和数量，用于以正确的方向和合适的量更新网络权重。

    在深层网络或循环神经网络中，误差梯度可在更新中累积，变成非常大的梯度，然后导致网络权重的大幅更新，并因此使网络变得不稳定。在极端情况下，权重的值变得非常大，以至于溢出，导致 NaN 值。

    网络层之间的梯度（值大于 1.0）重复相乘导致的指数级增长会产生梯度爆炸。

31. **梯度爆炸会引发什么问题？**

    在深度多层感知机网络中，梯度爆炸会引起网络不稳定，最好的结果是无法从训练数据中学习，而最坏的结果是出现无法再更新的 NaN 权重值。

    梯度爆炸导致学习过程不稳定。—《深度学习》，2016。

    在循环神经网络中，梯度爆炸会导致网络不稳定，无法利用训练数据学习，最好的结果是网络无法学习长的输入序列数据。

32. **如何确定是否出现梯度爆炸？**

    训练过程中出现梯度爆炸会伴随一些细微的信号，如：

    ​	模型无法从训练数据中获得更新（如低损失）

    ​	模型不稳定，导致更新过程中的损失出现显著变化。

    ​	训练过程中，模型损失变成 NaN。

    ​	如果你发现这些问题，那么你需要仔细查看是否出现梯度爆炸问题。

    以下是一些稍微明显一点的信号，有助于确认是否出现梯度爆炸问题。

    ​	训练过程中模型梯度快速变大。

    ​	训练过程中模型权重变成 NaN 值。

    ​	训练过程中，每个节点和层的误差梯度值持续超过 1.0。

33. **如何修复梯度爆炸问题？**

    有很多方法可以解决梯度爆炸问题，本节列举了一些最佳实验方法。

    （1） 重新设计网络模型

    在深度神经网络中，梯度爆炸可以通过重新设计层数更少的网络来解决。

    使用更小的批尺寸对网络训练也有好处。

    在循环神经网络中，训练过程中在更少的先前时间步上进行更新（沿时间的截断反向传播，truncated Backpropagation through time）可以缓解梯度爆炸问题。

    （2）使用 ReLU 激活函数

    在深度多层感知机神经网络中，梯度爆炸的发生可能是因为激活函数，如之前很流行的 Sigmoid 和 Tanh 函数。

    使用 ReLU 激活函数可以减少梯度爆炸。采用 ReLU 激活函数是最适合隐藏层的新实践。

    （3）使用长短期记忆网络

    在循环神经网络中，梯度爆炸的发生可能是因为某种网络的训练本身就存在不稳定性，如随时间的反向传播本质上将循环网络转换成深度多层感知机神经网络。

    使用长短期记忆（LSTM）单元和相关的门类型神经元结构可以减少梯度爆炸问题。

    采用 LSTM 单元是适合循环神经网络的序列预测的最新最好实践。

    （4）使用梯度截断（Gradient Clipping）

    在非常深且批尺寸较大的多层感知机网络和输入序列较长的 LSTM 中，仍然有可能出现梯度爆炸。如果梯度爆炸仍然出现，你可以在训练过程中检查和限制梯度的大小。这就是梯度截断。
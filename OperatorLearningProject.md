## 偏微分赛道




### 赛题题目

#### 第一届赛题

**10维泊松方程 (Poisson equation)**


要求解 $u \in L^2(\Omega)$，满足

$$
\begin{cases}
-\Delta u(x) = f(x), & x \in \Omega, \\[6pt]
u(x) = 0, & x \in \partial \Omega,
\end{cases}
$$

其中区域为

$$
\Omega = [-1,1]^d,
$$

右端项定义为

$$
f(x) = -\, d \, (4 \pi)^2 \, \prod_{i=1}^d \sin(4 \pi x_i), 
\quad x = (x_1, x_2, \dots, x_d) \in \Omega.
$$

赛题需要选手分别求出 $d=2$ 和 $d=10$ 时的数值解，并提交代码和结果。

1. **维度设定**

   * 当 $d = 2$ 时，可以用传统有限元 / 有限差分方法直接数值解。

   * 当 $d = 10$ 时，传统数值方法会遭遇 **维度灾难**，计算量急剧增加，难以求解。

2. **边界条件**

   * 在边界 $\partial \Omega$ 上，解满足齐次 Dirichlet 边界条件：

     $$
     u(x) = 0, \quad x \in \partial \Omega.
     $$

3. **问题难点**

   * 在高维情况下（如 $d=10$），传统网格法需要的自由度随维数指数增长，不切实际。

   * 因此，需要 **Monte Carlo 方法** 或 **基于深度学习的 PINNs 方法** 来求解。




##### 1. 随机方法（Monte Carlo / 随机游走）

* **思想**：利用泊松方程的 **Feynman–Kac 表示**，解可以写成布朗运动路径积分的期望。

* **数值方法**：从初始点 $x$ 发射大量随机游走，直到命中边界，积累源项 $f$ 的积分，再取平均。

* **优点**：维度不敏感，在 10 维也能运行。

* **缺点**：收敛速度慢（方差较大），需要很多随机样本。


##### 2. 深度学习方法（PINNs：Physics-Informed Neural Networks）

* **思想**：用一个神经网络 $u_\theta(x)$ 近似解，直接在 **10 维输入空间**上训练。

* **损失函数**：

  $$
  \mathcal{L}(\theta) = \mathbb{E}_{x \in \Omega} \left| -\Delta u_\theta(x) - f(x) \right|^2 
  + \mathbb{E}_{x \in \partial\Omega} \left| u_\theta(x) \right|^2
  $$

* **实现方式**：

  1. 用 PyTorch / TensorFlow 定义一个 MLP（输入维度 = 10，输出 = 1）。

  2. 在区域内部随机采样点（collocation points）计算 PDE 残差。

  3. 在边界随机采样点 enforcing 边界条件。

  4. 反向传播更新参数 $\theta$。

* **优点**：在高维 PDE 中非常有效（“打破维度灾难”）。

* **缺点**：训练时间长，结果依赖优化和网络结构。

> 推荐用 **PINNs**，这是目前高维 PDE 的主流方法。

去年一共有30%的参赛队伍选择此题，一支队伍最终获得特等奖。


#### 第二届赛题一(逆向)

**散射场反问题与点源定位**

我们考虑二维亥姆霍兹方程 (Helmholtz equation) 的点源散射问题。其在边界 $\partial \Omega$ 上的解可以表示为

$$
u(x) = -\frac{i}{4}\sum_{j=1}^m \lambda_j H_0^{(1)} \big( k|x-z_j| \big), 
\quad x \in \partial \Omega,
$$

其中：

* $H_0^{(1)}(\cdot)$：0阶第一类 Hankel 函数；
* $z_j$：未知点源的位置；
* $\lambda_j$：点源强度；
* $m$：点源个数（已知不超过4个，部分情况下固定为2个）。

**任务目标**：
已知边界 $\partial \Omega$ 上的有限角度测量数据 $u(x)$，反推出点源的：

1. 数量 $m$，
2. 位置 $\{z_j\}$，
3. 强度 $\{\lambda_j\}$。

**数据条件**：

* 测量数据有限，且只能在边界部分区域获得；
* 点源强度 $\lambda_j$ 在小范围内波动，可以看作随机变量；
* 数据生成中 Hankel 函数可通过 Matlab `besselh(0,1,x)` 计算。

这是一个 **PDE 反问题**，同时也是一个 **不适定问题**：有限测量数据下，解的唯一性与稳定性具有挑战。







**解反源问题的深度学习方法**

1. 数据驱动 (Supervised Learning)

* **思路**：用大量模拟数据训练神经网络，从边界散射场 $u(x)$ 预测点源参数 $(m, \{z_j\}, \{\lambda_j\})$。
* **步骤**：

  1. 随机生成点源位置和强度；
  2. 利用 Hankel 函数计算散射场数据；
  3. 构建训练集 $(u(x), (m, z_j, \lambda_j))$；
  4. 用 CNN/MLP/Transformer 进行回归，输出点源参数。


2. 物理约束神经网络 (PINNs)

* **思路**：利用 PDE 本身约束网络训练过程，减少对大规模标注数据的依赖。
* **方法**：

  * 构造神经网络 $u_\theta(x)$ 近似解散射场；
  * 约束其满足亥姆霍兹方程：

    $$
    (\Delta + k^2) u_\theta(x) = 0, \quad x \in \Omega \setminus \{z_j\}
    $$
  * 在边界 $\partial \Omega$ 上匹配观测数据 $u(x)$；
  * 点源位置 $\{z_j\}$ 与强度 $\{\lambda_j\}$ 作为待优化的隐参数。

这样可通过最小化物理残差 + 数据拟合误差，得到点源参数。


3. 混合方法 (Deep Inverse + Optimization)

* **先用深度学习网络** 粗略预测点源位置、数量和强度；
* **再用物理模型约束优化**（如最小二乘拟合散射场）做精修；
* 好处是既能保证速度，又能保证物理一致性。


![image-3.png](attachment:image-3.png)






#### 第二届赛题二(正向 )

**无粘 Burgers 方程**

$$
\frac{\partial u}{\partial t} + \frac{\partial}{\partial x}\!\left(\tfrac{1}{2} u^2 \right) = 0, 
\quad u = u(x,t).
$$

其中：

* $x$：空间变量
* $t$：时间变量
* $u(x,t)$：物理量（速度/密度）

它本质上是一个 **非线性守恒律方程**，会出现 **激波（shock）**。


##### 方法一：函数学习类神经网络方法


> 我们学习并尝试这两者即可

**随机特征方法 (RFM, 2022)** 和 **深度有限体积法 (DFVM, 2023)**


**1. 随机特征方法 (RFM, 2022)**

* **核心思想**
  利用随机特征（Random Features）近似核函数，把 PDE 的解函数映射到一个低维的随机特征空间里，再在这个空间中进行近似求解。
  形式上，它结合了：

  * **核方法 / 高维函数逼近能力**

  * **数值 PDE 的约束条件**

* **特点**

  * 在处理高维 PDE 时比传统网格法更高效（因为随机特征避免了指数级网格增长）。

  * 通过采样和投影把 PDE 转化为一个有限维的优化问题。

  * 可以视为 **数值方法与核回归的桥梁**。

* **应用场景**
  特别适合高维 PDE（如金融数学、量子力学、统计物理中的高维方程）。


**2. 深度有限体积法 (DFVM, 2023)**

* **核心思想**
  把 PDE 的 **守恒律（local conservation law）** 显式编码到深度学习框架中。

  * 类似传统的有限体积法 (Finite Volume Method, FVM)，对 PDE 做积分形式处理。

  * 但在近似解函数时引入深度神经网络，并在损失函数中强制 **局部守恒**。

* **特点**

  * 保证了 PDE 的物理结构（守恒性），避免出现违反物理规律的解。

  * 相比 PINNs（Physics-Informed Neural Networks），收敛性和稳定性更好。

  * 可以看作 **深度学习版本的有限体积法**。

* **应用场景**
  特别适合 **流体力学、传输问题、守恒律 PDE** 等物理背景。


| 方法          | 主要思路              | 优势            | 局限           |
| ----------- | ----------------- | ------------- | ------------ |
| RFM (2022)  | 随机特征空间近似 PDE 解    | 高维问题友好；核方法优势  | 守恒/物理结构不明显   |
| DFVM (2023) | 深度学习 + 有限体积法，显式守恒 | 保留物理结构，数值稳定性强 | 在高维问题上可能效率不足 |


##### 方法二：算子学习类神经网络方法

![image.png](attachment:image.png)


> [Lu et al., 2021] Lu Lu, Pengzhan Jin,et al.,  Learning nonlinear operators via deeponet based on the universal approximation theorem of operators. Nature machine intelligence, 3(3):218–229, 2021.

> 我们着重学习运用这类方法

![image-2.png](attachment:image-2.png)

> 需要学习的有

**神经网络方法族（函数逼近类）**

这些方法主要利用 **神经网络的表达能力** 来学习 PDE 解或算子映射关系：

1. **FNO (Fourier Neural Operator, 2020)**

   * 基于傅里叶变换，将算子学习任务转化到频域中。

   * 优势：对高维 PDE 泛化能力强，适合复杂物理系统。

2. **DeepONet (Deep Operator Network, 2021)**

   * 神经算子方法之一，学习从函数到函数的映射（算子学习）。

   * 优势：能处理广泛的 PDE 输入输出映射问题。

3. **U-Net (2015)**

   * 卷积神经网络结构，最初用于图像分割。

   * 在 PDE 中可作为空间映射的 backbone，处理局部特征。

4. **U-NO (U-shaped Neural Operator, 2022)**

   * 将 U-Net 结构与神经算子方法结合。

   * 优势：既能捕捉全局特征，也能保留局部结构。

5. **MPNN (Message Passing Neural Network PDE Solver, 2022)**

   * 消息传递图神经网络，用于 PDE 的数值解。

   * 优势：适合处理非规则网格和图结构上的 PDE。

6. **KNO (Koopman Neural Operator, 2023)**

   * 基于 Koopman 算子理论，学习非线性动力系统的线性表示。

   * 优势：适合动力学系统的长期预测。



**结合物理结构的方法（物理守恒 / 数值结构约束）**

这些方法不仅追求拟合 PDE，还强调 **保物理结构与数值稳定性**：

1. **AI Poincaré 2.0 \[1]**

   * 结合神经网络学习与流形学习，自动识别系统的守恒量。

   * 强调物理对称性和守恒律的发现。


2. **CFN (Conservative Form Network) \[2]**

   * 从守恒律出发，构造保守型神经网络。

   * 优势：能保证 PDE 的局部守恒性质。

3. **Physics-informed finite-volume scheme \[3]**

   * 数据驱动 + 物理约束，基于有限体积方法的混合方案。

   * 优势：对非经典激波 (underccompressive shocks) 仍有良好表现。

4. **RoeNet \[4]**

   * 基于 Roe 求解器和伪逆嵌入，学习超曲面系统的间断性。

   * 优势：对激波、间断点的处理效果突出。


**整体总结：**

* **算子学习类 (FNO, DeepONet, U-NO, KNO)**：强调端到端学习算子映射，适合高维、复杂 PDE。

* **结构保留类 (CFN, RoeNet, AI Poincaré 2.0, physics-informed FV)**：在网络设计/损失函数中显式保物理结构，更可靠。

* **网络架构类 (U-Net, MPNN)**：为 PDE 学习提供不同的空间或图结构特征提取方式。

> [1] Liu Z, Madhavan V, Tegmark M. Machine learning conservation laws from differential equations[J]. Physical Review E, 2022, 106(4): 045307.

> [2] Chen Z, Gelb A, Lee Y. Learning the Dynamics for Unknown Hyperbolic Conservation Laws Using Deep Neural Networks[J]. SIAM Journal on Scientific Computing, 2024, 46(2): A825-A850.

> [3] Bezgin D A, Schmidt S J, Adams N A. A data-driven physics-informed finite-volume scheme for nonclassical undercompressive shocks[J]. Journal of Computational Physics, 2021, 437: 110324.

> [4] Tong Y, Xiong S, He X, et al. RoeNet: Predicting discontinuity of hyperbolic systems from continuous data[J]. International Journal for Numerical Methods in Engineering, 2024, 125(6): e7406.


##### 注意

> 一般来说对于跟时间演化有关的方程，时间是有方向的，有的人喜欢用经典方法离散到一个点，然后用深度学习方法去做。

**基于时间离散的递推型方法**

* 思路：把 PDE 变成时间推进问题。
* 方法：

  1. 在空间上离散（比如有限差分、谱方法）。

  2. 在时间维度选取离散点 $t_0, t_1, \dots, t_N$。

  3. 用神经网络（RNN、Transformer、MLP）学一个 **时间推进算子**：

     $$
     u^{n+1} = \mathcal{N}_\theta(u^n, \Delta t)
     $$
  4. 这样模型学到的就是一个“黑箱时间积分器”。

* 优点：直观、和数值方法类似。

* 缺点：时间步长受限，误差会累积。



> 直接用pinns解与时间有关的方程，一般不是很好，需要额外的技术处理。


**PINNs（Physics-Informed Neural Networks）直接建模时空解**

* 思路：直接用一个神经网络 $u_\theta(x,t)$ 同时表示空间和时间上的解。

* 方法：

  1. 输入：$(x,t)$，输出：$u(x,t)$。

  2. 构造 PDE 残差损失：


     $$
     \mathcal{L}_\text{PDE} = \bigg| \frac{\partial u_\theta}{\partial t} + \frac{\partial}{\partial x}\!\left(\tfrac{1}{2} u_\theta^2 \right) \bigg|^2
     $$

  3. 加上初始条件、边界条件损失：

     $$
     \mathcal{L} = \mathcal{L}_\text{PDE} + \mathcal{L}_\text{IC} + \mathcal{L}_\text{BC}.
     $$

  4. 通过反向传播同时优化整个时空区域的解。

* 优点：不会累积时间误差，可以全局求解。

* 缺点：训练代价大，可能收敛困难。


**算子学习（Neural Operators）**

* **FNO (Fourier Neural Operator)**、**DeepONet**

* 目标：学一个 **从输入函数（初始条件）到解函数（时空解）** 的映射：

  $$
  \mathcal{G}: u(x,0) \mapsto u(x,t).
  $$

* 特点：一次训练，多次泛化，可以解决一类 PDE，而不仅是一个特定初值。

* 在 Burgers 方程、Navier–Stokes 方程上已经有成功应用。

**总结：**

* **时间离散递推法** → 适合短时预测，偏工程。

* **PINNs 全局法** → 适合科研、直接求解 PDE。

* **算子学习 (FNO, DeepONet)** → 适合大规模 PDE 泛化。

### 准备策略


#### 按照如下路径补充数学知识

![4c0ef0b18dc804ed5b25ab2b257e7170.png](attachment:4c0ef0b18dc804ed5b25ab2b257e7170.png)


\**核心内容：*\*为了理解算子学习求解PDE，需要补充泛函分析、Sobolev空间、变分法和数值PDE等背景。重点包括：度量与赋范空间、Hilbert空间投影定理，Sobolev空间 \$W^{k,p}\$ 定义及嵌入定理，PDE弱解理论（Lax-Milgram 定理等）、变分形式，以及有限差分/有限元的基本思想。

\**学习资源：*\*推荐阅读经典教材和公开课程：

* *Functional Analysis*（功能分析）教材（如 Kreyszig 或 Brezis），学习巴拿赫/希尔伯特空间、算子和谱理论基础。
* *Sobolev Spaces*（Sobolev空间）专题资料，如 Evans《偏微分方程》前几章和附录，对Sobolev空间及弱导数、Poincaré不等式有详细讲解。
* *Calculus of Variations*（变分法）基础，可参考《变分法及泛函分析引论》等教材，理解泛函极值问题和Euler-Lagrange方程。
* *数值PDE*：例如Larsson & Thomée《Partial Differential Equations with Numerical Methods》或 Quarteroni的数值分析教材，了解有限差分、有限元离散求解PDE的收敛性与稳定性分析。理解传统数值方法有助于对比AI方法的优劣。

\*\*目标：\*\*通过本阶段，您应掌握：Sobolev空间和弱解概念（理解PDE解空间的函数空间性质）、典型PDE（椭圆型、抛物型、双曲型）的理论和数值解法。这将为后续理解“算子”（即从函数到函数的映射）奠定数学基础。


#### 强化对深度学习的理解

\*\*核心内容：\*\*进一步钻研深度学习的理论基础，尤其是与函数逼近和泛化相关的内容。重点包括：

* **神经网络逼近理论：**深入理解**万能逼近定理**的证明思路和局限。例如，Cybenko(1989)、Hornik(1989)关于单隐层网络逼近任意连续函数的定理，以及进一步的结果如Barron空间（Barron 1993）对高维函数逼近的误差界。如果可能，研读Chen & Chen (1995)关于算子的万能逼近定理，这一定理指出\*“单隐层神经网络可以精确逼近任意非线性连续算子”\*（即从函数到函数的映射）。这一结果为后续算子学习网络（如DeepONet）的结构设计提供了理论基础。
* **深度学习的泛化理论：**学习近年的研究，例如过参数化下网络的泛化性质、神经切核(NTK)理论、梯度下降的收敛分析等。可关注2020年后兴起的深度学习理论，如动态系统观点、幅度-范数对泛化的影响等。不过鉴于您关注PDE问题，这部分可择要了解**统计学习理论**如何延伸到无限维情形。
* \*\*其他前沿理论：\*\*如深度网络的表示能力（深度优于宽度的定理）、符号性回归与自适应启发式等。

\*\*学习资源：\*\*推荐以下进阶资料：

* *Deep Learning Theory* 课程讲义（如 MIT 18.408 Theoretical Foundations for DL 或 Matus Telgarsky 的讲义），其中包含对万能逼近定理、Barron定理、深度与宽度等理论的讨论。
* Nadav Dym (2023) 《Deep Learning and Approximation Theory》课程笔记，专门探讨深度学习的逼近理论。
* Poggio等人在 *PNAS* 等刊物上的文章，讨论梯度训练下网络的泛化误差界。
* 若对形式化证明有兴趣，可考虑将Lean 4用于验证部分理论结果（例如验证简单网络对连续算子的逼近性质），将您的逻辑推理优势结合到这一阶段的学习中。

**目标：**通过深入研究这些理论，您将对**神经网络为什么能够高效逼近复杂函数/算子**有更系统的理解。这为后续理解算子学习理论文献中的**逼近误差分析**和**泛化误差分析**做好准备，也可以激发您从理论上研究“AI求解PDE”的信心。


#### 算子学习理论基础

**核心内容：**正式进入**算子学习**（Operator Learning）领域的理论与方法学习。算子学习关注**学习函数空间到函数空间的映射**，即**PDE解算子的近似**。这一领域涌现出多种新架构和理论成果，需要系统学习：

* **算子学习概念：**理解什么是“神经算子”（Neural Operator）。Kovachki et al (2023) 给出了广义定义：*“神经算子是对传统神经网络的推广，用于学习函数空间到函数空间的映射”*。神经算子通常由**一系列线性积分算子与非线性激活的组合**构成，并且**与离散化无关**，可以在不同网格上共享同一组参数。重要的是，Kovachki等证明了**神经算子的万能逼近定理**，保证其能以任意精度逼近给定的连续非线性算子。

* \*\*经典神经算子架构：\*\*重点研习以下方法：

  * **DeepONet（深度算子网络）：**由Lu等人提出的架构。DeepONet包含**Branch网络**和**Trunk网络**两部分：Branch网提取输入函数的特征，Trunk网处理输出位置，然后通过点积融合两者以输出目标函数值。DeepONet具有**算子万能逼近**能力并在大量算例中成功学习显式算子（如积分算子、分数阶拉普拉斯算子）以及隐式算子（PDE解映射）。建议阅读Lu et al. (Nature Mach. Intell. 2021)原始论文及其附录理论证明。此外，Lanthaler et al. (2022)针对DeepONet提供了严谨的误差分析，可加深对其泛化能力的理解。
  * **Fourier Neural Operator（FNO，傅里叶神经算子）：**Li等人提出的架构，将PDE算子的核函数在傅里叶空间参数化。FNO通过对中间表示做FFT、截断高频、线性变换再逆FFT，实现**全局卷积**效应，因而对输入函数分辨率不敏感（具有**离散尺度不变性**）。FNO在Burgers方程、Darcy流和Navier-Stokes等基准上表现优异，**首次用机器学习成功模拟了湍流流体，并实现零样本超分辨率重建**。它的推理速度比传统求解器快几个数量级，同时在固定网格下精度优于先前的数据驱动方法。建议研读Li et al. (2020 NeurIPS)论文以及Kovachki et al. (2021)对FNO的改进分析。
  * **PCA/POD Neural Operator（主成分/正交分解算子网络）：**如 Bhattacharya et al. (2021) 提出的 PCA-Net/POD-Net 方法，利用对输出（或输入）函数空间做PCA降维，然后训练网络在低维子空间中映射。这种方法对**低秩**的问题（即解依赖于有限主成分）很有效，可提高训练效率。

  上述三类方法在近期综述中都有介绍，它们代表了算子学习的主流方向。特别地，**DeepONet**和**FNO**分别代表两大流派：一个基于**局部传感+全局融合**（branch-trunk架构），一个基于**谱域全局卷积**。理解它们的结构和理论有助于把握算子学习其它变种（如基于图神经网络的算子、基于低秩分解的算子等）。

* \*\*关键理论问题：\*\*学习这些模型的同时，请关注其背后的理论挑战：

  * **离散化不变性：**算子学习模型应能泛化到不同网格分辨率上。Kovachki等人在理论上指出，如果模型参数依赖于输入函数的固定离散取样，会破坏离散不变性。原始DeepONet在输入依赖固定传感点时存在这种不足，但后续有方法通过随机取样、POD基底等增强其不变性。理解这一点对设计**稳健**的算子网络至关重要。
  * **万能逼近与误差分析：**深入研读Chen & Chen (1995)的算子UAT定理，以及近年来针对DeepONet/FNO的**误差上界**推导（例如Lanthaler 2022对DeepONet逼近误差与Branch/Trunk宽度的关系分析）。这些理论让您了解提高网络宽度/深度如何降低逼近误差，怎样的函数空间特性有利于神经算子高效学习等。
  * **泛化能力：**算子学习的**泛化**指训练网络能否对**未见过的输入函数**给出精确解。当训练和测试的输入分布不同（distribution shift）时，网络表现如何？Lu等人在DeepONet论文中针对不同输入参数化方法研究了对泛化误差的影响。建议阅读相关分析，并关注近年一些理论工作（如Mi et al 2022）关于**算子学习泛化误差随训练样本分布的定量分析**。

\*\*学习资源：\*\*除了上述论文，还建议：

* 阅读 *A Practical Introduction to Neural Operators in Scientific Computing* (Prashant Jha, 2025)。该53页综述对DeepONet、PCA-Net、FNO做了实践导向的讲解，并比较了它们在Poisson方程、线弹性问题上的表现。结论部分还讨论了算子学习面临的精度控制和泛化挑战及相应对策。
* Karniadakis教授团队的资料：例如 NVIDIA DLI推出的“科学与工程深度学习”教学套件（由Karniadakis参与开发），涵盖PINNs和DeepONet求解PDE的课程，可以通过案例学习将物理知识融入网络的方法。
* 关注相关领域的开源代码库，例如**DeepXDE** (Deep Learning for Differential Equations) 或 Karniadakis团队开发的**Phylanx**、以及近期出现的算子学习框架 **continuiti**（一个支持多种神经算子架构、物理约束损失和示例的Python包）。阅读这些项目的文档和源码，有助于理解实现细节，并可作为您动手实验的平台。

**目标：**阶段4结束时，您应能**清晰阐释主要神经算子模型的结构和原理**，理解它们各自的优势局限，并掌握文献中提出的**理论保证**（如万能逼近、误差界）背后的证明思路。这将使您有能力**阅读前沿论文**并着手设计改进您自己感兴趣的模型。

&#x20;*DeepONet 算法架构示意图：包含 Branch 和 Trunk 两个子网络。Branch网接收输入函数在若干观测点的取值，Trunk网接收输出位置 \$y\$，二者输出通过一次线性组合（点积）得到算子 \$G(u)\$ 在 \$y\$ 处的预测值。通过这样的设计，DeepONet能够高效地学习从输入函数 \$u(x)\$ 到输出函数 \$G(u)(y)\$ 的映射。*

#### 阅读拆解的论文，学习重要的方法

1. **1. 随机特征方法 (RFM, 2022)**

2. **2. 深度有限体积法 (DFVM, 2023)**

3. 

> ![image.png](attachment:image.png)

4. 论文

> [1] Liu Z, Madhavan V, Tegmark M. Machine learning conservation laws from differential equations[J]. Physical Review E, 2022, 106(4): 045307.

> [2] Chen Z, Gelb A, Lee Y. Learning the Dynamics for Unknown Hyperbolic Conservation Laws Using Deep Neural Networks[J]. SIAM Journal on Scientific Computing, 2024, 46(2): A825-A850.

> [3] Bezgin D A, Schmidt S J, Adams N A. A data-driven physics-informed finite-volume scheme for nonclassical undercompressive shocks[J]. Journal of Computational Physics, 2021, 437: 110324.

> [4] Tong Y, Xiong S, He X, et al. RoeNet: Predicting discontinuity of hyperbolic systems from continuous data[J]. International Journal for Numerical Methods in Engineering, 2024, 125(6): e7406.

> [Lu et al., 2021] Lu Lu, Pengzhan Jin,et al.,  Learning nonlinear operators via deeponet based on the universal approximation theorem of operators. Nature machine intelligence, 3(3):218–229, 2021.

5. 更多文献


1. Kovachki et al. *Neural Operator: Learning Maps Between Function Spaces*. JMLR, 2023

2. Lu et al. *Learning nonlinear operators via DeepONet (Deep Operator Network)*. Nature Mach. Intell., 3, 218–229, 2021

3. Li et al. *Fourier Neural Operator for Parametric Partial Differential Equations*. NeurIPS 2020

4. Jha, P. *From Theory to Application: A Practical Introduction to Neural Operators in Scientific Computing*. arXiv 2503.05598, 2025

5. Raissi, M. et al. *Physics-Informed Neural Networks: A Deep Learning Framework for Solving PDEs*. J. Comput. Phys., 378, 686–707, 2019

6. Wang, S. et al. *Learning the solution operator of parametric PDEs with physics-informed DeepONets*. arXiv:2103.10974, 2021

7. Aldirany, Z. et al. *Multi-level neural networks for accurate solutions of BVPs*. Comput. Meth. Appl. Mech. Eng., 419, 116666, 2024

8. Ainsworth, M., Dong, J. *Galerkin neural networks: A framework for approximating variational equations with error control*. SIAM SISC, 43(5), A2474–A2501, 2021

9. Cao, S. et al. *Residual-based error correction for neural operators*. arXiv:2303.00049, 2023

10. Lanthaler, S. et al. *Error estimates for DeepONet*. arXiv:2107.05102, 2021.

11. Chen, T., Chen, H. *Universal approximation to nonlinear operators by neural networks with arbitrary activation functions*. IEEE Trans. Neural Networks, 6(4):911–917, 1995.


#### 学习建议

1. **夯实基础，寻找兴趣切入点：**首先确保完成I部分自学路径的主要阶段，使自己对**PDE理论**和**深度学习**都有扎实理解。在此过程中留意哪些问题最引起您的兴趣——是理论证明（如算子逼近定理）、还是算法实现（如某网络架构的巧妙之处），亦或某应用领域（如流体、材料）。“兴趣驱动”对于科研至关重要。比如，如果您对**泛函分析**背景下的神经网络理论着迷，不妨侧重理论课题；如果对**编程实现**更感兴趣，可偏向算法改进方向。

2. **选择“小切口”作为入门课题：**在正式开展研究时，从一个具体且**可驾驭的小问题**入手。根据前文小切口示例，建议如下：

   * **算子学习方法对比分析（实践课题）**：选择一类简单PDE（如热传导方程），比较不同AI方法对其求解的效果。具体可以是：训练DeepONet来学习其解算子，同时用PINN直接求解，并对比二者在训练数据需求、误差随时间演化等方面的表现。通过这一过程，您将亲身了解不同方法的优势劣势，为日后改进方法打下基础。
   * **DeepONet/FNO 理论分析（理论课题）**：沿着算子学习的理论问题，挑选一个切入。例如研究\*\*“DeepONet在一种特定Sobolev空间上的逼近能力”\*\*。尝试证明或推导：若PDE解算子映射 \$G: X \to Y\$ 在某种光滑性条件下成立，DeepONet需要多少宽度/深度才能将其逼近到误差\$\epsilon\$。这个课题难度适中，可参考已有万能逼近定理并加以扩展，是把您的数学功底用于前沿问题的绝佳练习。
   * **网络架构改进（综合课题）**：如果您对实现和理论都有兴趣，可尝试改进现有网络。例如，设计**混合架构**将FNO的频域优点和DeepONet的灵活性结合，或者在网络中嵌入已知物理信息（如对称性、不变量）以提高模型性能。这种课题需要您阅读最新文献、构思新方法并验证，可从小规模实验做起，不断迭代改进。

3. \*\*争取导师和团队支持：\*\*算子学习是新兴交叉领域，很适合团队合作。如果条件允许，联系对此方向有兴趣的导师或研究小组参与讨论。在交叉学科团队中，您的纯数学背景将提供独特视角。有经验的导师也能帮助您把握课题难度、避免走入研究“雷区”。您可以在团队项目中承担理论分析部分，同时学习他人长处。

4. **渐进拓展，瞄准“大切口”：**在小课题取得阶段性成果后，逐步将视野拓展到**AI求解PDE的大图景**。例如，如果您前期研究了算子网络的误差，那么下一步可以考虑**将误差控制融入更复杂的应用**（如在湍流预测中实现精度保证）。最终，您可以尝试构思\*\*“大切口”课题\*\*：比如，“能否设计一个统一框架，将神经算子与PINNs融合，既具备泛化能力又无需大量数据？” 这样的宏大问题需要长期攻关，但提前思考有助于规划研究方向。

5. **持续学习与跟进行业动态：**保持对该领域最新进展的关注。定期阅读顶会（NeurIPS, ICML, ICLR）和主要期刊（SINUM, JCP 等）上的相关论文。加入相关学术论坛或研讨会（如SIAM CSE、AI for Science研讨）以获取灵感。利用Lean 4的特长，您甚至可以参与**验证AI模型可靠性**的前沿工作（例如，有团队在尝试用形式方法验证神经网络的性质）。总之，保持学习热情和好奇心，将使您在这一快速发展的领域站稳脚跟。

最后，请记住：**理论研究**往往需要耐心和反复试验。在这一征途中，循序渐进、学研结合是成功的关键。



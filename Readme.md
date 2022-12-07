# Readme

使用Q-Learning算法控制智能体走迷宫。

## 运行方法

运行命令：

``` shell
python3 main.py
```

## 迷宫场景

我们使用以下图片作为迷宫场景：

![](E:\Python_Project\SelfLearning\Deep_Reinforcement_Learning\Q-Learning-Maze\image_input\maze_image.png)

黄色部分为智能体能够到达的部分，蓝色部分为迷宫中的障碍部分（即不能到达）。SP和EP分别表示起点和终点。

经过Matlab分析，该原始迷宫图片分辨率为1004$\times$529，而蓝色部分的像素高度近似为25，宽度近似为50，因此可合理认为原始迷宫图片由20$\times$20个单元格组成，每个单元格的分辨率为25$\times$50。

为了确保最终结果的精确度，将原始图片利用OpenCV和Numpy进行重构，可得到以下标准迷宫图片：

![](E:\Python_Project\SelfLearning\Deep_Reinforcement_Learning\Q-Learning-Maze\image_input\maze_input.png)

## 蒙特卡洛方法

强化学习中一条完整的序列指智能体agent从开始状态连续跳转到结束状态，而为了充分训练智能体agent，要求智能体agent完整地执行多个序列。

以计算在$s_t$状态下采取动作决策$a_t$能获得的累计折扣奖励$G_t$的期望（即动作价值函数$Q_{\pi}(s_t, a_t)$）为例，将序列的个数记为$N$，当采样的序列足够多时，$N$足够大，**动作价值函数$Q_{\pi}(s_t, a_t)$的值**越近似于**随机采样的多条序列的累计折扣奖励$G_t$的期望**，即：
$$
Q_{\pi}(s_t,a_t) = \mathbb{E}[G_t|S_t=s_t,A_t=a_t] \approx \frac{1}{N} \sum^{N}_{i=1} G_t^{(i)}
$$
在每一轮序列结束后，计算该轮序列的累计折扣奖励$G_t$；当所有序列结束后，计算累计折扣奖励$G_t$的期望并近似为动作价值函数$Q_{\pi}(s_t, a_t)$。

除计算累计折扣奖励$G_t$总和并计算整体平均这一算法思路外，还有一种单步增量式更新动作价值函数的算法思路：
$$
N \leftarrow N+1
$$

$$
Q_{\pi}(s_t,a_t) \leftarrow Q_{\pi}(s_t,a_t) + \frac{1}{N}[G_t^{(i)} - Q_{\pi}(s_t,a_t)]
$$

使用该思路在每轮序列结束时就能利用累计折扣奖励$G_t$更新动作价值函数$Q_{\pi}(s_t, a_t)$。

## Temporal Difference

时序差分（Temporal Difference）算法是Value-based Learning中常见的算法，根据个人理解，**其思想源于蒙特卡洛算法的价值函数期望单步增量式更新算法思路**。

将蒙特卡洛方法中的采样序列的个数$N$替换为$\alpha$，表示动作价值函数$Q_{\pi}(s_t, a_t)$更新步长（定值）；将累计折扣奖励$G_t$替换为累计折扣奖励$G_t$的期望，即动作价值函数$Q_{\pi}(s_t, a_t)$，即：
$$
Q_{\pi}(s_t,a_t) \leftarrow Q_{\pi}(s_t,a_t) + \alpha [r_t + \gamma Q_{\pi}(s_{t+1}, a_{t+1}) - Q_{\pi}(s_t, a_t)]
$$
**Sarsa算法使用该更新公式**。该公式对$s_{t+1}$状态下的动作价值函数$Q_{\pi}(s_{t+1},a_{t+1})$做有偏估计，并辅助计算更新$s_t$状态下的动作价值函数$Q_{\pi}(s_t, a_t)$。

## Q-Learning

根据蒙特卡洛方法，Q-Learning的动作价值函数$Q_{\pi}(s_t, a_t)$更新方法如下：
$$
Q_{\pi}(s_t, a_t) \leftarrow Q_{\pi}(s_t, a_t) + \alpha \cdot [r + \gamma \max_{a}Q_{\pi}(s_{t+1}, a) - Q_{\pi}(s_t, a_t)]
$$
在$s_{t}$状态时，智能体采取动作决策$a_{t}$跳转到$s_{t+1}$状态，并直接选取最大的动作价值函数$Q_{\pi}(s_{t+1}, a)$。

此外，在Q-Learning算法中，我们需要创建Q-Table记录每一个环境状态、每一个动作决策对应的动作价值函数$Q_{\pi}(s_t, a_t)$。

## 最终运行结果

最终智能体agent规划的路径图如下：

![](E:\Python_Project\SelfLearning\Deep_Reinforcement_Learning\Q-Learning-Maze\image_output\maze_image_out.png)

可以看到智能体agent走的是最短路径。

Q-Learning的训练结果如下：

![](E:\Python_Project\SelfLearning\Deep_Reinforcement_Learning\Q-Learning-Maze\image_output\training_discounted_return.jpg)

可以看到经过多轮序列训练后，累计折扣回报$G_t$迅速收敛。

除此之外，智能体agent每一个动作决策的即时奖励图如下：

![](E:\Python_Project\SelfLearning\Deep_Reinforcement_Learning\Q-Learning-Maze\image_output\inference_instant_reward.jpg)

可以看到智能体agent从起点到终点采取的动作决策数小于30，且没有触碰障碍。

## 遇到的问题与解决方法

#### 1. 任务推理

选用1000轮序列训练智能体agent后，agent类内部的Q-Table处于合理的值。进行使用Q-Table进行任务推理时，仍然需要更新Q-Table，**否则智能体agent可能会在两个状态间来回跳转**，实际遇到了这种情况。

#### 2. 推理动作决策

在Q-Learning类中，设置了两种动作决策方法：一种是名为**`training_take_action`**的方法，另一种是名为**`inference_take_action`**的方法。两种方法的区别在于：**`training_take_action`方法引入了$\epsilon-greedy$算法**，在确保动作价值函数$Q_{\pi}(s_{t}, a_{t})$最大的同时探索额外的最优解，**均衡了利用（exploitation）和探索（exploration）**；而**`inference_take_action`方法仅采用使动作价值函数$Q_{\pi}(s_{t}, a_{t})$最大的动作决策**。

经过实际验证，我认为在智能体agent执行推理任务时**必须使用`inference_take_action`方法**，这是因为**`training_take_action`方法中引入了随机抽样，可能存在误触障碍的情况**，并且也实际遇到了这种情况。

#### 3. 结果不稳定

智能体推理路线结果并不稳定，即时奖励$r_t$、$\epsilon-greedy$算法参数以及随机数种子都可较大程度影响最终结果。目前不清楚如何解决。
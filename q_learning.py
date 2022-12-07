# 该文件用于实现Q-Learning算法

import numpy as np


class QLearning(object):
    """
    使用Q-Learning方法求解迷宫问题，并构建Q-Table
    """
    def __init__(self, row, col, epsilon, alpha, gamma, action_dim):
        """
        初始化Q-Learning方法
        :param row: 单元格行数
        :param col: 单元格列数
        :param epsilon: greedy算法参数
        :param alpha: 动作价值函数Q的更新步长
        :param gamma: 折扣因子
        :param action_dim: 动作决策维度
        """
        self.Q_table = np.zeros([row * col, action_dim])                  # 构建Q-Table
        self.action_dim = action_dim                                      # 保存变量
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma

    def training_take_action(self, state):
        """
        训练过程中，使用greedy算法根据当前状态选择动作决策，平衡exploration和exploitation
        greedy算法的激进特性可帮助寻找最短路线
        :param state: 环境（迷宫）的当前状态
        :return: 智能体在当前状态下采取的动作决策
        """
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.action_dim)                   # 以一定的概率随机选择动作决策（探索exploration）
        else:
            action = np.argmax(self.Q_table[state])                       # 以一定的概率选择让动作价值函数最大的动作决策（利用exploitation）
        return action

    def inference_take_action(self, state):
        """
        推理过程中，在state状态下采取最佳动作决策，帮助智能体规划最佳路线
        智能体训练完毕后再调用此方法
        :param state: 环境的当前状态
        :return: 智能体在当前状态下的最佳动作决策
        """
        action = np.argmax(self.Q_table[state])                           # 直接选择让动作价值函数最大的动作决策
        return action

    def update(self, state, action, reward, next_state):
        """
        使用Temporal Difference方法更新动作价值函数Q以及Q-Table
        :param state: 当前环境状态
        :param action: 在当前环境状态下采取的动作决策
        :param reward: 即时奖励
        :param next_state: 环境下一个状态
        :return: 无
        """
        td_error = reward + self.gamma * np.max(self.Q_table[next_state]) - self.Q_table[state][action]
        self.Q_table[state][action] += self.alpha * td_error

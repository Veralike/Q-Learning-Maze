# 该文件用于实现智能体和环境的交互

import os
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
from image_maze import maze_image_process
from env import MazeWalking
from q_learning import QLearning

RANDOM_SEED = 0
EPSILON = 0.001
ALPHA = 0.1
GAMMA = 0.95
ACTION_DIM = 4
NUM_EPISODES = 1000                                                                     # 选择1000次序列
OUTPUT_PATH = './image_output'


def q_agent():
    """
    智能体和环境进行交互，并使用Q-Learning算法
    :return: 环境对象和智能体对象
    """
    np.random.seed(RANDOM_SEED)

    if not os.path.exists(OUTPUT_PATH):                                                 # 没有目录就创建目录
        os.makedirs(OUTPUT_PATH)

    row, col, axis_height, axis_width, \
        start_x, start_y, end_x, end_y, barrier \
        = maze_image_process()                                                          # 获取迷宫信息

    env = MazeWalking(row, col, axis_height, axis_width, start_x, start_y,
                      end_x, end_y, barrier)                                            # 初始化环境
    agent = QLearning(row, col, EPSILON, ALPHA, GAMMA, ACTION_DIM)                      # 初始化智能体

    return_list = []                                                                    # 保存每一轮的累计折扣回报

    for i in range(10):                                                                 # 循环10次，每次执行100次序列
        with tqdm(total=int(NUM_EPISODES/10), desc='Iteration %d' % (i+1)) as pbar:     # 使用with语句构建10个进度条
            for i_episode in range(int(NUM_EPISODES/10)):                               # 使用多个序列训练智能体
                episode_return = 0                                                      # 初始化这一轮的累计折扣回报
                state = env.reset()                                                     # 初始化环境状态
                done = False                                                            # 结束标志位置False，并完整执行序列
                while not done:
                    action = agent.training_take_action(state)                          # 获取当前环境状态下的动作决策
                    next_state, reward, done = env.step(action)                         # 当前环境状态采取动作决策并跳转状态
                    agent.update(state, action, reward, next_state)                     # 更新Q-Table，即动作价值函数
                    state = next_state                                                  # 跳转至下一个状态
                    episode_return += GAMMA * reward                                    # 更新累计折扣回报
                return_list.append(episode_return)                                      # 一轮序列执行完毕后，保存这轮序列的累计折扣回报

                if (i_episode + 1) % 10 == 0:                                           # 每10轮就打印一次信息
                    pbar.set_postfix({
                        'episodes':
                            '%d' % (NUM_EPISODES / 10 * i + i_episode + 1),
                        'discounted return':
                            '%.3f' % np.mean(return_list[-10:])
                    })
                pbar.update(1)

    episode_list = list(range(len(return_list)))                                        # 从头构建一组列表
    plt.figure(figsize=(10, 4), dpi=200)                                                # 最后绘制并保存训练过程的累计折扣回报变化图
    plt.plot(episode_list, return_list, label='Q-Learning Training')
    plt.grid(visible=True, axis='both', linestyle='-.', alpha=0.7)
    plt.title('Q-Learning Training on Maze')
    plt.xlabel('Episodes')
    plt.ylabel('Discounted Return')
    plt.legend()
    plt.savefig(OUTPUT_PATH + '/training_discounted_return.jpg')
    plt.show()

    return env, agent                                                                   # 最后返回环境和智能体对象

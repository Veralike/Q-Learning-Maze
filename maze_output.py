# 该文件用于输出智能体探索的最佳路径图片

import os
import cv2
from matplotlib import pyplot as plt

IMAGE_PATH = './image_input/maze_input.png'
OUTPUT_PATH = './image_output'


def maze_image_output(env, agent):
    """
    使用智能体推理迷宫路径
    :param env: 环境对象
    :param agent: 智能体对象
    :return: 无
    """
    if not os.path.exists(OUTPUT_PATH):                                      # 没有目录就创建目录
        os.makedirs(OUTPUT_PATH)

    maze_image = cv2.imread(IMAGE_PATH)

    reward_list = []                                                         # 创建列表保存整轮序列的所有即时奖励
    q_env, q_agent = env, agent
    state = q_env.reset()
    while True:
        action = q_agent.inference_take_action(state)                        # 获取当前状态下的动作决策
        next_state, reward, done = q_env.step(action)
        q_agent.update(state, action, reward, next_state)                    # 要写这步，如果不更新Q-Table可能会出问题
        state = next_state
        reward_list.append(reward)                                           # 保存即时奖励
        if done:
            break

        for i in range(q_env.current_x*q_env.axis_height, (q_env.current_x+1)*q_env.axis_height, 1):
            for j in range(q_env.current_y*q_env.axis_width, (q_env.current_y+1)*q_env.axis_width, 1):
                maze_image[i][j] = [255, 255, 255]                           # 将单元格变为白色，显示智能体的路径

    episode_list = list(range(len(reward_list)))
    plt.figure(figsize=(10, 4), dpi=200)                                     # 最后绘制并保存推理过程的即时奖励变化图
    plt.plot(episode_list, reward_list, label='Q-Learning Inference')
    plt.grid(visible=True, axis='both', linestyle='-.', alpha=0.7)
    plt.title('Q-Learning Inference on Maze')
    plt.xlabel('Episodes')
    plt.ylabel('Instant Reward')
    plt.xticks(range(0, 30, 1))
    plt.legend()
    plt.savefig(OUTPUT_PATH + '/inference_instant_reward.jpg')
    plt.show()

    cv2.imwrite(OUTPUT_PATH + '/maze_image_out.png', maze_image)             # 最后保存智能体的路径现实图片
    print('\n已生成迷宫路径规划图片。\n')

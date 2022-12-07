# 该文件用于构建迷宫环境


class MazeWalking(object):
    """
    使用OpenCV的返回值构建迷宫环境
    OpenCV的返回值将作为迷宫的先验知识
    """
    def __init__(self, row, col, axis_height, axis_width, start_x, start_y, end_x, end_y, barrier):
        """
        初始化迷宫环境环境
        :param row: 行数
        :param col: 列数
        :param axis_height: 单元格高度
        :param axis_width: 单元格宽度
        :param start_x: 迷宫起点横坐标
        :param start_y: 迷宫起点纵坐标
        :param end_x: 迷宫终点横坐标
        :param end_y: 迷宫终点纵坐标
        :param barrier: 障碍坐标
        """
        self.row, self.col = row, col                                                    # 记录迷宫信息
        self.axis_height, self.axis_width = axis_height, axis_width
        self.start_x, self.start_y = start_x, start_y
        self.end_x, self.end_y = end_x, end_y

        self.current_x, self.current_y = self.start_x, self.start_y                      # 记录当前智能体坐标

        self.barrier = barrier                                                           # 记录障碍坐标

    def step(self, action):
        """
        根据智能体采取的动作返回下一时刻的状态、即时奖励
        定义四种动作：[-1, 0]上，[1, 0]下，[0, -1]左，[0, 1]右
        :param action: 智能体采取的动作决策
        :return: 下一时刻状态、即时奖励以及整个序列是否执行完毕
        """
        change = [[-1, 0], [1, 0], [0, -1], [0, 1]]
        self.current_x = min(self.row - 1, max(0, self.current_x + change[action][0]))   # min的作用是为了不让智能体超出下（右）边界
        self.current_y = min(self.col - 1, max(0, self.current_y + change[action][1]))   # max的作用是为了不让智能体超出上（左）边界
        next_state = self.current_x * self.col + self.current_y                          # 状态的值为当前单元格的顺序
        reward = -1                                                                      # 即时奖励设置为-1
        done = False
        for [i, j] in self.barrier:                                                      # 判断是否遇到障碍，否则即时奖励设置为-100
            if self.current_x == i and self.current_y == j:
                reward = -100                                                            # 即时奖励设置为-100
        if self.current_x == self.end_x and self.current_y == self.end_y:                # 到达终点，标志位置为True
            done = True
        return next_state, reward, done

    def reset(self):
        """
        智能体复位到初始位置
        :return: 初始状态
        """
        self.current_x, self.current_y = self.start_x, self.start_y                      # 恢复到初始状态
        return self.current_x * self.col + self.current_y                                # 返回Q-Table的索引

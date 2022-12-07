# 这是一个示例 Python 脚本。

# 按 Shift+F10 执行或将其替换为您的代码。
# 按 双击 Shift 在所有地方搜索类、文件、工具窗口、操作和设置。

from agent import q_agent
from maze_output import maze_image_output


# 按间距中的绿色按钮以运行脚本。
if __name__ == '__main__':
    """
    主函数部分，获取环境和智能体对象，执行训练和推理任务
    """
    qlearning_env, qlearning_agent = q_agent()
    maze_image_output(qlearning_env, qlearning_agent)

# 访问 https://www.jetbrains.com/help/pycharm/ 获取 PyCharm 帮助

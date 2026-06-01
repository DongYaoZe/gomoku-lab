import os
import copy
from pathlib import Path

try:
    from alphazero_net import PolicyValueNet
    from alphazero_mcts import MCTSPlayer
    HAS_PYTORCH = True
except ImportError:
    HAS_PYTORCH = False

_DEFAULT_MODEL = str(Path(__file__).with_name("current_policy_15x15.model"))


class AlphaZeroAI:
    """
    原生实现的 AlphaZero AI
    使用重写的 alphazero_net.py (PyTorch 模型) 和 alphazero_mcts.py (搜索器)
    无需依赖任何第三方外部框架，直接对接 core.GomokuGame。
    """
    def __init__(self, player, model_file=None, n_playout=400):
        if model_file is None:
            model_file = _DEFAULT_MODEL
        self.player = player
        self.latest_win_rate = 50.0
        
        if not HAS_PYTORCH:
            print("警告: 缺少 PyTorch 依赖！AlphaZero 将无法正常启动。")
            return
            
        # 1. 实例化策略价值网络
        # 注意: 即使找不到模型也会初始化一个随机的“婴儿网络”，保证游戏不崩溃
        self.policy_value_net = PolicyValueNet(
            board_width=15, 
            board_height=15, 
            model_file=model_file, 
            use_gpu=True # 如果有 CUDA，底层会自动启用
        )
            
        # 2. 实例化配合神经网络评估的蒙特卡洛树搜索器
        self.mcts_player = MCTSPlayer(
            self.policy_value_net.policy_value_fn,
            c_puct=5, 
            n_playout=n_playout,
            is_selfplay=0 # 评估/实战模式
        )

    def get_best_move(self, game):
        """
        利用当前 game 生成一次落子
        """
        if not HAS_PYTORCH:
            import random
            print("AlphaZero 降级为随机落子...")
            return (random.randint(0, 14), random.randint(0, 14))

        # 传入的 game 可能是界面的直接引用，保险起见进行深拷贝
        game_copy = copy.deepcopy(game)
        
        # 检查是否还有空位
        has_empty = False
        for r in range(game.board_size):
            for c in range(game.board_size):
                if game.board[r][c] == 0:
                    has_empty = True
                    break
            if has_empty:
                break
                
        if not has_empty:
            print("AlphaZero: Board is full, no moves available.")
            return None

        # 1. 调用 MCTS 获取最佳动作 (内部包含了 n_playout 次的树推演)
        action = self.mcts_player.get_action(game_copy, temp=1e-3, return_prob=False)
        
        # 2. 释放树结构状态，以保证每次对局重新同步（防止因玩家落子超出了当前树分支而崩溃）
        self.mcts_player.reset_player()
        
        r = action // game.board_size
        c = action % game.board_size
        
        # 3. 网络评估当前盘面，转换为黑方胜率显示
        # policy_value_fn 返回当前玩家视角的价值 [-1, 1]
        _, value = self.policy_value_net.policy_value_fn(game)
        self_win_rate = (value + 1.0) / 2.0 * 100.0
        if self.player == 1:
            self.latest_win_rate = self_win_rate
        else:
            self.latest_win_rate = 100.0 - self_win_rate
        
        return r, c

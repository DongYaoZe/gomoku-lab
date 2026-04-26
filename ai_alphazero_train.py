# -*- coding: utf-8 -*-
"""
快速训练原生 AlphaZero 模型 (15x15 试水版)
完全原生实现的自我对弈训练框架，直接基于 core.GomokuGame 采集数据，脱离任何第三方依赖。
"""

import os
import random
import numpy as np
from collections import deque
import copy

try:
    from alphazero_net import PolicyValueNet
    from alphazero_mcts import MCTSPlayer
    from core import GomokuGame
except ImportError as e:
    print(f"导入失败: {e}。请确保依赖都在同一目录下且安装了 PyTorch。")
    import sys
    sys.exit(1)

class FastTrainPipeline():
    def __init__(self, init_model='./current_policy_15x15.model'):
        self.board_width = 15
        self.board_height = 15
        
        # =======================================================
        # 🚨 [关键可调参数区: 用户试水专用] 🚨
        # 根据算力和时间，您可以随时修改以下参数来加速训练或提高质量！
        # =======================================================
        
        # MCTS 模拟次数 (核心耗时点)：官方为 400，单机推荐 50 以追求快速验证。
        self.n_playout = 10      
        
        # 每次策略更新前的独立棋局数量 (收集数据的速度)：
        self.play_batch_size = 1  
        
        # 训练过程最大的对局批次，供预览：
        self.game_batch_num = 20  
        
        # =======================================================

        # 训练引擎参数
        self.learn_rate = 2e-3
        self.lr_multiplier = 1.0  
        self.temp = 1.0  
        self.c_puct = 5
        self.buffer_size = 10000
        self.batch_size = 128  
        self.data_buffer = deque(maxlen=self.buffer_size)
        self.epochs = 5  
        self.kl_targ = 0.02
        self.check_freq = 1  
        
        print("[*] 初始化 15x15 残差神经网络...")
        self.policy_value_net = PolicyValueNet(
            self.board_width, 
            self.board_height, 
            model_file=init_model if os.path.exists(init_model) else None,
            use_gpu=True
        )
            
        self.mcts_player = MCTSPlayer(
            self.policy_value_net.policy_value_fn,
            c_puct=self.c_puct,
            n_playout=self.n_playout,
            is_selfplay=1
        )

    def get_equi_data(self, play_data):
        """
        利用棋盘的对称性扩充训练数据
        play_data: [(state, mcts_prob, winner_z), ...]
        """
        extend_data = []
        for state, mcts_prob, winner in play_data:
            # 这里的 state 是 4x15x15 的矩阵
            for i in [1, 2, 3, 4]:
                # 旋转
                equi_state = np.array([np.rot90(s, i) for s in state])
                equi_mcts_prob = np.rot90(np.flipud(mcts_prob.reshape(self.board_height, self.board_width)), i)
                extend_data.append((equi_state, np.flipud(equi_mcts_prob).flatten(), winner))
                
                # 翻转
                equi_state = np.array([np.fliplr(s) for s in equi_state])
                equi_mcts_prob = np.fliplr(equi_mcts_prob)
                extend_data.append((equi_state, np.flipud(equi_mcts_prob).flatten(), winner))
        return extend_data

    def start_self_play(self, temp=1.0):
        """
        运行一局自我对弈，记录并返回这局产生的数据
        """
        game = GomokuGame(board_size=self.board_width)
        self.mcts_player.reset_player()
        
        states, mcts_probs, current_players = [], [], []
        
        while True:
            # 获取当前盘面的神经网络特征输入
            state_input = self.policy_value_net.current_state(game)
            
            # 使用 MCTS 决定下哪一步
            action, action_probs = self.mcts_player.get_action(game, temp=temp, return_prob=True)
            
            # 保存数据 (state_input, \pi, current_player)
            states.append(state_input)
            mcts_probs.append(action_probs)
            current_players.append(game.current_player)
            
            # 执行落子
            r, c = action // self.board_width, action % self.board_width
            game.make_move(r, c)
            
            # 判断游戏是否结束
            if game.winner != 0 or game.check_draw():
                winners_z = np.zeros(len(current_players))
                if game.winner != 0:
                    # 对于记录里的每一步，如果当前走棋的玩家正是赢家，奖励1；否则惩罚-1
                    winners_z[np.array(current_players) == game.winner] = 1.0
                    winners_z[np.array(current_players) != game.winner] = -1.0
                # 重置 MCTS 树结构
                self.mcts_player.reset_player()
                return game.winner, zip(states, mcts_probs, winners_z)

    def collect_selfplay_data(self, n_games=1):
        for i in range(n_games):
            print(f" >> 正在进行自我博弈收集数据 (MCTS 思考次数:{self.n_playout}) ...")
            winner, play_data = self.start_self_play(temp=self.temp)
            play_data = list(play_data)[:]
            play_data = self.get_equi_data(play_data)
            self.data_buffer.extend(play_data)

    def policy_update(self):
        """更新策略价值网络"""
        mini_batch = random.sample(self.data_buffer, self.batch_size)
        state_batch = [data[0] for data in mini_batch]
        mcts_probs_batch = [data[1] for data in mini_batch]
        winner_batch = [data[2] for data in mini_batch]
        
        old_probs, old_v = self.policy_value_net.policy_value(state_batch)
        
        for i in range(self.epochs):
            loss, entropy = self.policy_value_net.train_step(
                state_batch, 
                mcts_probs_batch, 
                winner_batch, 
                self.learn_rate * self.lr_multiplier
            )
            new_probs, new_v = self.policy_value_net.policy_value(state_batch)
            kl = np.mean(np.sum(old_probs * (np.log(old_probs + 1e-10) - np.log(new_probs + 1e-10)), axis=1))
            
            if kl > self.kl_targ * 4:  
                break
                
        if kl > self.kl_targ * 2 and self.lr_multiplier > 0.1:
            self.lr_multiplier /= 1.5
        elif kl < self.kl_targ / 2 and self.lr_multiplier < 10:
            self.lr_multiplier *= 1.5

        print(f" => 训练更新 | Loss: {loss:.4f} | Entropy: {entropy:.4f} | KL: {kl:.5f}")
        return loss, entropy

    def run(self):
        print("======== 原生 AlphaZero 强化学习微小训练计划启动 ========")
        try:
            for i in range(self.game_batch_num):
                print(f"\n[Batch {i+1} / {self.game_batch_num}]")
                self.collect_selfplay_data(self.play_batch_size)
                print(f" -> 已记录棋步数据. 当前缓存区大小: {len(self.data_buffer)}")
                
                if len(self.data_buffer) > self.batch_size:
                    loss, entropy = self.policy_update()
                    
                if (i+1) % self.check_freq == 0:
                    model_path = './current_policy_15x15.model'
                    self.policy_value_net.save_model(model_path)
                    print(f" -> 模型骨架权重已保存至: {model_path}")

            print("\n 所有演示训练规划执行完毕。")
        except KeyboardInterrupt:
            print('\n\r[!] 用户手动退出')

if __name__ == '__main__':
    pipeline = FastTrainPipeline()
    pipeline.run()

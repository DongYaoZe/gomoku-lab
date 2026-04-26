import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os

class GomokuNet(nn.Module):
    """
    五子棋的残差卷积神经网络模型
    包含公共卷积层，以及分离的动作策略头(Policy Head)和价值头(Value Head)
    """
    def __init__(self, board_width=15, board_height=15):
        super(GomokuNet, self).__init__()
        self.board_width = board_width
        self.board_height = board_height
        
        # 公共特征提取层
        self.conv1 = nn.Conv2d(4, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        # Policy 头：输出落子概率
        self.act_conv1 = nn.Conv2d(128, 4, kernel_size=1)
        self.act_fc1 = nn.Linear(4 * board_width * board_height, board_width * board_height)
        
        # Value 头：输出局面胜率估值 [-1, 1]
        self.val_conv1 = nn.Conv2d(128, 2, kernel_size=1)
        self.val_fc1 = nn.Linear(2 * board_width * board_height, 64)
        self.val_fc2 = nn.Linear(64, 1)

    def forward(self, state_input):
        # 提取特征
        x = F.relu(self.conv1(state_input))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        # 计算策略 (Policy)
        x_act = F.relu(self.act_conv1(x))
        x_act = x_act.view(-1, 4 * self.board_width * self.board_height)
        x_act = F.log_softmax(self.act_fc1(x_act), dim=1)
        
        # 计算价值 (Value)
        x_val = F.relu(self.val_conv1(x))
        x_val = x_val.view(-1, 2 * self.board_width * self.board_height)
        x_val = F.relu(self.val_fc1(x_val))
        x_val = torch.tanh(self.val_fc2(x_val))
        
        return x_act, x_val

class PolicyValueNet:
    """
    神经网络的包装类，处理 numpy <-> tensor 的转换，以及训练相关步骤
    """
    def __init__(self, board_width=15, board_height=15, model_file=None, use_gpu=False):
        self.board_width = board_width
        self.board_height = board_height
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_gpu else "cpu")
        self.l2_const = 1e-4
        
        self.net = GomokuNet(board_width, board_height).to(self.device)
        self.optimizer = optim.Adam(self.net.parameters(), weight_decay=self.l2_const)

        if model_file and os.path.exists(model_file):
            try:
                self.net.load_state_dict(torch.load(model_file, map_location=self.device))
                print(f"[PolicyValueNet] 成功加载模型权重: {model_file}")
            except Exception as e:
                print(f"[PolicyValueNet] 模型加载失败: {e}")

    def current_state(self, game):
        """
        从 core.GomokuGame 提取 4 通道的神经网络输入状态:
        通道 0: 当前玩家的棋子位置
        通道 1: 对手的棋子位置
        通道 2: 上一步落子位置 (全 0 或只有一个 1)
        通道 3: 当前玩家颜色 (黑棋全 1，白棋全 0)
        """
        square_state = np.zeros((4, self.board_height, self.board_width), dtype=np.float32)
        
        current_player = game.current_player
        opponent = 3 - current_player
        
        for r in range(self.board_height):
            for c in range(self.board_width):
                if game.board[r][c] == current_player:
                    square_state[0][r][c] = 1.0
                elif game.board[r][c] == opponent:
                    square_state[1][r][c] = 1.0
                    
        if game.history:
            last_r, last_c, _ = game.history[-1]
            square_state[2][last_r][last_c] = 1.0
            
        if current_player == 1:
            square_state[3][:, :] = 1.0
            
        return square_state

    def policy_value(self, state_batch):
        """输入批量状态，返回批量动作概率和局面价值"""
        self.net.eval()
        with torch.no_grad():
            state_tensor = torch.FloatTensor(np.array(state_batch)).to(self.device)
            log_act_probs, value = self.net(state_tensor)
            act_probs = torch.exp(log_act_probs).cpu().numpy()
            return act_probs, value.cpu().numpy()

    def policy_value_fn(self, game):
        """
        输入一个 game 实例，返回所有合法落子的 (动作, 概率) 列表，以及局面评分
        动作被编码为一个标量: r * width + c
        """
        legal_positions = []
        for r in range(self.board_height):
            for c in range(self.board_width):
                if game.board[r][c] == 0:
                    legal_positions.append(r * self.board_width + c)
                    
        if not legal_positions:
            return [], 0.0
            
        state = self.current_state(game)
        # 添加 batch 维度
        state_batch = np.expand_dims(state, axis=0)
        
        act_probs, value = self.policy_value(state_batch)
        act_probs = act_probs[0] # 取出 batch 中唯一的元素
        value = value[0][0]
        
        # 仅打包合法动作
        legal_probs = zip(legal_positions, act_probs[legal_positions])
        return list(legal_probs), value

    def train_step(self, state_batch, mcts_probs, winner_batch, lr):
        """执行单步训练"""
        self.net.train()
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
            
        state_tensor = torch.FloatTensor(np.array(state_batch)).to(self.device)
        mcts_probs_tensor = torch.FloatTensor(np.array(mcts_probs)).to(self.device)
        winner_tensor = torch.FloatTensor(np.array(winner_batch)).to(self.device)
        
        self.optimizer.zero_grad()
        log_act_probs, value = self.net(state_tensor)
        
        # 价值损失: 预测值和真实结果的均方误差
        value_loss = F.mse_loss(value.view(-1), winner_tensor)
        # 策略损失: MCTS得出的目标概率 与 神经网络输出的对数概率 的交叉熵
        policy_loss = -torch.mean(torch.sum(mcts_probs_tensor * log_act_probs, 1))
        
        loss = value_loss + policy_loss
        loss.backward()
        self.optimizer.step()
        
        # 顺便算一下交叉熵用于监测
        entropy = -torch.mean(torch.sum(torch.exp(log_act_probs) * log_act_probs, 1))
        return loss.item(), entropy.item()

    def save_model(self, model_file):
        """保存模型"""
        torch.save(self.net.state_dict(), model_file)

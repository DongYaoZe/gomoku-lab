import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os


class ResidualBlock(nn.Module):
    def __init__(self, n_filters):
        super().__init__()
        self.conv1 = nn.Conv2d(n_filters, n_filters, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(n_filters)
        self.conv2 = nn.Conv2d(n_filters, n_filters, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(n_filters)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return F.relu(out)


class GomokuNet(nn.Module):
    """
    残差卷积神经网络：初始卷积 + N 个残差块 + 策略头/价值头
    """
    def __init__(self, board_width=15, board_height=15, n_filters=128, n_res_blocks=4):
        super().__init__()
        self.board_width = board_width
        self.board_height = board_height

        # 初始特征提取
        self.init_conv = nn.Conv2d(4, n_filters, kernel_size=3, padding=1)
        self.init_bn = nn.BatchNorm2d(n_filters)

        # 残差块堆叠
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(n_filters) for _ in range(n_res_blocks)]
        )

        # Policy 头
        self.policy_conv = nn.Conv2d(n_filters, 4, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(4)
        self.policy_fc = nn.Linear(4 * board_width * board_height, board_width * board_height)

        # Value 头
        self.value_conv = nn.Conv2d(n_filters, 2, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(2)
        self.value_fc1 = nn.Linear(2 * board_width * board_height, 64)
        self.value_fc2 = nn.Linear(64, 1)

    def forward(self, x):
        # 公共特征
        x = F.relu(self.init_bn(self.init_conv(x)))
        x = self.res_blocks(x)

        # 策略输出
        p = F.relu(self.policy_bn(self.policy_conv(x)))
        p = p.view(-1, 4 * self.board_width * self.board_height)
        p = F.log_softmax(self.policy_fc(p), dim=1)

        # 价值输出
        v = F.relu(self.value_bn(self.value_conv(x)))
        v = v.view(-1, 2 * self.board_width * self.board_height)
        v = F.relu(self.value_fc1(v))
        v = torch.tanh(self.value_fc2(v))

        return p, v


class PolicyValueNet:
    """神经网络包装类：处理 numpy/tensor 转换、推理与训练"""

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
                print(f"[PolicyValueNet] 模型加载成功: {model_file}")
            except Exception as e:
                print(f"[PolicyValueNet] 模型结构不兼容，使用随机初始化: {e}")

    def current_state(self, game):
        """
        提取 4 通道神经网络输入:
        0: 当前玩家棋子  1: 对手棋子  2: 上一步位置  3: 当前玩家颜色
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
        self.net.eval()
        with torch.no_grad():
            state_tensor = torch.FloatTensor(np.array(state_batch)).to(self.device)
            log_act_probs, value = self.net(state_tensor)
            act_probs = torch.exp(log_act_probs).cpu().numpy()
            return act_probs, value.cpu().numpy()

    def policy_value_fn(self, game):
        """返回所有合法落子的 (动作, 概率) 列表和局面评分"""
        legal_positions = []
        for r in range(self.board_height):
            for c in range(self.board_width):
                if game.board[r][c] == 0:
                    legal_positions.append(r * self.board_width + c)

        if not legal_positions:
            return [], 0.0

        state = self.current_state(game)
        state_batch = np.expand_dims(state, axis=0)
        act_probs, value = self.policy_value(state_batch)
        act_probs = act_probs[0]
        value = value[0][0]

        legal_probs = zip(legal_positions, act_probs[legal_positions])
        return list(legal_probs), value

    def train_step(self, state_batch, mcts_probs, winner_batch, lr):
        self.net.train()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        state_tensor = torch.FloatTensor(np.array(state_batch)).to(self.device)
        mcts_probs_tensor = torch.FloatTensor(np.array(mcts_probs)).to(self.device)
        winner_tensor = torch.FloatTensor(np.array(winner_batch)).to(self.device)

        self.optimizer.zero_grad()
        log_act_probs, value = self.net(state_tensor)

        value_loss = F.mse_loss(value.view(-1), winner_tensor)
        policy_loss = -torch.mean(torch.sum(mcts_probs_tensor * log_act_probs, 1))

        loss = value_loss + policy_loss
        loss.backward()
        self.optimizer.step()

        entropy = -torch.mean(torch.sum(torch.exp(log_act_probs) * log_act_probs, 1))
        return loss.item(), entropy.item()

    def save_model(self, model_file):
        torch.save(self.net.state_dict(), model_file)

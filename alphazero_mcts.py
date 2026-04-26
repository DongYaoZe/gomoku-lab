import math
import copy
import numpy as np

def softmax(x):
    probs = np.exp(x - np.max(x))
    probs /= np.sum(probs)
    return probs

class TreeNode:
    """MCTS 树的一个节点。包含指向父节点和子节点的指针，以及相关的搜索状态值。"""
    def __init__(self, parent, prior_p):
        self.parent = parent
        self.children = {}  # action -> TreeNode
        self.n_visits = 0
        self.Q = 0.0
        self.u = 0.0
        self.P = prior_p

    def expand(self, action_priors):
        """展开子节点
        action_priors: 一个包含 (action, prior_prob) 的列表
        """
        for action, prob in action_priors:
            if action not in self.children:
                self.children[action] = TreeNode(self, prob)

    def select(self, c_puct):
        """选择最大化 Q + U 的子节点"""
        return max(self.children.items(), key=lambda act_node: act_node[1].get_value(c_puct))

    def get_value(self, c_puct):
        """计算并返回 Q + U"""
        self.u = c_puct * self.P * math.sqrt(self.parent.n_visits) / (1 + self.n_visits)
        return self.Q + self.u

    def update(self, leaf_value):
        """从叶子节点的评估值反向更新自身的 Q 和 N"""
        self.n_visits += 1
        # 增量更新 Q 值
        self.Q += 1.0 * (leaf_value - self.Q) / self.n_visits

    def update_recursive(self, leaf_value):
        """递归向上更新整条路径，直到根节点"""
        if self.parent:
            self.parent.update_recursive(-leaf_value) # 对方的得分是己方的负分
        self.update(leaf_value)

    def is_leaf(self):
        """判断是否为叶子节点（没有展开过子节点）"""
        return self.children == {}

    def is_root(self):
        return self.parent is None


class MCTSPlayer:
    """基于神经网络先验概率的 MCTS 搜索器"""
    def __init__(self, policy_value_fn, c_puct=5, n_playout=400, is_selfplay=0):
        self.root = TreeNode(None, 1.0)
        self.policy_value_fn = policy_value_fn
        self.c_puct = c_puct
        self.n_playout = n_playout
        self.is_selfplay = is_selfplay

    def _playout(self, state):
        """执行一次从根到叶的 MCTS 模拟，并使用神经网络评估叶子节点"""
        node = self.root
        
        # 1. 选择 (Selection)
        while not node.is_leaf():
            action, node = node.select(self.c_puct)
            # 根据 action 提取 r, c 落子
            r, c = action // state.board_size, action % state.board_size
            state.make_move(r, c)
            
        # 判断游戏是否结束
        is_end = state.winner != 0 or state.check_draw()
        
        # 2. 评估与展开 (Evaluation & Expansion)
        if not is_end:
            # 使用神经网络预测候选动作的概率分布和当前局面的价值
            action_priors, leaf_value = self.policy_value_fn(state)
            node.expand(action_priors)
        else:
            # 如果游戏结束，计算真实价值
            if state.winner == 0:  # 平局
                leaf_value = 0.0
            else:
                # 若刚好分出胜负，意味着上一步走棋的玩家赢了，当前 node 是待落子状态，
                # 对于当前正在思考的玩家来说，这就是必败局面 (-1)
                leaf_value = -1.0
                
        # 3. 回溯更新 (Backpropagation)
        # 注意: leaf_value 是对当前玩家的价值，我们要回溯更新
        node.update_recursive(-leaf_value)

    def get_action(self, state, temp=1e-3, return_prob=False):
        """
        在给定的游戏状态下，经过多次 playout 得到最终的动作
        temp: 温度参数，控制探索程度。自对弈前期可以用 1.0，后期和评估时趋近于 0
        return_prob: 是否返回计算出来的 $\\pi$ 概率分布（训练时需要）
        """
        for n in range(self.n_playout):
            state_copy = copy.deepcopy(state)
            self._playout(state_copy)

        # 收集根节点的所有可行走子和访问次数
        act_visits = [(act, node.n_visits) for act, node in self.root.children.items()]
        acts, visits = zip(*act_visits)
        
        act_probs = softmax(1.0 / temp * np.log(np.array(visits) + 1e-10))

        if self.is_selfplay:
            # 引入狄利克雷噪声，增加自对弈在根节点的探索
            action = np.random.choice(
                acts,
                p=0.75 * act_probs + 0.25 * np.random.dirichlet(0.3 * np.ones(len(act_probs)))
            )
            # 在自对弈中，我们会沿着选定的节点前进一步，保留部分树结构
            self.root = self.root.children[action]
            self.root.parent = None
        else:
            # 竞技或评估阶段，直接取访问次数最多的点
            action = acts[np.argmax(act_probs)]
            self.reset_player()
            
        if return_prob:
            # 返回整个棋盘大小的概率分布向量，未访问的动作概率为 0
            pi = np.zeros(state.board_size * state.board_size)
            for act, prob in zip(acts, act_probs):
                pi[act] = prob
            return action, pi
        else:
            return action

    def reset_player(self):
        """重置整棵树（每次人类对弈回合重新开始构建）"""
        self.root = TreeNode(None, 1.0)

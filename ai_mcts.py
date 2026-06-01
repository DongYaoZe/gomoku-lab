import time
import math
import random
from ai_advanced import AdvancedAI


class MCTSNode:
    __slots__ = ('parent', 'move', 'player_just_moved', 'children',
                 'wins', 'visits', 'rave_wins', 'rave_visits', 'untried_moves')

    def __init__(self, parent, move, player_just_moved):
        self.parent = parent
        self.move = move
        self.player_just_moved = player_just_moved
        self.children = []
        self.wins = 0.0
        self.visits = 0
        self.rave_wins = 0.0
        self.rave_visits = 0
        self.untried_moves = None


class MCTSAI:
    RAVE_K = 300
    C_EXPLORE = 1.414
    PROG_WIDEN_C = 2.0
    PROG_WIDEN_ALPHA = 0.5

    def __init__(self, player, time_limit=2.5):
        self.player = player
        self.time_limit = time_limit
        self.evaluator = AdvancedAI(player=player, depth=1)
        self.latest_win_rate = 50.0

    def get_best_move(self, game):
        start_time = time.time()

        root_candidates = self._get_prioritized_candidates(game)
        if not root_candidates:
            return (game.board_size // 2, game.board_size // 2)

        root = MCTSNode(parent=None, move=None, player_just_moved=3 - self.player)
        root.untried_moves = list(root_candidates)

        iterations = 0

        while time.time() - start_time < self.time_limit:
            node = root
            moves_in_path = []

            # 1. SELECTION
            while node.untried_moves is not None and len(node.untried_moves) == 0 and node.children:
                node = self._select_child(node)
                game.board[node.move[0]][node.move[1]] = node.player_just_moved
                moves_in_path.append(node.move)

            # 2. EXPANSION with progressive widening
            if node.untried_moves is None:
                node.untried_moves = self._get_prioritized_candidates(game)

            max_children = int(self.PROG_WIDEN_C * ((node.visits + 1) ** self.PROG_WIDEN_ALPHA))
            if node.untried_moves and len(node.children) < max_children:
                move = node.untried_moves.pop(0)
                next_player = 3 - node.player_just_moved
                child = MCTSNode(parent=node, move=move, player_just_moved=next_player)
                node.children.append(child)
                node = child
                game.board[move[0]][move[1]] = next_player
                moves_in_path.append(move)

            # 3. EVALUATION (heuristic value substitute)
            score = self.evaluator._evaluate_board(game)

            if node.move:
                r, c = node.move
                if game.check_winner(r, c):
                    score = 100000 if node.player_just_moved == self.player else -100000
                elif game.check_draw():
                    score = 0

            exponent = max(min(-score / 2000.0, 50), -50)
            win_prob = 1.0 / (1.0 + math.exp(exponent))

            # 4. BACKPROPAGATION with RAVE
            moves_set = set(moves_in_path)
            while node is not None:
                node.visits += 1
                if node.player_just_moved == self.player:
                    node.wins += win_prob
                else:
                    node.wins += (1.0 - win_prob)

                # Update RAVE for siblings
                if node.parent:
                    for sibling in node.parent.children:
                        if sibling.move in moves_set and sibling is not node:
                            sibling.rave_visits += 1
                            if sibling.player_just_moved == self.player:
                                sibling.rave_wins += win_prob
                            else:
                                sibling.rave_wins += (1.0 - win_prob)

                if node.move is not None:
                    game.board[node.move[0]][node.move[1]] = 0

                node = node.parent

            iterations += 1

        print(f"[MCTSAI] {iterations} iterations in {self.time_limit}s")

        if not root.children:
            return random.choice(root_candidates)

        best_child = max(root.children, key=lambda c: c.visits)

        if best_child.visits > 0:
            if best_child.player_just_moved == self.player:
                ratio = best_child.wins / best_child.visits
            else:
                ratio = 1.0 - (best_child.wins / best_child.visits)
            self.latest_win_rate = ratio * 100.0
        else:
            self.latest_win_rate = 50.0

        return best_child.move

    def _select_child(self, node):
        best_score = -float('inf')
        best_child = None
        log_parent = math.log(node.visits)

        for child in node.children:
            if child.visits == 0:
                ucb = float('inf')
            else:
                exploit = child.wins / child.visits
                explore = self.C_EXPLORE * math.sqrt(log_parent / child.visits)
                ucb = exploit + explore

                # Blend with RAVE
                if child.rave_visits > 0:
                    beta_sq = self.RAVE_K / (3 * child.visits + self.RAVE_K)
                    beta = math.sqrt(beta_sq)
                    rave_val = child.rave_wins / child.rave_visits
                    ucb = (1 - beta) * (exploit + explore) + beta * rave_val

            if ucb > best_score:
                best_score = ucb
                best_child = child
        return best_child

    def _get_prioritized_candidates(self, game):
        """收集邻近空位并按启发评分排序，返回 Top-K"""
        radius = 2
        candidates = set()
        has_stone = False
        board_size = game.board_size
        board = game.board

        for r in range(board_size):
            for c in range(board_size):
                if board[r][c] != 0:
                    has_stone = True
                    for dr in range(-radius, radius + 1):
                        for dc in range(-radius, radius + 1):
                            if dr == 0 and dc == 0:
                                continue
                            nr, nc = r + dr, c + dc
                            if 0 <= nr < board_size and 0 <= nc < board_size and board[nr][nc] == 0:
                                candidates.add((nr, nc))

        if not has_stone:
            return []

        current_player = game.current_player
        scored = []
        for r, c in candidates:
            s = self.evaluator._evaluate_point(board, r, c, current_player) + \
                self.evaluator._evaluate_point(board, r, c, 3 - current_player) * 0.8
            scored.append((s, r, c))

        scored.sort(reverse=True, key=lambda x: x[0])
        return [(r, c) for _, r, c in scored[:20]]

import time
import math
import random
from ai_advanced import AdvancedAI

class MCTSNode:
    def __init__(self, parent, move, player_just_moved):
        self.parent = parent
        self.move = move
        self.player_just_moved = player_just_moved
        self.children = []
        self.wins = 0.0  # Fraction of wins, relative to player_just_moved
        self.visits = 0
        self.untried_moves = None

class MCTSAI:
    def __init__(self, player, time_limit=2.5):
        self.player = player
        self.time_limit = time_limit
        # Re-use AdvancedAI's static board evaluator as the "Value Network" substitute
        self.evaluator = AdvancedAI(player=player, depth=1)
        self.latest_win_rate = 50.0

    def get_best_move(self, game):
        start_time = time.time()
        
        root_candidates = self._get_candidates(game)
        if not root_candidates:
            return (game.board_size // 2, game.board_size // 2)
            
        root = MCTSNode(parent=None, move=None, player_just_moved=3 - self.player)
        root.untried_moves = root_candidates
        
        iterations = 0
        
        while time.time() - start_time < self.time_limit:
            node = root
            
            # 1. SELECTION
            # Traverse down the tree to a node that has untried child nodes
            while node.untried_moves is not None and len(node.untried_moves) == 0 and len(node.children) > 0:
                node = self._select_child(node)
                game.board[node.move[0]][node.move[1]] = node.player_just_moved
                
            # 2. EXPANSION
            if node.untried_moves is None:
                node.untried_moves = self._get_candidates(game)
                
            if node.untried_moves:
                move = random.choice(node.untried_moves)
                node.untried_moves.remove(move)
                
                next_player = 3 - node.player_just_moved
                child = MCTSNode(parent=node, move=move, player_just_moved=next_player)
                node.children.append(child)
                node = child
                
                game.board[move[0]][move[1]] = next_player

            # 3. SIMULATION / EVALUATION (Heuristic Value Substitute)
            # Instead of full random playouts, use heuristic evaluation from AdvancedAI
            score = self.evaluator._evaluate_board(game)
            
            if node.move:
                r, c = node.move
                if game.check_winner(r, c):
                    # Direct checkmate found
                    score = 100000 if node.player_just_moved == self.player else -100000
                elif game.check_draw():
                    score = 0
                    
            # Map deterministic heuristic score to a [0, 1] probability curve using Sigmoid
            # Positive score means good for 'self.player'
            exponent = max(min(-score / 2000.0, 100), -100) # Prevents math range overflow
            win_prob = 1.0 / (1.0 + math.exp(exponent))
            
            # 4. BACKPROPAGATION
            while node is not None:
                node.visits += 1
                if node.player_just_moved == self.player:
                    node.wins += win_prob
                else:
                    node.wins += (1.0 - win_prob)
                    
                if node.move is not None:
                    # Undo move trace
                    game.board[node.move[0]][node.move[1]] = 0
                    
                node = node.parent
                
            iterations += 1
            
        print(f"[MCTSAI DeepMind-Style] Iterations: {iterations} within {self.time_limit}s.")
        
        if not root.children:
            return random.choice(root_candidates)
            
        # Select best node based on max visits (most robust behavior for MCTS)
        best_child = max(root.children, key=lambda c: c.visits)
        
        if best_child.player_just_moved == self.player:
            expected_win_ratio = best_child.wins / best_child.visits
        else:
            expected_win_ratio = 1.0 - (best_child.wins / best_child.visits)
            
        self.latest_win_rate = expected_win_ratio * 100.0
        
        return best_child.move
        
    def _select_child(self, node):
        best_score = -float('inf')
        best_child = None
        for child in node.children:
            # UCB1 Formula (Upper Confidence Bound)
            exploit = child.wins / child.visits
            explore = math.sqrt(2.0 * math.log(node.visits) / child.visits)
            ucb_score = exploit + 1.414 * explore
            
            if ucb_score > best_score:
                best_score = ucb_score
                best_child = child
        return best_child

    def _get_candidates(self, game):
        # MCTS uses 1-unit radius exclusively to ensure it scales deeper
        radius = 1
        candidates = []
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
                                candidates.append((nr, nc))
        if not has_stone:
            return []
        return list(set(candidates))

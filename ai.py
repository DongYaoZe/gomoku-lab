import random

class BaselineAI:
    def __init__(self, player):
        self.player = player # 1 for Black, 2 for White

    def get_best_move(self, game):
        # 1. 尝试直接赢
        best_move = self._find_winning_move(game, self.player)
        if best_move:
            return best_move
            
        # 2. 尝试防守被连赢
        opponent = 3 - self.player
        best_move = self._find_winning_move(game, opponent)
        if best_move:
            return best_move
            
        # 3. 附近下子
        nearby_moves = self._get_nearby_empty_cells(game)
        if nearby_moves:
            return random.choice(nearby_moves)
            
        # 4. 兜底，随便找个能下的
        empty_cells = []
        for r in range(game.board_size):
            for c in range(game.board_size):
                if game.board[r][c] == 0:
                    empty_cells.append((r, c))
        if empty_cells:
            return random.choice(empty_cells)
        return None

    def _find_winning_move(self, game, player_to_check):
        """如果player_to_check落子能连五，返回坐标"""
        for r in range(game.board_size):
            for c in range(game.board_size):
                if game.board[r][c] == 0:
                    # 模拟落子
                    game.board[r][c] = player_to_check
                    is_win = game.check_winner(r, c)
                    game.board[r][c] = 0 # 恢复
                    if is_win:
                        return (r, c)
        return None

    def _get_nearby_empty_cells(self, game):
        """寻找已有棋子周围半径为1的空位"""
        nearby = set()
        directions = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1)
        ]
        
        for r in range(game.board_size):
            for c in range(game.board_size):
                if game.board[r][c] != 0:
                    for dr, dc in directions:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < game.board_size and 0 <= nc < game.board_size and game.board[nr][nc] == 0:
                            nearby.add((nr, nc))
        return list(nearby)

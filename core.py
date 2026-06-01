import random as _random

class GomokuGame:
    _zobrist_table = None
    _zobrist_player_key = None
    _zobrist_seed = 42

    @classmethod
    def _init_zobrist(cls, board_size=15):
        if cls._zobrist_table is not None:
            return
        rng = _random.Random(cls._zobrist_seed)
        cls._zobrist_table = [
            [[rng.getrandbits(64) for _ in range(2)] for _ in range(board_size)]
            for _ in range(board_size)
        ]
        cls._zobrist_player_key = rng.getrandbits(64)

    def __init__(self, board_size=15):
        self._init_zobrist(board_size)
        self.board_size = board_size
        self.board = [[0] * board_size for _ in range(board_size)]
        self.current_player = 1
        self.winner = 0
        self.history = []
        self.move_count = 0
        self.hash = 0
        self.forbidden_enabled = False

    def reset(self):
        self.board = [[0] * self.board_size for _ in range(self.board_size)]
        self.current_player = 1
        self.winner = 0
        self.history = []
        self.move_count = 0
        self.hash = 0

    def is_valid_move(self, r, c):
        if r < 0 or r >= self.board_size or c < 0 or c >= self.board_size:
            return False
        return self.board[r][c] == 0

    def make_move(self, r, c):
        if not self.is_valid_move(r, c) or self.winner != 0:
            return False

        if self.forbidden_enabled and self.current_player == 1:
            forbidden_msg = self.check_forbidden(r, c)
            if forbidden_msg:
                self.winner = 2
                self.board[r][c] = 1
                self.history.append((r, c, 1))
                self.move_count += 1
                self.hash ^= self._zobrist_table[r][c][0]
                return forbidden_msg

        self.history.append((r, c, self.current_player))
        self.board[r][c] = self.current_player
        self.move_count += 1
        self.hash ^= self._zobrist_table[r][c][self.current_player - 1]

        if self.check_winner(r, c):
            self.winner = self.current_player
        else:
            self.current_player = 3 - self.current_player
        return True

    def undo_move(self):
        if not self.history:
            return False
        r, c, player = self.history.pop()
        self.board[r][c] = 0
        self.move_count -= 1
        self.hash ^= self._zobrist_table[r][c][player - 1]
        self.winner = 0
        self.current_player = player
        return True

    def simulate_move(self, r, c, player):
        """轻量模拟落子（AI搜索用），仅维护 board/hash/move_count"""
        self.board[r][c] = player
        self.move_count += 1
        self.hash ^= self._zobrist_table[r][c][player - 1]

    def undo_simulate(self, r, c, player):
        """撤销 simulate_move"""
        self.board[r][c] = 0
        self.move_count -= 1
        self.hash ^= self._zobrist_table[r][c][player - 1]

    @property
    def full_hash(self):
        """含当前玩家信息的完整哈希，用于转置表键"""
        if self.current_player == 1:
            return self.hash
        return self.hash ^ self._zobrist_player_key

    def check_winner(self, r, c):
        player = self.board[r][c]
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]

        for dr, dc in directions:
            count = 1
            nr, nc = r + dr, c + dc
            while 0 <= nr < self.board_size and 0 <= nc < self.board_size and self.board[nr][nc] == player:
                count += 1
                nr += dr
                nc += dc
            nr, nc = r - dr, c - dc
            while 0 <= nr < self.board_size and 0 <= nc < self.board_size and self.board[nr][nc] == player:
                count += 1
                nr -= dr
                nc -= dc
            if count >= 5:
                return True
        return False

    def check_draw(self):
        return self.move_count >= self.board_size * self.board_size

    def check_forbidden(self, r, c):
        """禁手检测：长连、四四、三三（仅黑棋）"""
        if self.current_player != 1:
            return False

        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
        size = self.board_size

        self.board[r][c] = 1

        overs, fours, open_threes = 0, 0, 0

        for dr, dc in directions:
            s = []
            for i in range(-4, 5):
                nr, nc = r + i * dr, c + i * dc
                if 0 <= nr < size and 0 <= nc < size:
                    s.append(str(self.board[nr][nc]))
                else:
                    s.append('2')
            line = "".join(s)

            # 恰好五连不是禁手，直接放行
            if "11111" in line and "111111" not in line:
                self.board[r][c] = 0
                return False

            if "111111" in line:
                overs += 1

            four_patterns = ["01111", "11110", "10111", "11011", "11101"]
            if any(p in line for p in four_patterns):
                fours += 1

            three_patterns = ["011100", "001110", "010110", "011010"]
            if any(p in line for p in three_patterns):
                open_threes += 1

        self.board[r][c] = 0

        if overs > 0:
            return "长连禁手"
        if fours >= 2:
            return "四四禁手"
        if open_threes >= 2:
            return "三三禁手"
        return False

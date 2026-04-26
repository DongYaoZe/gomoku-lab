class GomokuGame:
    def __init__(self, board_size=15):
        self.board_size = board_size
        self.board = [[0 for _ in range(board_size)] for _ in range(board_size)]
        self.current_player = 1  # 1 is Black, 2 is White
        self.winner = 0
        self.history = []
        self.forbidden_enabled = False

    def reset(self):
        self.board = [[0 for _ in range(self.board_size)] for _ in range(self.board_size)]
        self.current_player = 1
        self.winner = 0
        self.history = []

    def is_valid_move(self, r, c):
        if r < 0 or r >= self.board_size or c < 0 or c >= self.board_size:
            return False
        if self.board[r][c] != 0:
            return False
        return True

    def make_move(self, r, c):
        if not self.is_valid_move(r, c) or self.winner != 0:
            return False
            
        if self.forbidden_enabled and self.current_player == 1:
            forbidden_msg = self.check_forbidden(r, c)
            if forbidden_msg:
                self.winner = 2 # Black loses instantly on forbidden move
                self.board[r][c] = 1
                self.history.append((r, c, 1))
                return forbidden_msg
        
        self.history.append((r, c, self.current_player))
        self.board[r][c] = self.current_player
        if self.check_winner(r, c):
            self.winner = self.current_player
        else:
            self.current_player = 3 - self.current_player # Swap between 1 and 2
        return True

    def undo_move(self):
        if not self.history:
            return False
        r, c, player = self.history.pop()
        self.board[r][c] = 0
        self.winner = 0 # Reset winner if undoing
        self.current_player = player
        return True

    def check_winner(self, r, c):
        directions = [
            (0, 1),   # Horizontal
            (1, 0),   # Vertical
            (1, 1),   # Main Diagonal
            (1, -1)   # Anti-Diagonal
        ]
        
        player = self.board[r][c]
        
        for dr, dc in directions:
            count = 1
            # Check positive direction
            nr, nc = r + dr, c + dc
            while 0 <= nr < self.board_size and 0 <= nc < self.board_size and self.board[nr][nc] == player:
                count += 1
                nr += dr
                nc += dc
                
            # Check negative direction
            nr, nc = r - dr, c - dc
            while 0 <= nr < self.board_size and 0 <= nc < self.board_size and self.board[nr][nc] == player:
                count += 1
                nr -= dr
                nc -= dc
                
            if count >= 5:
                return True
                
        return False
        
    def check_draw(self):
        for r in range(self.board_size):
            for c in range(self.board_size):
                if self.board[r][c] == 0:
                    return False
        return True

    def check_forbidden(self, r, c):
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
            
            if "11111" in line and "111111" not in line:
                self.board[r][c] = 0
                return False
                
            if "111111" in line: overs += 1
                
            four_patterns = ["01111", "11110", "10111", "11011", "11101"]
            if any(p in line for p in four_patterns): fours += 1
                
            three_patterns = ["011100", "001110", "010110", "011010"]
            if any(p in line for p in three_patterns): open_threes += 1
                
        self.board[r][c] = 0
        
        if overs > 0: return "长连禁手"
        if fours >= 2: return "四四禁手"
        if open_threes >= 2: return "三三禁手"
        return False


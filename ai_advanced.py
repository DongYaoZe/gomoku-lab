import time

class AdvancedAI:
    def __init__(self, player, depth=3):
        self.player = player
        self.depth = depth
        self.latest_win_rate = 50.0
        self.memo = {}
        self.vcf_enabled = False

    def get_best_move(self, game):
        start_time = time.time()
        
        if self.vcf_enabled:
            vcf_best = self.find_vcf(game, self.player, depth=11)
            if vcf_best:
                print(f"[AdvancedAI] VCF Killer sequence found! Initiating sequence at {vcf_best}")
                end_time = time.time()
                self.latest_win_rate = 99.9
                return vcf_best
        
        best_val = -float('inf')
        best_move = None
        self.memo.clear()
        
        candidates = self._get_candidates(game)
        if not candidates:
            return (game.board_size // 2, game.board_size // 2)
            
        alpha = -float('inf')
        beta = float('inf')
        
        scored_candidates = []
        for r, c in candidates:
            score = self._evaluate_point(game.board, r, c, self.player) + \
                    self._evaluate_point(game.board, r, c, 3 - self.player)
            scored_candidates.append((score, r, c))
            
        scored_candidates.sort(reverse=True, key=lambda x: x[0])
        
        for score, r, c in scored_candidates:
            prev_winner = game.winner
            game.board[r][c] = self.player
            if game.check_winner(r, c):
                game.winner = self.player
            game.current_player = 3 - self.player
            
            val = self._minimax(game, self.depth - 1, alpha, beta, False)
            
            game.board[r][c] = 0
            game.winner = prev_winner
            game.current_player = self.player

            if val > best_val:
                best_val = val
                best_move = (r, c)
            
            alpha = max(alpha, best_val)
            if alpha >= beta:
                break
                
        clamped_val = max(-10000, min(10000, best_val))
        self.latest_win_rate = 50.0 + (clamped_val / 200.0)
        
        end_time = time.time()
        print(f"[AdvancedAI Depth {self.depth}] Decision took {end_time - start_time:.3f}s. Selected: {best_move} with val: {best_val}. Cache: {len(self.memo)}")
        return best_move

    def _minimax(self, game, depth, alpha, beta, is_maximizing):
        if game.winner == self.player:
            return 1000000 + depth  
        elif game.winner == 3 - self.player:
            return -1000000 - depth 
        elif game.check_draw():
            return 0
            
        if depth == 0:
            return self._evaluate_board(game)

        board_tuple = tuple(tuple(row) for row in game.board)
        if board_tuple in self.memo:
            cached_depth, cached_val = self.memo[board_tuple]
            if cached_depth >= depth:
                return cached_val

        candidates = self._get_candidates(game)
        
        if is_maximizing:
            max_eval = -float('inf')
            scored_candidates = []
            for r, c in candidates:
                sc = self._evaluate_point(game.board, r, c, self.player) + self._evaluate_point(game.board, r, c, 3 - self.player)
                scored_candidates.append((sc, r, c))
            scored_candidates.sort(reverse=True, key=lambda x: x[0])
            
            for score, r, c in scored_candidates:
                prev_winner = game.winner
                game.board[r][c] = self.player
                if game.check_winner(r, c):
                    game.winner = self.player
                game.current_player = 3 - self.player
                
                eval_val = self._minimax(game, depth - 1, alpha, beta, False)
                
                game.board[r][c] = 0
                game.winner = prev_winner
                game.current_player = self.player
                
                max_eval = max(max_eval, eval_val)
                alpha = max(alpha, eval_val)
                if beta <= alpha:
                    break
                    
            self.memo[board_tuple] = (depth, max_eval)
            return max_eval
        else:
            min_eval = float('inf')
            opponent = 3 - self.player
            
            scored_candidates = []
            for r, c in candidates:
                sc = self._evaluate_point(game.board, r, c, opponent) + self._evaluate_point(game.board, r, c, self.player)
                scored_candidates.append((sc, r, c))
            scored_candidates.sort(reverse=True, key=lambda x: x[0])
            
            for score, r, c in scored_candidates:
                prev_winner = game.winner
                game.board[r][c] = opponent
                if game.check_winner(r, c):
                    game.winner = opponent
                game.current_player = self.player
                
                eval_val = self._minimax(game, depth - 1, alpha, beta, True)
                
                game.board[r][c] = 0
                game.winner = prev_winner
                game.current_player = opponent
                
                min_eval = min(min_eval, eval_val)
                beta = min(beta, eval_val)
                if beta <= alpha:
                    break
                    
            self.memo[board_tuple] = (depth, min_eval)
            return min_eval

    def find_vcf(self, game, player, depth=10):
        if depth <= 0:
            return None
            
        opponent = 3 - player
        candidates = self._get_candidates(game)
        
        opponent_wins = []
        for r, c in candidates:
            game.board[r][c] = opponent
            if game.check_winner(r, c):
                opponent_wins.append((r, c))
            game.board[r][c] = 0
            
        for r, c in candidates:
            game.board[r][c] = player
            if game.check_winner(r, c):
                game.board[r][c] = 0
                return (r, c)
            game.board[r][c] = 0
            
        if len(opponent_wins) > 1:
            return None
        if len(opponent_wins) == 1:
            candidates = [opponent_wins[0]]
            
        for r, c in candidates:
            score = self._evaluate_point(game.board, r, c, player)
            if score >= 1000:
                game.board[r][c] = player
                
                block_r, block_c = -1, -1
                for opp_r, opp_c in candidates:
                    if opp_r == r and opp_c == c: continue
                    game.board[opp_r][opp_c] = player
                    if game.check_winner(opp_r, opp_c):
                        block_r, block_c = opp_r, opp_c
                        game.board[opp_r][opp_c] = 0
                        break
                    game.board[opp_r][opp_c] = 0
                    
                if block_r != -1:
                    game.board[block_r][block_c] = opponent
                    if not game.check_winner(block_r, block_c):
                        sub_vcf = self.find_vcf(game, player, depth - 2)
                        game.board[block_r][block_c] = 0
                        game.board[r][c] = 0
                        if sub_vcf is not None:
                            return (r, c)
                    else:
                        game.board[block_r][block_c] = 0
                    
                game.board[r][c] = 0
        return None

    def _get_candidates(self, game):
        """仅返回已有棋子周围半径为1或2的空位"""
        radius = 1 if self.depth >= 4 else 2
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
        return list(candidates)

    def _evaluate_board(self, game):
        my_score = 0
        opp_score = 0
        opponent = 3 - self.player
        board = game.board
        
        for r in range(game.board_size):
            for c in range(game.board_size):
                if board[r][c] == self.player:
                    my_score += self._evaluate_point(board, r, c, self.player)
                elif board[r][c] == opponent:
                    opp_score += self._evaluate_point(board, r, c, opponent)
                    
        return my_score - opp_score

    def _evaluate_point(self, board, r, c, focus_player):
        score = 0
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
        size = len(board)
        
        for dr, dc in directions:
            consecutive = 1
            blocks = 0
            
            nr, nc = r + dr, c + dc
            while 0 <= nr < size and 0 <= nc < size and board[nr][nc] == focus_player:
                consecutive += 1
                nr += dr; nc += dc
            if nr < 0 or nr >= size or nc < 0 or nc >= size or board[nr][nc] != 0:
                blocks += 1
                
            nr, nc = r - dr, c - dc
            while 0 <= nr < size and 0 <= nc < size and board[nr][nc] == focus_player:
                consecutive += 1
                nr -= dr; nc -= dc
            if nr < 0 or nr >= size or nc < 0 or nc >= size or board[nr][nc] != 0:
                blocks += 1
                
            if consecutive >= 5:
                score += 100000
            elif consecutive == 4:
                if blocks == 0:
                    score += 10000
                elif blocks == 1:
                    score += 1000
            elif consecutive == 3:
                if blocks == 0:
                    score += 1000
                elif blocks == 1:
                    score += 100
            elif consecutive == 2:
                if blocks == 0:
                    score += 100
                elif blocks == 1:
                    score += 10
                    
        return score

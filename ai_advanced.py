import time
import math

EXACT, LOWER_BOUND, UPPER_BOUND = 0, 1, 2


class AdvancedAI:
    def __init__(self, player, depth=3, time_limit=5.0):
        self.player = player
        self.depth = depth
        self.time_limit = time_limit
        self.latest_win_rate = 50.0
        self.memo = {}
        self.vcf_enabled = False
        self.killer_moves = [[None, None] for _ in range(20)]
        self.history_table = [[0] * 225 for _ in range(2)]
        self._start_time = 0
        self._timed_out = False

    def get_best_move(self, game):
        self._start_time = time.time()
        self._timed_out = False
        self.memo.clear()

        if self.vcf_enabled:
            vcf_best = self.find_vcf(game, self.player, depth=11)
            if vcf_best:
                print(f"[AdvancedAI] VCF forced win at {vcf_best}")
                self.latest_win_rate = 99.9 if self.player == 1 else 0.1
                return vcf_best

        candidates = self._get_candidates(game)
        if not candidates:
            return (game.board_size // 2, game.board_size // 2)

        best_move = candidates[0] if candidates else None
        best_val = -float('inf')

        # Iterative deepening
        for current_depth in range(1, self.depth + 1):
            if time.time() - self._start_time > self.time_limit * 0.8:
                break

            alpha = -float('inf')
            beta = float('inf')
            depth_best_move = None
            depth_best_val = -float('inf')

            scored = self._order_moves(game, candidates, self.player, current_depth)

            for r, c in scored:
                if self._timed_out:
                    break

                game.board[r][c] = self.player
                game.move_count += 1
                game.hash ^= game._zobrist_table[r][c][self.player - 1]
                won = game.check_winner(r, c)
                prev_winner = game.winner
                if won:
                    game.winner = self.player

                val = self._minimax(game, current_depth - 1, alpha, beta, False, r, c)

                game.board[r][c] = 0
                game.move_count -= 1
                game.hash ^= game._zobrist_table[r][c][self.player - 1]
                game.winner = prev_winner

                if val > depth_best_val:
                    depth_best_val = val
                    depth_best_move = (r, c)

                alpha = max(alpha, val)
                if alpha >= beta:
                    break

            if not self._timed_out and depth_best_move:
                best_move = depth_best_move
                best_val = depth_best_val

        clamped = max(-100000, min(100000, best_val))
        # Sigmoid 映射：val 为己方视角，转换为黑方胜率
        self_win_prob = 1.0 / (1.0 + math.exp(-clamped / 5000.0))
        if self.player == 1:
            self.latest_win_rate = self_win_prob * 100.0
        else:
            self.latest_win_rate = (1.0 - self_win_prob) * 100.0
        self.latest_win_rate = max(0.1, min(99.9, self.latest_win_rate))

        elapsed = time.time() - self._start_time
        print(f"[AdvancedAI Depth {self.depth}] {elapsed:.3f}s | Move: {best_move} | Val: {best_val} | Cache: {len(self.memo)}")
        return best_move

    def _minimax(self, game, depth, alpha, beta, is_maximizing, last_r, last_c):
        if time.time() - self._start_time > self.time_limit:
            self._timed_out = True
            return 0

        if game.winner == self.player:
            return 1000000 + depth
        elif game.winner == 3 - self.player:
            return -1000000 - depth
        elif game.check_draw():
            return 0
        if depth == 0:
            return self._evaluate_board(game)

        # Transposition table lookup
        tt_key = game.hash
        if tt_key in self.memo:
            cached_depth, cached_val, flag = self.memo[tt_key]
            if cached_depth >= depth:
                if flag == EXACT:
                    return cached_val
                elif flag == LOWER_BOUND and cached_val >= beta:
                    return cached_val
                elif flag == UPPER_BOUND and cached_val <= alpha:
                    return cached_val

        candidates = self._get_candidates(game)

        if is_maximizing:
            ordered = self._order_moves(game, candidates, self.player, depth)
            max_eval = -float('inf')
            orig_alpha = alpha

            for r, c in ordered:
                if self._timed_out:
                    return 0

                game.board[r][c] = self.player
                game.move_count += 1
                game.hash ^= game._zobrist_table[r][c][self.player - 1]
                prev_winner = game.winner
                if game.check_winner(r, c):
                    game.winner = self.player

                val = self._minimax(game, depth - 1, alpha, beta, False, r, c)

                game.board[r][c] = 0
                game.move_count -= 1
                game.hash ^= game._zobrist_table[r][c][self.player - 1]
                game.winner = prev_winner

                if val > max_eval:
                    max_eval = val
                alpha = max(alpha, val)
                if alpha >= beta:
                    self._update_killer(depth, r, c)
                    self.history_table[self.player - 1][r * 15 + c] += depth * depth
                    break

            if not self._timed_out:
                flag = EXACT
                if max_eval <= orig_alpha:
                    flag = UPPER_BOUND
                elif max_eval >= beta:
                    flag = LOWER_BOUND
                self.memo[tt_key] = (depth, max_eval, flag)
            return max_eval
        else:
            opponent = 3 - self.player
            ordered = self._order_moves(game, candidates, opponent, depth)
            min_eval = float('inf')
            orig_beta = beta

            for r, c in ordered:
                if self._timed_out:
                    return 0

                game.board[r][c] = opponent
                game.move_count += 1
                game.hash ^= game._zobrist_table[r][c][opponent - 1]
                prev_winner = game.winner
                if game.check_winner(r, c):
                    game.winner = opponent

                val = self._minimax(game, depth - 1, alpha, beta, True, r, c)

                game.board[r][c] = 0
                game.move_count -= 1
                game.hash ^= game._zobrist_table[r][c][opponent - 1]
                game.winner = prev_winner

                if val < min_eval:
                    min_eval = val
                beta = min(beta, val)
                if alpha >= beta:
                    self._update_killer(depth, r, c)
                    self.history_table[opponent - 1][r * 15 + c] += depth * depth
                    break

            if not self._timed_out:
                flag = EXACT
                if min_eval >= orig_beta:
                    flag = LOWER_BOUND
                elif min_eval <= alpha:
                    flag = UPPER_BOUND
                self.memo[tt_key] = (depth, min_eval, flag)
            return min_eval

    def _update_killer(self, depth, r, c):
        move = (r, c)
        if depth < len(self.killer_moves):
            if self.killer_moves[depth][0] != move:
                self.killer_moves[depth][1] = self.killer_moves[depth][0]
                self.killer_moves[depth][0] = move

    def _order_moves(self, game, candidates, player, depth):
        scored = []
        for r, c in candidates:
            score = self._evaluate_point(game.board, r, c, player) * 2 + \
                    self._evaluate_point(game.board, r, c, 3 - player)
            # Killer move bonus
            if depth < len(self.killer_moves):
                if (r, c) == self.killer_moves[depth][0]:
                    score += 80000
                elif (r, c) == self.killer_moves[depth][1]:
                    score += 60000
            # History heuristic
            score += self.history_table[player - 1][r * 15 + c]
            scored.append((score, r, c))
        scored.sort(reverse=True, key=lambda x: x[0])
        return [(r, c) for _, r, c in scored]

    def find_vcf(self, game, player, depth=11):
        """威胁空间搜索：连续冲四迫使对手防守直至成五"""
        if depth <= 0:
            return None

        opponent = 3 - player
        candidates = self._get_candidates(game)

        # 先检查对手是否有即时杀
        for r, c in candidates:
            game.board[r][c] = opponent
            if game.check_winner(r, c):
                game.board[r][c] = 0
                # 对手有杀，必须先堵
                game.board[r][c] = player
                if game.check_winner(r, c):
                    game.board[r][c] = 0
                    return (r, c)
                game.board[r][c] = 0
                return None
            game.board[r][c] = 0

        # 检查己方是否有即时杀
        for r, c in candidates:
            game.board[r][c] = player
            if game.check_winner(r, c):
                game.board[r][c] = 0
                return (r, c)
            game.board[r][c] = 0

        # 寻找冲四点（落子后形成四连，对手必须防守）
        for r, c in candidates:
            score = self._evaluate_point(game.board, r, c, player)
            if score < 1000:
                continue

            game.board[r][c] = player
            game.move_count += 1

            # 找到对手必须防守的点
            defense_points = self._find_forced_defenses(game, player, r, c, candidates)

            if len(defense_points) == 0:
                # 无法防守 = 胜利
                game.board[r][c] = 0
                game.move_count -= 1
                return (r, c)
            elif len(defense_points) == 1:
                dr, dc = defense_points[0]
                game.board[dr][dc] = opponent
                game.move_count += 1

                if not game.check_winner(dr, dc):
                    sub = self.find_vcf(game, player, depth - 2)
                    game.board[dr][dc] = 0
                    game.move_count -= 1
                    game.board[r][c] = 0
                    game.move_count -= 1
                    if sub is not None:
                        return (r, c)
                else:
                    game.board[dr][dc] = 0
                    game.move_count -= 1

                if game.board[r][c] == player:
                    game.board[r][c] = 0
                    game.move_count -= 1
            else:
                game.board[r][c] = 0
                game.move_count -= 1

        return None

    def _find_forced_defenses(self, game, attacker, ar, ac, candidates):
        """找到攻击者落子后对手必须防守的点"""
        defenses = []
        for r, c in candidates:
            if game.board[r][c] != 0:
                continue
            if r == ar and c == ac:
                continue
            game.board[r][c] = attacker
            if game.check_winner(r, c):
                defenses.append((r, c))
            game.board[r][c] = 0
        return defenses

    def _get_candidates(self, game):
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
        size = game.board_size
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        counted = set()

        for r in range(size):
            for c in range(size):
                if board[r][c] == 0:
                    continue
                player = board[r][c]
                for di, (dr, dc) in enumerate(directions):
                    key = (r, c, di)
                    if key in counted:
                        continue
                    score = self._score_line_from(board, r, c, dr, dc, player, size)
                    if score > 0:
                        # Mark all stones in this line as counted for this direction
                        nr, nc = r, c
                        while 0 <= nr < size and 0 <= nc < size and board[nr][nc] == player:
                            counted.add((nr, nc, di))
                            nr += dr
                            nc += dc
                        if player == self.player:
                            my_score += score
                        else:
                            opp_score += score

        return my_score - opp_score * 1.1

    def _score_line_from(self, board, r, c, dr, dc, player, size):
        """从 (r,c) 沿正方向评估一条连续线段"""
        consecutive = 0
        nr, nc = r, c
        while 0 <= nr < size and 0 <= nc < size and board[nr][nc] == player:
            consecutive += 1
            nr += dr
            nc += dc

        if consecutive < 2:
            return 0

        # Check blocks at both ends
        blocks = 0
        # Positive end
        if not (0 <= nr < size and 0 <= nc < size) or board[nr][nc] != 0:
            blocks += 1
        # Negative end
        br, bc = r - dr, c - dc
        if not (0 <= br < size and 0 <= bc < size) or board[br][bc] != 0:
            blocks += 1

        if blocks == 2:
            return 0

        if consecutive >= 5:
            return 1000000
        elif consecutive == 4:
            return 50000 if blocks == 0 else 5000
        elif consecutive == 3:
            return 5000 if blocks == 0 else 500
        elif consecutive == 2:
            return 200 if blocks == 0 else 50
        return 0

    def _evaluate_point(self, board, r, c, focus_player):
        """评估某个空位对 focus_player 的价值"""
        score = 0
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
        size = len(board)

        for dr, dc in directions:
            # 提取以 (r,c) 为中心的 9 格线段
            line = []
            for i in range(-4, 5):
                nr, nc = r + i * dr, c + i * dc
                if 0 <= nr < size and 0 <= nc < size:
                    line.append(board[nr][nc])
                else:
                    line.append(3 - focus_player)  # 边界视为对手

            # 模拟落子
            line[4] = focus_player
            score += self._pattern_score(line, focus_player)

        # 中心位置微小加分
        center = size // 2
        dist = abs(r - center) + abs(c - center)
        score += max(0, 14 - dist)

        return score

    def _pattern_score(self, line, player):
        """对 9 格线段进行模式匹配评分"""
        s = ""
        for v in line:
            if v == player:
                s += "1"
            elif v == 0:
                s += "0"
            else:
                s += "2"

        score = 0

        # 五连
        if "11111" in s:
            return 1000000

        # 活四 (两端开放的四连)
        if "011110" in s:
            score += 50000

        # 冲四 (一端被堵的四连或有间隔的四)
        four_patterns = ["211110", "011112", "10111", "11011", "11101"]
        for p in four_patterns:
            if p in s:
                score += 5000
                break

        # 活三
        three_open = ["01110", "010110", "011010"]
        for p in three_open:
            if p in s:
                score += 5000
                break

        # 眠三
        three_dead = ["21110", "01112", "10110", "01011", "11010", "01011"]
        for p in three_dead:
            if p in s:
                score += 500
                break

        # 活二
        if "00110" in s or "01100" in s or "01010" in s or "010010" in s:
            score += 200

        # 眠二
        if "21100" in s or "00112" in s or "10010" in s or "01001" in s:
            score += 50

        return score


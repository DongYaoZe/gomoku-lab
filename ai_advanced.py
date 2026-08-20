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

                game.simulate_move(r, c, self.player)
                won = game.check_winner(r, c)
                prev_winner = game.winner
                if won:
                    game.winner = self.player

                val = self._minimax(game, current_depth - 1, alpha, beta, False, r, c)

                game.undo_simulate(r, c, self.player)
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
        tt_key = game.full_hash
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

                game.simulate_move(r, c, self.player)
                prev_winner = game.winner
                if game.check_winner(r, c):
                    game.winner = self.player

                val = self._minimax(game, depth - 1, alpha, beta, False, r, c)

                game.undo_simulate(r, c, self.player)
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

                game.simulate_move(r, c, opponent)
                prev_winner = game.winner
                if game.check_winner(r, c):
                    game.winner = opponent

                val = self._minimax(game, depth - 1, alpha, beta, True, r, c)

                game.undo_simulate(r, c, opponent)
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
        opponent = 3 - player
        for r, c in candidates:
            attack = self._evaluate_point(game.board, r, c, player)
            defense = self._evaluate_point(game.board, r, c, opponent)
            # 取攻防最大值，确保关键防守点不被漏掉
            score = max(attack, defense) + min(attack, defense) * 0.5
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
        # Top-N 剪枝：深层搜索只保留最有价值的候选
        max_width = 15 if self.depth >= 4 else 25
        return [(r, c) for _, r, c in scored[:max_width]]

    def _vcf_move_is_legal(self, game, player, r, c):
        """Return whether a hypothetical VCF move is legal under the repo's optional forbidden mode."""
        if game.board[r][c] != 0:
            return False
        if not game.forbidden_enabled or player != 1:
            return True

        previous_player = game.current_player
        game.current_player = 1
        try:
            return not bool(game.check_forbidden(r, c))
        finally:
            game.current_player = previous_player

    def _vcf_winning_completions(self, game, player):
        """List legal empty points where ``player`` wins immediately."""
        wins = []
        for r, c in sorted(self._get_candidates(game)):
            if not self._vcf_move_is_legal(game, player, r, c):
                continue
            game.simulate_move(r, c, player)
            try:
                if game.check_winner(r, c):
                    wins.append((r, c))
            finally:
                game.undo_simulate(r, c, player)
        return wins

    def find_vcf(self, game, player, depth=11):
        """Search continuous-four forcing lines without treating non-threats as wins."""
        if depth <= 0:
            return None

        opponent = 3 - player
        candidates = sorted(self._get_candidates(game))

        # The side to move always takes a legal immediate win first, even if the
        # opponent also threatens to win on their next turn.
        own_wins = self._vcf_winning_completions(game, player)
        if own_wins:
            return own_wins[0]

        # Once the opponent already has an immediate legal win, a pure VCF
        # attack is not a valid substitute for defending it.
        if self._vcf_winning_completions(game, opponent):
            return None

        # A VCF move must create at least one *actual legal winning completion*.
        # Two or more completions form an unstoppable double threat.  Exactly
        # one completion means the defender has one forced blocking point.
        for r, c in candidates:
            if not self._vcf_move_is_legal(game, player, r, c):
                continue
            score = self._evaluate_point(game.board, r, c, player)
            if score < 1000:
                continue

            game.simulate_move(r, c, player)
            try:
                completions = self._vcf_winning_completions(game, player)
                if len(completions) >= 2:
                    return (r, c)
                if len(completions) != 1:
                    continue

                dr, dc = completions[0]
                # If the only nominal block is itself illegal (notably a black
                # forbidden move), the defender has no legal reply.
                if not self._vcf_move_is_legal(game, opponent, dr, dc):
                    return (r, c)

                game.simulate_move(dr, dc, opponent)
                try:
                    # A forced block that wins for the defender refutes the line.
                    if game.check_winner(dr, dc):
                        continue
                    if self.find_vcf(game, player, depth - 2) is not None:
                        return (r, c)
                finally:
                    game.undo_simulate(dr, dc, opponent)
            finally:
                game.undo_simulate(r, c, player)

        return None

    def _find_forced_defenses(self, game, attacker, ar, ac, candidates=None):
        """Compatibility helper: legal immediate winning completions after an attack."""
        return self._vcf_winning_completions(game, attacker)

    def _get_candidates(self, game):
        # 统一使用半径2，通过 top-N 排序控制分支因子
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
        return list(candidates)

    def _evaluate_board(self, game):
        """全盘评估：扫描所有方向线段，累计双方得分"""
        my_score = 0
        opp_score = 0
        opponent = 3 - self.player
        board = game.board
        size = game.board_size
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]

        for di, (dr, dc) in enumerate(directions):
            # 遍历该方向所有起始线段
            for r in range(size):
                for c in range(size):
                    # 只从线段起点开始（避免重复计数）
                    pr, pc = r - dr, c - dc
                    if 0 <= pr < size and 0 <= pc < size and board[pr][pc] != 0:
                        continue
                    if board[r][c] == 0:
                        continue
                    player = board[r][c]
                    score = self._score_line_from(board, r, c, dr, dc, player, size)
                    if player == self.player:
                        my_score += score
                    else:
                        opp_score += score

        return my_score - opp_score * 1.3

    def _score_line_from(self, board, r, c, dr, dc, player, size):
        """从 (r,c) 沿正方向提取线段并评分（含间隔棋形）"""
        # 提取最多 9 格的线段上下文
        cells = []
        nr, nc = r - dr, c - dc
        # 前方一格
        if 0 <= nr < size and 0 <= nc < size:
            cells.append(board[nr][nc])
        else:
            cells.append(3 - player)
        # 从 (r,c) 开始向正方向取 6 格
        nr, nc = r, c
        for _ in range(6):
            if 0 <= nr < size and 0 <= nc < size:
                cells.append(board[nr][nc])
            else:
                cells.append(3 - player)
            nr += dr
            nc += dc

        # 转为字符串: 1=己方, 0=空, 2=对手/边界
        s = ""
        for v in cells:
            if v == player:
                s += "1"
            elif v == 0:
                s += "0"
            else:
                s += "2"

        return self._line_pattern_score(s)

    def _line_pattern_score(self, s):
        """对一条方向线段字符串评分"""
        # 成五
        if "11111" in s:
            return 100000

        score = 0
        # 活四
        if "011110" in s:
            score += 50000
        # 冲四（含跳四）
        elif "11110" in s or "01111" in s:
            score += 6000
        elif "11101" in s or "10111" in s or "11011" in s:
            score += 5500

        # 活三（含跳活三）
        if "01110" in s or "011010" in s or "010110" in s:
            score += 6000
        elif "001110" in s or "011100" in s:
            score += 5000

        # 眠三
        if "21110" in s or "01112" in s or "211010" in s or "010112" in s:
            score += 600
        elif "10110" in s or "11010" in s or "01011" in s or "01101" in s:
            score += 500

        # 活二
        if "00110" in s or "01100" in s or "01010" in s or "010010" in s:
            score += 400

        # 眠二
        if "21100" in s or "00112" in s or "010012" in s or "210010" in s:
            score += 80

        return score

    def _evaluate_point(self, board, r, c, focus_player):
        """评估某个空位对 focus_player 的价值（含组合威胁检测）"""
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
        size = len(board)

        total_score = 0
        n_fours = 0      # 冲四数
        n_open_threes = 0  # 活三数
        has_open_four = False  # 活四

        for dr, dc in directions:
            line = []
            for i in range(-4, 5):
                nr, nc = r + i * dr, c + i * dc
                if 0 <= nr < size and 0 <= nc < size:
                    line.append(board[nr][nc])
                else:
                    line.append(3 - focus_player)
            line[4] = focus_player

            dir_score, threats = self._pattern_score(line, focus_player)
            total_score += dir_score

            if threats.get('five'):
                return 1000000
            if threats.get('open_four'):
                has_open_four = True
            n_fours += threats.get('fours', 0)
            n_open_threes += threats.get('open_threes', 0)

        # 组合威胁加分（必胜棋形）
        if has_open_four:
            total_score += 100000
        if n_fours >= 2:
            # 双冲四 = 必胜
            total_score += 90000
        if n_fours >= 1 and n_open_threes >= 1:
            # 四三 = 必胜
            total_score += 80000
        if n_open_threes >= 2:
            # 双活三 = 必胜（非禁手模式下）
            total_score += 70000

        # 中心位置微小加分
        center = size // 2
        dist = abs(r - center) + abs(c - center)
        total_score += max(0, 14 - dist)

        return total_score

    def _pattern_score(self, line, player):
        """对 9 格线段进行全量模式匹配，返回 (分数, 威胁字典)"""
        s = ""
        for v in line:
            if v == player:
                s += "1"
            elif v == 0:
                s += "0"
            else:
                s += "2"

        threats = {}
        score = 0

        # === 成五 ===
        if "11111" in s:
            threats['five'] = True
            return 1000000, threats

        # === 活四（无法防守）===
        if "011110" in s:
            threats['open_four'] = True
            score += 100000
            return score, threats

        # === 冲四（含跳四，迫使对手防守）===
        fours = 0
        # 连续冲四
        four_rush = ["211110", "011112", "11110", "01111"]
        for p in four_rush:
            if p in s:
                fours += 1
                break
        # 跳冲四
        four_jump = ["11101", "10111", "11011"]
        for p in four_jump:
            if p in s:
                fours += 1
                break
        if fours > 0:
            score += 8000 * fours
            threats['fours'] = fours

        # === 活三（一步变活四）===
        open_threes = 0
        # 连活三
        if "011100" in s or "001110" in s or "01110" in s:
            open_threes += 1
        # 跳活三
        if "010110" in s or "011010" in s:
            open_threes += 1
        if open_threes > 0:
            score += 6000 * open_threes
            threats['open_threes'] = open_threes

        # === 眠三（一端被堵或需要跳步）===
        dead_threes = 0
        dead_three_patterns = [
            "211100", "001112",  # 边堵连三
            "210110", "011012",  # 边堵跳三
            "211010", "010112",  # 边堵跳三变体
            "10011", "11001",    # 两端间隔眠三
        ]
        for p in dead_three_patterns:
            if p in s:
                dead_threes += 1
                break
        if dead_threes > 0:
            score += 600

        # === 活二 ===
        open_twos = 0
        open_two_patterns = [
            "001100", "011000", "000110",  # 连活二
            "010100", "001010", "010010",  # 跳活二
        ]
        for p in open_two_patterns:
            if p in s:
                open_twos += 1
                break
        if open_twos > 0:
            score += 400

        # === 眠二 ===
        dead_two_patterns = ["211000", "000112", "210100", "001012", "10001"]
        for p in dead_two_patterns:
            if p in s:
                score += 80
                break

        return score, threats


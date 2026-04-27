import tkinter as tk
from tkinter import messagebox, filedialog
from tkinter import ttk
import datetime
from core import GomokuGame
from ai import BaselineAI
from ai_advanced import AdvancedAI
from ai_mcts import MCTSAI

class GomokuGUI:
    @property
    def mode(self):
        if getattr(self, '_mode_override', None):
            return self._mode_override
        b = self.black_player_var.get()
        w = self.white_player_var.get()
        if b == "玩家" and w == "玩家": return "PvP"
        if b != "玩家" and w != "玩家": return "EvE"
        return "PvE"
        
    @mode.setter
    def mode(self, val):
        if val == "Replay":
            self._mode_override = "Replay"
        else:
            self._mode_override = None

    def __init__(self, root):
        self.root = root
        self.root.title("幸福五子棋")
        self.game = GomokuGame()
        
        self._mode_override = None
        self._replay_mode_active = False
        self.black_player_var = tk.StringVar(value="玩家")
        self.white_player_var = tk.StringVar(value="高级(E)")
        self.theme_var = tk.StringVar(value="木制棋盘")
        
        self.replay_moves = []
        self.replay_index = 0
        
        self.flash_state = True
        self.flash_item = None
        self._flash_loop()
        
        self._instantiate_ai()
        
        self.cell_size = 40
        self.board_margin = 30
        self.canvas_size = self.game.board_size * self.cell_size + self.board_margin * 2
        
        self._setup_menu()
        self._setup_ui()
        self._set_theme()
        
        self._trigger_ai_if_needed()
            
    def _flash_loop(self):
        self.flash_state = not self.flash_state
        if getattr(self, 'flash_item', None):
            try:
                state = "normal" if self.flash_state else "hidden"
                self.canvas.itemconfigure(self.flash_item, state=state)
            except Exception:
                pass
        self.root.after(400, self._flash_loop)

    def _create_ai_by_level(self, player_color, level):
        if level == "玩家": return None
        if level == "深智(MCTS)":
            return MCTSAI(player=player_color, time_limit=5)  # 将时间限制从2.5s提高到5s
        elif level == "大师(M)":
            ai = AdvancedAI(player=player_color, depth=4)
            val = False
            if hasattr(self, 'vcf_var'): val = self.vcf_var.get()
            ai.vcf_enabled = val
            return ai
        elif level == "高级(E)":
            ai = AdvancedAI(player=player_color, depth=3)
            val = False
            if hasattr(self, 'vcf_var'): val = self.vcf_var.get()
            ai.vcf_enabled = val
            return ai
        elif level == "中级(I)":
            return AdvancedAI(player=player_color, depth=2)
        elif level == "深度学习(AlphaZero)":
            from ai_alphazero import AlphaZeroAI
            return AlphaZeroAI(player=player_color, model_file='./current_policy_15x15.model', n_playout=400)
        else: # 初级(B)
            return BaselineAI(player=player_color)

    def _instantiate_ai(self):
        self.ai_black = self._create_ai_by_level(1, self.black_player_var.get())
        self.ai_white = self._create_ai_by_level(2, self.white_player_var.get())
        if self.mode == "PvE":
            self.ai = self.ai_black if self.ai_black else self.ai_white
            if self.ai:
                self.player_color = "White" if self.ai.player == 1 else "Black"

    def _setup_menu(self):
        menubar = tk.Menu(self.root)
        
        # 游戏(G)
        game_menu = tk.Menu(menubar, tearoff=0)
        game_menu.add_command(label="新建(N)", command=self.restart_game)
        game_menu.add_command(label="保存(S)", command=self.save_record)
        game_menu.add_command(label="回放打开(L)", command=self.open_record)
        game_menu.add_separator()
        game_menu.add_command(label="退出(X)", command=self.root.quit)
        menubar.add_cascade(label="游戏(G)", menu=game_menu)
        
        # 选项(O)
        option_menu = tk.Menu(menubar, tearoff=0)
        option_menu.add_radiobutton(label="木制棋盘", variable=self.theme_var, value="木制棋盘", command=self._set_theme)
        option_menu.add_radiobutton(label="大理石棋盘", variable=self.theme_var, value="大理石棋盘", command=self._set_theme)
        option_menu.add_radiobutton(label="水晶棋盘", variable=self.theme_var, value="水晶棋盘", command=self._set_theme)
        menubar.add_cascade(label="选项(O)", menu=option_menu)
        
        # 帮助(H)
        help_menu = tk.Menu(menubar, tearoff=0)
        help_menu.add_command(label="关于", command=self.show_about)
        menubar.add_cascade(label="帮助(H)", menu=help_menu)
        
        self.root.config(menu=menubar)

    def _setup_ui(self):
        # 棋盘框架
        self.board_frame = tk.Frame(self.root)
        self.board_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # 右侧控制面板
        self.side_frame = tk.Frame(self.root, width=280)
        self.side_frame.pack(side=tk.RIGHT, fill=tk.Y)
        self.side_frame.pack_propagate(False) 
        
        # 画布
        self.canvas = tk.Canvas(self.board_frame, width=self.canvas_size, height=self.canvas_size, highlightthickness=0)
        self.canvas.pack(padx=20, pady=20)
        self.canvas.bind("<Button-1>", self.on_click)
        
        # 右侧元素
        self.title_lbl = tk.Label(self.side_frame, text="幸福五子棋", font=("楷体", 26, "bold"))
        self.title_lbl.pack(pady=20)
        
        # 对战双方设置
        mode_frame = tk.Frame(self.side_frame, bg="#d0d0d0")
        mode_frame.pack(fill=tk.X, padx=10, pady=10)
        
        tk.Label(mode_frame, text="黑方(先手):", font=("Arial", 10), bg="#d0d0d0").grid(row=0, column=0, sticky=tk.W)
        roles = ["玩家", "初级(B)", "中级(I)", "高级(E)", "大师(M)", "深智(MCTS)", "深度学习(AlphaZero)"]
        self.black_combo = ttk.Combobox(mode_frame, textvariable=self.black_player_var, values=roles, state="readonly", width=16)
        self.black_combo.grid(row=0, column=1, padx=5, pady=2)
        self.black_combo.bind("<<ComboboxSelected>>", self._on_role_change)
        
        tk.Label(mode_frame, text="白方(后手):", font=("Arial", 10), bg="#d0d0d0").grid(row=1, column=0, sticky=tk.W)
        self.white_combo = ttk.Combobox(mode_frame, textvariable=self.white_player_var, values=roles, state="readonly", width=16)
        self.white_combo.grid(row=1, column=1, padx=5, pady=2)
        self.white_combo.bind("<<ComboboxSelected>>", self._on_role_change)
        
        self.info_label = tk.Label(self.side_frame, text="当前执棋: 黑方", font=("Arial", 12))
        self.info_label.pack(pady=5)
        
        self.win_rate_lbl = tk.Label(self.side_frame, text="预期胜率: -", font=("Arial", 12))
        self.win_rate_lbl.pack(pady=2)
        
        self.forbidden_var = tk.BooleanVar(value=False)
        tk.Checkbutton(self.side_frame, text="启用禁手规则(仅黑棋)", variable=self.forbidden_var, font=("Arial", 10), command=self._toggle_forbidden).pack(fill=tk.X, padx=5, pady=2)
        
        self.vcf_var = tk.BooleanVar(value=False)
        tk.Checkbutton(self.side_frame, text="启用AI VCF强杀搜索", variable=self.vcf_var, font=("Arial", 10)).pack(fill=tk.X, padx=5, pady=2)
        
        self.progress_var = tk.DoubleVar()
        self.progress_var.set(0)
        
        prog_frame = tk.Frame(self.side_frame)
        prog_frame.pack(fill=tk.X, padx=30, pady=10)
        self.progress = ttk.Progressbar(prog_frame, variable=self.progress_var, maximum=100)
        self.progress.pack(fill=tk.X)
        self.progress_lbl = tk.Label(prog_frame, text="0%")
        self.progress_lbl.pack()
        
        btn_frame = tk.Frame(self.side_frame)
        btn_frame.pack(pady=20)
        
        tk.Button(btn_frame, text="📂 打谱", font=("Arial", 12), command=self.open_record, width=8, relief=tk.FLAT).grid(row=0, column=0, padx=15, pady=15)
        tk.Button(btn_frame, text="▶ 重来", font=("Arial", 12), command=self.restart_game, width=8, relief=tk.FLAT).grid(row=0, column=1, padx=15, pady=15)
        tk.Button(btn_frame, text="💾 保存", font=("Arial", 12), command=self.save_record, width=8, relief=tk.FLAT).grid(row=1, column=0, padx=15, pady=15)
        tk.Button(btn_frame, text="⟲ 悔棋", font=("Arial", 12), command=self.undo, width=8, relief=tk.FLAT).grid(row=1, column=1, padx=15, pady=15)

    def _set_theme(self):
        theme = self.theme_var.get()
        if theme == "木制棋盘":
            bg_color = "#DEB887"
            panel_color = "#D2B48C"
        elif theme == "大理石棋盘":
            bg_color = "#F0F0F0"
            panel_color = "#E0E0E0"
        elif theme == "水晶棋盘":
            bg_color = "#E0FFFF"
            panel_color = "#AFEEEE"
            
        self.root.config(bg=panel_color)
        self.board_frame.config(bg=panel_color)
        self.side_frame.config(bg=panel_color)
        self.title_lbl.config(bg=panel_color)
        self.progress_lbl.config(bg=panel_color)
        self.info_label.config(bg=panel_color)
        if hasattr(self, 'win_rate_lbl'):
            self.win_rate_lbl.config(bg=panel_color)
            
        for child in self.side_frame.winfo_children():
            if isinstance(child, tk.Frame) or isinstance(child, tk.Label):
                child.config(bg=panel_color)
                for gchild in child.winfo_children():
                    if isinstance(gchild, tk.Frame) or isinstance(gchild, tk.Label) or isinstance(gchild, tk.Radiobutton):
                        gchild.config(bg=panel_color)
        
        self.canvas.config(bg=bg_color)
        self._draw_board()

    def _toggle_forbidden(self):
        self.game.forbidden_enabled = self.forbidden_var.get()

    def _on_role_change(self, event=None):
        self._instantiate_ai()
        self.restart_game()

    def restart_game(self):
        if self.mode == "Replay":
            self.mode = "PvE" # Reset mode override
            
        self.game.reset()
        self.progress_var.set(0)
        self.progress_lbl.config(text="0%")
        if hasattr(self, 'win_rate_lbl'):
            self.win_rate_lbl.config(text="预期胜率: -")
        self._draw_board()
        self.update_info()
        self._trigger_ai_if_needed()

    def update_info(self):
        if hasattr(self, 'info_label') and self.info_label.winfo_exists():
            if self.game.winner != 0:
                winner_str = "黑方" if self.game.winner == 1 else "白方"
                self.info_label.config(text=f"游戏结束，{winner_str} 获胜！")
            elif self.game.check_draw():
                self.info_label.config(text="游戏结束，平局！")
            else:
                current_str = "黑方" if self.game.current_player == 1 else "白方"
                self.info_label.config(text=f"当前执棋: {current_str}")

    def _trigger_ai_if_needed(self):
        if self.game.winner != 0 or self.game.check_draw() or self.mode == "Replay":
            return
            
        if self.mode == "PvE":
            if self.game.current_player == self.ai.player:
                self.root.after(100, self.ai_turn)
        elif self.mode == "EvE":
            self.root.after(500, self.ai_turn) # Added slight delay for visual tracking

    def _draw_board(self):
        self.canvas.delete("all")
        self.flash_item = None
        theme = self.theme_var.get()
        line_color = "black" if theme != "水晶棋盘" else "#4682B4"
        
        for i in range(self.game.board_size):
            y = self.board_margin + i * self.cell_size
            self.canvas.create_line(self.board_margin, y, (self.canvas_size - self.board_margin)*15//16 , y, fill=line_color)
            x = self.board_margin + i * self.cell_size
            self.canvas.create_line(x, self.board_margin, x, (self.canvas_size - self.board_margin)*15//16, fill=line_color)
            
        star_points = [(3, 3), (3, 11), (7, 7), (11, 3), (11, 11)]
        for r, c in star_points:
            x = self.board_margin + c * self.cell_size
            y = self.board_margin + r * self.cell_size
            self.canvas.create_oval(x - 3, y - 3, x + 3, y + 3, fill=line_color, outline=line_color)
            
        for r in range(self.game.board_size):
            for c in range(self.game.board_size):
                if self.game.board[r][c] != 0:
                    self._draw_stone_only(r, c, self.game.board[r][c])
                    
        if self.game.history:
            last_r, last_c, _ = self.game.history[-1]
            x = self.board_margin + last_c * self.cell_size
            y = self.board_margin + last_r * self.cell_size
            size = self.cell_size * 0.15
            self.flash_item = self.canvas.create_rectangle(
                x - size, y - size, x + size, y + size, 
                fill="red", outline="red"
            )
            if not self.flash_state:
                self.canvas.itemconfigure(self.flash_item, state="hidden")

    def _draw_stone_only(self, r, c, player):
        x = self.board_margin + c * self.cell_size
        y = self.board_margin + r * self.cell_size
        r_size = self.cell_size * 0.42
        color = "black" if player == 1 else "white"
        self.canvas.create_oval(x - r_size, y - r_size, x + r_size, y + r_size, fill=color, outline="black")

    def _draw_stone(self, r, c, player):
        self._draw_stone_only(r, c, player)
        if getattr(self, 'flash_item', None):
            try:
                self.canvas.delete(self.flash_item)
            except Exception:
                pass
        x = self.board_margin + c * self.cell_size
        y = self.board_margin + r * self.cell_size
        size = self.cell_size * 0.15
        self.flash_item = self.canvas.create_rectangle(
            x - size, y - size, x + size, y + size, 
            fill="red", outline="red"
        )
        if getattr(self, 'flash_state', True) == False:
            self.canvas.itemconfigure(self.flash_item, state="hidden")

    def undo(self):
        if self.mode == "PvP" or self.mode == "EvE":
            if len(self.game.history) < 1:
                return
            self.game.undo_move()
        elif self.mode == "PvE":
            if len(self.game.history) < 2:
                messagebox.showinfo("提示", "当前无法悔棋！")
                return
            self.game.undo_move()
            self.game.undo_move()
        elif self.mode == "Replay":
            messagebox.showinfo("提示", "回放模式不支持手动悔棋。")
            return
            
        self._draw_board()
        self.update_info()

    def on_click(self, event):
        if self.game.winner != 0 or self.game.check_draw() or self.mode == "Replay":
            return
            
        if self.mode == "PvE" and self.game.current_player == self.ai.player:
            return 
            
        if self.mode == "EvE":
            return # Block click entirely
            
        # Calculate grid position
        c = round((event.x - self.board_margin) / self.cell_size)
        r = round((event.y - self.board_margin) / self.cell_size)
        
        self.make_move(r, c)
        
    def make_move(self, r, c):
        if not self.game.is_valid_move(r, c):
            return
            
        player_made_move = self.game.current_player
        result = self.game.make_move(r, c)
        if result is False:
            return
            
        self._draw_stone(r, c, player_made_move)
        self.update_info()
        
        if isinstance(result, str):
            messagebox.showwarning("禁手判负", f"黑方落子触发了【{result}】！\n连珠规则下黑棋落败。")
            self.update_info()
            return
            
        self.check_game_over()
        
        if self.mode != "Replay":
            self._trigger_ai_if_needed()
                
    def ai_turn(self):
        ai_obj = self.ai_black if self.game.current_player == 1 else self.ai_white
        if not ai_obj: return
            
        self.progress_var.set(50)
        if hasattr(self, 'progress_lbl'): self.progress_lbl.config(text="AI 思考中...")
        self.root.update()
        
        move = ai_obj.get_best_move(self.game)
        if move:
            if len(self.game.history) == 1 and self.game.current_player == 2:
                import random
                if random.random() < 0.75:
                    black_r, black_c, _ = self.game.history[0]
                    opt_r, opt_c = move
                    dr, dc = opt_r - black_r, opt_c - black_c
                    rotated = random.choice([(dr, dc), (-dc, dr), (-dr, -dc), (dc, -dr)])
                    new_r, new_c = black_r + rotated[0], black_c + rotated[1]
                    if self.game.is_valid_move(new_r, new_c):
                        move = (new_r, new_c)
            self.make_move(move[0], move[1])
            
        if hasattr(ai_obj, "latest_win_rate"):
            self.win_rate_lbl.config(text=f"预期胜率: {ai_obj.latest_win_rate:.1f}%")
            
        self.progress_var.set(100)
        self.progress_lbl.config(text="完毕")

    def check_game_over(self):
        if self.game.winner != 0:
            winner_str = "黑方" if self.game.winner == 1 else "白方"
            if self.mode != "Replay":
                messagebox.showinfo("游戏结束", f"{winner_str} 获胜！\n若要保留棋局请点击保存记录。")
        elif self.game.check_draw():
            if self.mode != "Replay":
                messagebox.showinfo("游戏结束", "平局！")

    def save_record(self):
        if not self.game.history:
            messagebox.showinfo("提示", "当前没有落子记录可保存。")
            return
            
        header_b = self.black_player_var.get()
        header_w = self.white_player_var.get()
            
        if self.game.winner == 1:
            res = "先手胜"
        elif self.game.winner == 2:
            res = "后手胜"
        else:
            res = "平局"
            
        now_dt = datetime.datetime.now()
        now_str = now_dt.strftime("%Y.%m.%d %H:%M")
        
        # Format explicitly asks for this layout inside brackets 
        info_header = f"{{[C5][{header_b}][{header_w}][{res}][{now_str} 本地][DYZ-2026课程设计]"
        
        moves_str = ""
        for r, c, player in self.game.history:
            col_char = chr(ord('A') + c)
            row_num = 15 - r
            color_char = "B" if player == 1 else "W"
            moves_str += f";{color_char}({col_char},{row_num})"
            
        record = info_header + moves_str + "}"
        safe_header_w = header_w.replace("(", "_").replace(")", "")
        
        time_prefix = now_dt.strftime("%Y%m%d_%H%M%S")
        filename = f"{time_prefix}_C5_{header_b}_vs_{safe_header_w}_{res}.txt"
        
        filepath = filedialog.asksaveasfilename(
            defaultextension=".txt", 
            initialfile=filename,
            filetypes=[("Text Files", "*.txt")])
        if not filepath:
            return
            
        try:
            with open(filepath, "w", encoding="gb2312") as f:
                f.write(record)
            messagebox.showinfo("保存成功", "棋谱已成功保存！")
        except Exception as e:
            messagebox.showerror("保存失败", f"保存棋谱时出现错误: {e}")

    def open_record(self):
        filepath = filedialog.askopenfilename(filetypes=[("Text Files", "*.txt")])
        if filepath:
            try:
                try:
                    with open(filepath, "r", encoding="gb2312") as f:
                        content = f.read()
                except UnicodeDecodeError:
                    with open(filepath, "r", encoding="utf-8") as f:
                        content = f.read()
                        
                self.parse_and_play(content)
            except Exception as e:
                messagebox.showerror("错误", f"读取失败: {e}")

    def parse_and_play(self, content):
        self.game.reset()
        self._draw_board()
        self.update_info()
        
        content = content.replace('\\n', '').replace(' ', '')
        start_idx = content.find('{')
        end_idx = content.rfind('}')
        if start_idx == -1 or end_idx == -1:
            raise ValueError("棋谱格式不正确，缺少大括号。")
            
        content = content[start_idx+1:end_idx]
        parts = content.split(';')
        if len(parts) < 2:
            raise ValueError("棋谱信息不足。")
            
        moves = [p for p in parts if p.startswith('B(') or p.startswith('W(')]
        
        self.replay_moves = []
        for move_str in moves:
            try:
                brace_in = move_str.find('(')
                brace_out = move_str.find(')')
                if brace_in != -1 and brace_out != -1:
                    pos_str = move_str[brace_in+1:brace_out]
                    c_char, r_str = pos_str.split(',')
                    c = ord(c_char) - ord('A')
                    r = 15 - int(r_str)
                    self.replay_moves.append((r, c))
            except Exception:
                continue 
                
        self._replay_mode_active = True
        self.mode = "Replay"
        self.replay_index = 0
        self.progress_lbl.config(text="回放中...")
        self.replay_next()

    def replay_next(self):
        if self.mode != "Replay":
            return
            
        if self.replay_index < len(self.replay_moves):
            r, c = self.replay_moves[self.replay_index]
            self.make_move(r, c)
            self.replay_index += 1
            self.progress_var.set(int((self.replay_index / max(1, len(self.replay_moves))) * 100))
            self.root.after(800, self.replay_next) # 800ms per move frame
        else:
            self.progress_var.set(100)
            self.progress_lbl.config(text="完毕")
            messagebox.showinfo("完成", "棋谱回放已结束！")
            self._replay_mode_active = False
            self._trigger_ai_if_needed()

    def show_about(self):
        messagebox.showinfo("关于", "幸福五子棋\n版本：DYZ-2026课程设计-v1.1\n基于纯Python Tkinter开发")


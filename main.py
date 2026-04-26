import tkinter as tk
from gui import GomokuGUI

if __name__ == "__main__":
    root = tk.Tk()
    app = GomokuGUI(root)
    # 限制窗口最小大小避免形变
    root.minsize(root.winfo_width(), root.winfo_height())
    root.mainloop()

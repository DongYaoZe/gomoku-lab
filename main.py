import tkinter as tk
from gui import GomokuGUI

if __name__ == "__main__":
    root = tk.Tk()
    app = GomokuGUI(root)
    root.update_idletasks()
    root.minsize(root.winfo_width(), root.winfo_height())
    root.mainloop()

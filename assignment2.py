import tkinter as tk
import tkinter.messagebox as messagebox

class Calculator:
    def __init__(self, master):
        self.master = master
        master.title("Enhanced Calculator")

        self.expression = ""
        self.input_text = tk.StringVar()

        self.input_frame = tk.Frame(master, bd=10, bg='lightgrey')
        self.input_frame.pack(side=tk.TOP)

        self.input_field = tk.Entry(self.input_frame, textvariable=self.input_text, font=('arial', 18, 'bold'), bd=10, insertwidth=4, width=14, borderwidth=4, relief=tk.RIDGE, justify='right')
        self.input_field.grid(row=0, column=0)
        self.input_field.pack(ipady=10)  # internal padding for the text field

        self.buttons_frame = tk.Frame(master, bg='grey')
        self.buttons_frame.pack()

        self.create_buttons()

    def create_buttons(self):
        buttons = [
            ('7', 1, 0), ('8', 1, 1), ('9', 1, 2), ('/', 1, 3),
            ('4', 2, 0), ('5', 2, 1), ('6', 2, 2), ('*', 2, 3),
            ('1', 3, 0), ('2', 3, 1), ('3', 3, 2), ('-', 3, 3),
            ('0', 4, 0), ('.', 4, 1), ('=', 4, 2), ('+', 4, 3),
        ]

        for (text, row, col) in buttons:
            self.create_button(text, row, col)

        self.clear_button = tk.Button(self.buttons_frame, text='Clear', width=42, height=3, bd=1, bg='#ff6666', fg='white', command=self.clear, font=('arial', 12, 'bold'))
        self.clear_button.grid(row=5, column=0, columnspan=4)

    def create_button(self, text, row, col):
        button = tk.Button(self.buttons_frame, text=text, width=10, height=3, bd=1, bg='#333333', fg='white', font=('arial', 12, 'bold'), relief=tk.RAISED, command=lambda: self.click_button(text))
        button.grid(row=row, column=col, padx=1, pady=1)

    def click_button(self, item):
        if item == '=':
            self.calculate()
        else:
            self.expression += str(item)
            self.input_text.set(self.expression)

    def calculate(self):
        try:
            result = str(eval(self.expression))
            self.input_text.set(result)
            self.expression = result
        except:
            messagebox.showinfo("Error", "Syntax Error")
            self.expression = ""

    def clear(self):
        self.expression = ""
        self.input_text.set("")

if __name__ == "__main__":
    root = tk.Tk()
    root.configure(bg='grey')
    calc = Calculator(root)
    root.mainloop()

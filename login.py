import tkinter as tk
from tkinter import messagebox

def perform_login():
    """
    Function to validate the username and password.
    """
    username = entry_username.get()
    password = entry_password.get()

    # --- HARDCODED CREDENTIALS FOR DEMO ---
    # In a real app, you would check these against a database
    valid_username = "admin"
    valid_password = "password123"

    if username == valid_username and password == valid_password:
        messagebox.showinfo("Status", "Congragulations your account has been succesfully hacked.")
        # Add code here to open the next window
    else:
        messagebox.showerror("Error", "You have violated the zone rules now face the consequences.")
def exit_app(event=None):
    root.destroy()
# --- MAIN WINDOW SETUP ---
root = tk.Tk()
root.title("☠️☠️☠️☠️☠️")
root.attributes('-fullscreen', True)
root.bind("<Escape>", exit_app)
root.configure(bg="black")

# Center the window content
# We use a 'Frame' to hold the widgets neatly
frame = tk.Frame(root, bg="black")
frame.pack(expand=True)

# --- WIDGETS ---

# 1. Heading
label_header = tk.Label(frame, text="RED ZONE", font=("Chiller", 20, "bold"), bg="black", fg="red")
label_header.pack(pady=20)

# 2. Username Field
label_user = tk.Label(frame, text="Username:", font=("Chiller", 20, "bold"), bg="black", fg="red")
label_user.pack()
entry_username = tk.Entry(frame)
entry_username.pack(pady=5)

# 3. Password Field
label_pass = tk.Label(frame, text="Password:", font=("Chiller", 20, "bold"), bg="black", fg="red")
label_pass.pack()
entry_password = tk.Entry(frame, show="*") # 'show' hides the text
entry_password.pack(pady=5)

# 4. Login Button
button_login = tk.Button(frame, text="Enter The zone", font=("Chiller", 22, "bold"), command=perform_login, bg="black", fg="red", width=15)
button_login.pack(pady=20)

# --- START THE APP ---
root.mainloop()
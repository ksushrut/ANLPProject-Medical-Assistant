import openpyxl
from tkinter import *
from tkinter import messagebox


def register():
    # Get the user input from the form
    first_name = first_name_entry.get()
    messagebox.showinfo("Success", "Registration successful!")
    return first_name


# Create the main tkinter window
root = Tk()
root.title("Registration Form")
root.geometry('300x300')

# Create labels and entry fields for each input
first_name_label = Label(root, text="First Name:")
first_name_label.pack()
first_name_entry = Entry(root)
first_name_entry.pack()

register_button = Button(root, text="Register", command=register)
register_button.pack()

root.mainloop()
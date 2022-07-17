#!/usr/bin/python3


import io
import os
import sys
import random
import struct
import base64
import tempfile
import hstegolib
import threading

from tkinter import *
from tkinter.ttk import *
from tkinter import filedialog
from tkinter import messagebox
from tkinter import scrolledtext

from PIL import Image, ImageTk
import numpy as np

WIN_W = 600
WIN_H = 450
PANE_W = 600
PANE_H = 380
FONT = ("Helvetica", "11", "normal")
FONT_B = ("Helvetica", "11", "bold")


class CustomText(scrolledtext.ScrolledText):
    def __init__(self, *args, **kwargs):
        # {{{
        """A text widget that report on internal widget commands"""
        scrolledtext.ScrolledText.__init__(self, *args, **kwargs)

        # create a proxy for the underlying widget
        self._orig = self._w + "_orig"
        self.tk.call("rename", self._w, self._orig)
        self.tk.createcommand(self._w, self._proxy)
        # }}}

    def _proxy(self, command, *args):
        # {{{
        cmd = (self._orig, command) + args
        result = self.tk.call(cmd)

        if command in ("insert", "delete", "replace"):
            self.event_generate("<<TextModified>>")

        return result
        # }}}




class Wizard:
    def __init__(self, window):
        # {{{
        self.window = window
        self.hide_btn = None
        self.extract_btn = None

        self.progressbar = None
        self.cover_entry = None
        self.dst_stego_image_path = None
        self.msg_text = None
        self.use_msg_file = None
        self.msg_entry = None
        self.stego_entry = None
        self.passw_hide_entry = None
        self.passw_extract_entry = None
        self.capacity_entry = None
        self.msg_size_entry = None
        self.dest_msg = None

        step = {
            "1H": PanedWindow(window),
            "2H": PanedWindow(window),
            "3H": PanedWindow(window),
            "4H": PanedWindow(window),
            "5H": PanedWindow(window),
            "1E": PanedWindow(window),
            "2E": PanedWindow(window),
            "3E": PanedWindow(window),
            "4E": PanedWindow(window),
            "5E": PanedWindow(window),
        }
        step["1E"] = step["1H"] # First step is the same for H(ide) and E(xtract)
        self.step = step
        self.key = "1H"


        nav_panel = PanedWindow(window)
        nav_panel.place(x=0, y=PANE_H, width=PANE_W, height=WIN_H-PANE_H)

        def about_fn():
            print("about")

        self.about_btn = navbtn(nav_panel, "About", about_fn,  100, 20)
        self.restart_btn = navbtn(nav_panel, "Restart", self.restart,  205, 20)
        self.prev_btn = navbtn(nav_panel, "Previous", self.prev,  320, 20, disabled=True)
        self.next_btn = navbtn(nav_panel, "Next", self.next,  425, 20, disabled=True)
        # }}}

    def panel(self, k):
        # {{{
        return self.step[k]
        # }}}

    def restart(self):
        """ Restart the interface """
        # {{{
        self.step[self.key].place_forget()
        self.key = "1H"
        self.step[self.key].place(x=0, y=0, width=PANE_W, height=PANE_H)
        self.prev_btn["state"] = "disable" 
        self.next_btn["state"] = "disable"
        if self.hide_btn:
            self.hide_btn.config(style='TButton')
        if self.extract_btn:
           self.extract_btn.config(style='TButton')
        # }}}

    def has_errors(self):
        """ Check inputs and display errors """
        # {{{
        if self.key == "2H":
            cover_image_path = self.cover_entry.get()
            if not os.path.exists(cover_image_path):
                messagebox.showerror('Error', 'Please, select a valid cover image')
                return True

        if self.key == "2E":
            stego_image_path = self.stego_entry.get()
            if not os.path.exists(stego_image_path):
                messagebox.showerror('Error', 'Please, select a valid stego image')
                return True

        if self.key == "3H":
            msg_path = self.msg_entry.get()
            input_msg_content = self.msg_text.get("1.0", END)
            if not self.use_msg_file.get():
                if len(input_msg_content.strip())==0:
                    messagebox.showerror('Error', 'Please, write a longer message')
                    return True
            elif not os.path.exists(msg_path):
                messagebox.showerror('Error', 'Please, select a valid message file')
                return True

        if self.key == "3E":
            password = self.passw_extract_entry.get()
            if len(password) == 0:
                messagebox.showerror('Error', 'Please, enter a password')
                return True

        if self.key == "4H":
            password = self.passw_hide_entry.get()
            if len(password) < 8:
                messagebox.showerror('Error', 'Password too short')
                return True


        return False
        # }}}

    def next(self):
        """ Go to next screen/panel """
        # {{{
        if self.has_errors():
            return

        self.step[self.key].place_forget()
        n = int(self.key[0])+1
        self.key = str(n)+self.key[1]
        self.step[self.key].place(x=0, y=0, width=PANE_W, height=PANE_H)
        if self.key[0] == "4":
            self.next_btn["state"] = "disable"
        self.prev_btn["state"] = "enable" 
        # }}}

    def prev(self):
        """ Go to previous screen/panel """
        # {{{
        self.step[self.key].place_forget()
        n = int(self.key[0])-1
        self.key = str(n)+self.key[1]
        self.step[self.key].place(x=0, y=0, width=PANE_W, height=PANE_H)
        if self.key[0] == "1":
            self.prev_btn["state"] = "disable" 
        self.next_btn["state"] = "enable" 
        # }}}

    def get_cover_capacity(self):
        # {{{
        image_capacity = 0
        img_path = self.cover_entry.get()

        if hstegolib.is_ext(img_path, hstegolib.SPATIAL_EXT):
            I = np.asarray(Image.open(img_path))
            image_capacity = hstegolib.spatial_capacity(I)
        else:
            jpg = hstegolib.jpeg_load(img_path)
            image_capacity = hstegolib.jpg_capacity(jpg)
        return image_capacity
        # }}}

    def get_msg_size(self):
        # {{{
        msg_size = 0
        content = ""

        if not self.use_msg_file.get():
            content = self.msg_text.get("1.0", END).strip()
        else:
            msg_path = self.msg_entry.get()
            with open('data.txt', 'r') as f:
                content = f.read()

        return len(content.encode())
        # }}}

    def hide(self):
        """ Hide message using the info in the inputs """
        # {{{
        cover_image_path = self.cover_entry.get()
        stego_image_path = self.dst_stego_image_path

        if cover_image_path == stego_image_path:
            messagebox.showerror('Error', 'Cover and stego cannot be the same image')
            return

        msg_path = self.msg_entry.get()
        password = self.passw_hide_entry.get()
        tmp = None
        if not self.use_msg_file.get():
            input_msg_content = self.msg_text.get("1.0", END).strip()
            tmp = tempfile.NamedTemporaryFile(delete=False)
            tmp.write(input_msg_content.encode())
            msg_path = tmp.name

        if hstegolib.is_ext(self.cover_entry.get(), hstegolib.SPATIAL_EXT):
            hill = hstegolib.HILL()
            hill.embed(cover_image_path, msg_path, password, stego_image_path)
        else:
            juniw = hstegolib.J_UNIWARD()
            juniw.embed(cover_image_path, msg_path, password, stego_image_path)

        if tmp:
            tmp.close()
            os.unlink(tmp.name)

        # }}}




class StepScreen:
    def __init__(self, wz):
        # {{{
        self.wz = wz
        # }}}

    def create_step_1H_screen(self):
        # {{{ 
        wz = self.wz
        text_frame(
            wz.panel("1H"), 
            ' Step 1 ',
            'Welcome to HStego!',
            'HStego allows you to hide information inside an image.',
            10, 10, 580, 80
        )

        def hide_btn_click():
            global S
            wz.hide_btn.config(style='Custom.TButton')
            wz.extract_btn.config(style='TButton')
            wz.next_btn["state"] = "enable"
            wz.key = wz.key[0]+'H'

        def extract_btn_click():
            global S
            wz.extract_btn.config(style='Custom.TButton')
            wz.hide_btn.config(style='TButton')
            wz.next_btn["state"] = "enable"
            wz.key = wz.key[0]+'E'

        wz.hide_btn = Button(wz.panel("1H"), command=hide_btn_click)
        img = Image.open(os.path.join("resources", "hide.png"))
        hide_img = ImageTk.PhotoImage(img)
        # keep a reference to ensure it's not garbage collected
        wz.hide_btn.image = hide_img 
        wz.hide_btn.config(image=hide_img)
        wz.hide_btn.place(x=100, y=130, width="180", height='180')



        Label(wz.panel("1H"), text='Hide information').place(x=130, y=315)

        wz.extract_btn = Button(wz.panel("1H"), command=extract_btn_click)
        extract_img = ImageTk.PhotoImage(Image.open(os.path.join("resources", "extract.png")))
        wz.extract_btn.image = extract_img 
        wz.extract_btn.config(image=extract_img)
        wz.extract_btn.place(x=320, y=130, width="180", height='180')
        Label(wz.panel("1H"), text='Extract information').place(x=345, y=315)
        # }}}

    def create_step_2H_screen(self):
        # {{{ 
        wz = self.wz

        text_frame(
            wz.panel("2H"), 
            ' Step 2 ',
            'Select a cover image!',
            'Use your own photography and delete the original source after hidding data.',
            10, 10, 580, 80
        )

        if sys.platform == "win32":
            canvas = Canvas(wz.panel("2H"), width=WIN_W, height=250, 
                            bg="#D9D9D9", bd=0, relief='ridge', highlightthickness=0)
        else:
            canvas = Canvas(wz.panel("2H"), width=WIN_W, height=250)

        canvas.place(x=0, y=150)

        def loadcover():
            filename = filedialog.askopenfilename(
                            initialdir = ".",
                            title = "Select a File",
                            filetypes = (
                                ("PNG files", "*.png"),
                                ("JPEG files", "*.jpg"),
                            )
            )
              
            wz.cover_entry.delete(0, END)
            wz.cover_entry.insert(0, filename)

            img = Image.open(filename)
            w, h = img.size
            new_h = 250
            new_w = int(new_h * w / h)
            img = img.resize((new_w, new_h))
            img = ImageTk.PhotoImage(img)
            x_offset = int(WIN_W/2-new_w/2)
            canvas.create_image(x_offset, 0, anchor=NW, image=img)
            canvas.img = img

            # Update image capacity entry
            capacity = wz.get_cover_capacity()
            wz.capacity_entry["state"] = "normal"
            wz.capacity_entry.delete(0, END)
            wz.capacity_entry.insert(0, str(capacity)+' ')
            wz.capacity_entry["state"] = "disabled"


        btn = Button(wz.panel("2H"), command=loadcover, text="Select image")
        btn.place(x=10, y=100, width=150, height=30)

        wz.cover_entry = Entry(wz.panel("2H"))
        wz.cover_entry.place(x=170, y=100, width=420, height=30)

        # }}}

    def create_step_3H_screen(self):
        # {{{
        wz = self.wz

        text_frame(
            wz.panel("3H"), 
            ' Step 3 ',
            'Write a message!',
            'You can write a message or select a file.',
            10, 10, 580, 80
        )

        label = Label(wz.panel("3H"), text='Message:', font=FONT)
        label.place(x=10, y=110)
        #wz.msg_text = scrolledtext.ScrolledText(wz.panel("3H"), height=9, width=70)
        wz.msg_text = CustomText(wz.panel("3H"), height=9, width=70)
        wz.msg_text.place(x=10, y=130)

        def on_change(event):
            sz = wz.get_msg_size()
            wz.msg_size_entry["state"] = "normal"
            wz.msg_size_entry.delete(0, END)
            wz.msg_size_entry.insert(0, str(sz)+' ')
            wz.msg_size_entry["state"] = "disabled"

        wz.msg_text.bind("<<TextModified>>", on_change)


        def loadmsg():
            filename = filedialog.askopenfilename(
                            initialdir = ".",
                            title = "Select a File",
                            filetypes = (
                                ("All files", "*.*"),
                            )
            )
            wz.msg_entry.delete(0, END)
            wz.msg_entry.insert(0, filename)

            # Update msg size entry
            sz = wz.get_msg_size()
            wz.msg_size_entry["state"] = "normal"
            wz.msg_size_entry.delete(0, END)
            wz.msg_size_entry.insert(0, str(sz)+' ')
            wz.msg_size_entry["state"] = "disabled"



        msg_btn = Button(wz.panel("3H"), command=loadmsg, text="Select message file")
        msg_btn.place(x=10, y=340, width=150, height=30)
        wz.msg_entry = Entry(wz.panel("3H"))
        wz.msg_entry.place(x=170, y=340, width=420, height=30)
        msg_btn["state"] = "disabled"
        wz.msg_entry["state"] = "disabled"


        def checkbox_fn():
            if wz.use_msg_file.get():
                wz.msg_text["state"] = "disabled"
                wz.msg_text.configure(bg='#D9D9D9')
                msg_btn["state"] = "normal"
                wz.msg_entry["state"] = "normal"
            else:
                wz.msg_text["state"] = "normal"
                wz.msg_text.configure(bg='#FFFFFF')
                msg_btn["state"] = "disabled"
                wz.msg_entry["state"] = "disabled"
        wz.use_msg_file = BooleanVar(wz.panel("3H"))
        checkbox = Checkbutton(wz.panel("3H"), text="I prefer to hide a file",
                               variable=wz.use_msg_file, command=checkbox_fn)
        checkbox.place(x=10, y=310)



        # }}}

    def create_step_4H_screen(self):
        #  {{{ 
        wz = self.wz

        text_frame(
            wz.panel("4H"), 
            ' Step 4 ',
            'Hide data!',
            'Set a password and choose the destination stego image.',
            10, 10, 580, 80
        )

        label = Label(wz.panel("4H"), text='Password:', font=FONT)
        label.place(x=10, y=120)
        wz.passw_hide_entry = Entry(wz.panel("4H"), show="*", justify='center')
        wz.passw_hide_entry.place(x=100, y=110, width=270, height=30)


        # Capacity info
        label = Label(wz.panel("4H"), 
                      text='Capacity of the selected image:', font=FONT)
        label.place(x=10, y=180)
        wz.capacity_entry = Entry(wz.panel("4H"), style='Custom.TEntry', justify='right')
        wz.capacity_entry.place(x=230, y=170, width=140, height=30)
        wz.capacity_entry["state"] = 'disabled';
        label = Label(wz.panel("4H"), text='bytes', font=FONT)
        label.place(x=380, y=180)

        label = Label(wz.panel("4H"), 
                      text='Size of the message to send:', font=FONT)
        label.place(x=10, y=220)
        wz.msg_size_entry = Entry(wz.panel("4H"), style='Custom.TEntry', justify='right')
        wz.msg_size_entry.place(x=230, y=210, width=140, height=30)
        wz.msg_size_entry["state"] = 'disabled';
        label = Label(wz.panel("4H"), text='bytes', font=FONT)
        label.place(x=380, y=220)




        wz.progressbar = Progressbar(wz.panel("4H"), orient="horizontal", 
                                                         mode="indeterminate")
        wz.progressbar.start()


        def threaded_hide(wz):
            wz.hide()
            wz.progressbar.place_forget()
            messagebox.showinfo('Success', 'The message has been hidden')
            wz.restart()

        def savestego():
            if wz.has_errors():
                return

            ext = 'jpg'
            if hstegolib.is_ext(wz.cover_entry.get(), hstegolib.SPATIAL_EXT):
                ext = 'png'

            f = filedialog.asksaveasfilename(
                    initialfile = 'stego.'+ext,
                    defaultextension="."+ext
            )
            wz.dst_stego_image_path = f
            
            wz.progressbar.place(x=10, y=300, width=580, height=30)
            t = threading.Thread(target=threaded_hide, args=[wz])
            t.start()

        btn = Button(wz.panel("4H"), command=savestego, text="Save")
        btn.place(x=400, y=110, width=180, height=30)



        # }}}

    def create_step_2E_screen(self):
        # {{{ 
        wz = self.wz

        text_frame(
            wz.panel("2E"), 
            ' Step 2 ',
            'Select a stego image!',
            'The stego image needs to contain a message hidden with HStego.',
            10, 10, 580, 80
        )

        if sys.platform == "win32":
            canvasE = Canvas(wz.panel("2E"), width=WIN_W, height=250, 
                            bg="#D9D9D9", bd=0, relief='ridge', highlightthickness=0)
        else:
            canvasE = Canvas(wz.panel("2E"), width=WIN_W, height=250)

        canvasE.place(x=0, y=150)

        def loadstego():
              filename = filedialog.askopenfilename(
                            initialdir = ".",
                            title = "Select a File",
                            filetypes = (
                                ("PNG files", "*.png"),
                                ("JPEG files", "*.jpg"),
                            )
              )
              
              wz.stego_entry.delete(0, END)
              wz.stego_entry.insert(0, filename)

              img = Image.open(filename)
              w, h = img.size
              new_h = 250
              new_w = int(new_h * w / h)
              img = img.resize((new_w, new_h))
              img = ImageTk.PhotoImage(img)
              x_offset = int(WIN_W/2-new_w/2)
              canvasE.create_image(x_offset, 0, anchor=NW, image=img)
              canvasE.img = img


        btn = Button(wz.panel("2E"), command=loadstego, text="Select image")
        btn.place(x=10, y=100, width=150, height=30)

        wz.stego_entry = Entry(wz.panel("2E"))
        wz.stego_entry.place(x=170, y=100, width=420, height=30)
        # }}}

    def create_step_3E_screen(self):
        # {{{
        wz = self.wz

        text_frame(
            wz.panel("3E"), 
            ' Step 3 ',
            'Enter the password!',
            'It is used to decrypt and extract the message.',
            10, 10, 580, 80
        )


        label = Label(wz.panel("3E"), text='Password:', font=FONT)
        label.place(x=10, y=120)
        wz.passw_extract_entry = Entry(wz.panel("3E"), show="*", width=5)
        wz.passw_extract_entry.place(x=100, y=110, width=250, height=30)


        wz.dest_msg = StringVar()
        Radiobutton(
            wz.panel("3E"), 
            variable=wz.dest_msg, 
            value="SCREEN", 
            width=40, 
            text="Show the extracted message"
        ).place(x=10, y=275)
        Radiobutton(
            wz.panel("3E"), 
            variable=wz.dest_msg, 
            value="FILE", 
            width=40, 
            text="Save the extracted message into a file"
        ).place(x=10, y=300)
        wz.dest_msg.set("SCREEN")




        # }}}

    def create_step_4E_screen(self):
        # {{{
        wz = self.wz

        text_frame(
            wz.panel("4E"), 
            ' Step 4 ',
            'Extract the message!',
            'You can extract the message here or into a file.',
            10, 10, 580, 80
        )
        # }}}





# {{{ init()
def init():
    window = Tk()
    window.title("HStego")
    window.geometry(f'{WIN_W}x{WIN_H}')
    window.resizable(False, False)

    window.style = Style()
    #print(window.style.theme_names())
    window.style.theme_use("default")
    #if sys.platform == "win32":
    #    window.style.theme_use("vista") # vista|winnative|xpnative
    
    # Custom styles
    window.option_add('*Dialog.msg.font', 'Helvetica 12')

    window.style.configure(
        'Custom.TButton', 
        background='#77DD77', 
    )
    window.style.map(
        'Custom.TButton', 
        background=[
            ('active', '#88EE88'),
        ]
    )

    window.style.map(
        "Custom.TEntry",
        fieldbackground=[
            ("active", "black"), 
            ("disabled", "white")
        ]
    )


    return window
# }}}

# {{{ text_frame()
def text_frame(window, title, text1, text2, x, y, w, h):
    Labelframe(window, text=title).place(x=x, y=y, width=w, height=h)
    l1 = Label(window, text=text1, font=FONT_B)
    l1.place(x=x+10, y=y+25)
    l2= Label(window, text=text2, font=FONT)
    l2.place(x=x+10, y=y+45)
# }}}

# {{{ navbtn()
def navbtn(window, text, cmd, x, y, disabled=False):
    btn = Button(window, command=cmd, text=text)
    btn.place(x=x, y=y, width=80, height=30)
    if disabled:
        btn["state"] = "disabled"
    return btn
# }}}


def run(window):

    wz = Wizard(window)
    wz.restart()

    ss = StepScreen(wz)

    # Hiding screens
    ss.create_step_1H_screen()
    ss.create_step_2H_screen()
    ss.create_step_3H_screen()
    ss.create_step_4H_screen()

    # Extraction screens
    ss.create_step_2E_screen()
    ss.create_step_3E_screen()
    ss.create_step_4E_screen()

    window.mainloop()




if __name__ == "__main__":
    gui()





#!/usr/bin/python3


import io
import os
import sys
import random
import struct
import base64
import tempfile
import traceback
import hstegolib
import threading
import webbrowser

import tkinter as tk
from tkinter import *
from tkinter.ttk import *
from tkinter import filedialog
from tkinter import messagebox
from tkinter import scrolledtext
from tkinter import simpledialog

from PIL import Image, ImageTk
import numpy as np

WIN_W = 600
WIN_H = 450
PANE_W = 600
PANE_H = 380
FONT = ("Helvetica", "11", "normal")
FONT_B = ("Helvetica", "11", "bold")


base = os.path.dirname(__file__)
if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
    base = sys._MEIPASS



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



class About(simpledialog.Dialog):
    def __init__(self, parent, title):
        # {{{
        super().__init__(parent, title)
        # }}}

    def body(self, frame):
        # {{{
        super().configure(background='#D9D9D9')
        frame.configure(width=350, height=250, background='#D9D9D9')

        Label(frame, text='Developed by', font=FONT).place(x=12, y=20)
        Label(frame, text='Daniel Lerch', font=FONT_B).place(x=107, y=20)
        Label(frame, text='<dlerch@gmail.com>', font=FONT).place(x=200, y=20)

        def click_www(event):
            webbrowser.open_new(event.widget.cget("text"))
        lnk = tk.Label(frame, text='https://daniellerch.me', font=FONT,
                        foreground="blue", background="#D9D9D9", cursor="hand2")
        lnk.bind("<Button-1>", click_www)
        lnk.place(x=100, y=46)

        Label(frame, text='Version: 0.4 (alpha)', font=FONT).place(x=110, y=80)


        Label(frame, text='GitHub:', font=FONT).place(x=150, y=140)
        def click_github(event):
            webbrowser.open_new(event.widget.cget("text"))
        lnk = tk.Label(frame, text='https://github.com/daniellerch/hstego', 
                       font=FONT, foreground="blue", background="#D9D9D9", 
                       cursor="hand2")
        lnk.bind("<Button-1>", click_github)
        lnk.place(x=50, y=160)

        return frame
        # }}}

    def ok_pressed(self):
        # {{{
        self.destroy()
        # }}}

    def cancel_pressed(self):
        # {{{
        self.destroy()
        # }}}

    def buttonbox(self):
        # {{{
        self.ok_button = Button(self, text='OK', width=5, command=self.ok_pressed)
        self.ok_button.pack(side="left")
        self.ok_button.place(x=240, y=210, width=100, height=30)

        self.bind("<Return>", lambda event: self.ok_pressed())
        self.bind("<Escape>", lambda event: self.cancel_pressed())
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
        self.output_msg_path = None
        self.msg_output_text = None       


        step = {
            "1H": PanedWindow(window),
            "2H": PanedWindow(window),
            "3H": PanedWindow(window),
            "4H": PanedWindow(window),
            "1E": PanedWindow(window),
            "2E": PanedWindow(window),
            "3E": PanedWindow(window),
        }
        step["1E"] = step["1H"] # First step is the same for H(ide) and E(xtract)
        self.step = step
        self.key = "1H"


        nav_panel = PanedWindow(window)
        nav_panel.place(x=0, y=PANE_H, width=PANE_W, height=WIN_H-PANE_H)

        def about_fn():
            about = About(window, "About")

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

        if self.msg_text:
            state = self.msg_text["state"] 
            self.msg_text["state"] = 'normal'
            self.msg_text.delete("1.0", END)
            self.msg_text["state"] = state

        if self.msg_output_text:
            state = self.msg_output_text["state"] 
            self.msg_output_text["state"] = 'normal'
            self.msg_output_text.delete("1.0", END)
            self.msg_output_text["state"] = state

        if self.passw_hide_entry:
            self.passw_hide_entry.delete(0, END)

        if self.passw_extract_entry:
            self.passw_extract_entry.delete(0, END)

        if self.cover_entry:
            self.cover_entry .delete(0, END)

        if self.stego_entry:
            self.stego_entry .delete(0, END)

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
                if len(input_msg_content.strip())<=8:
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
        if self.key == "4H" or self.key == "3E":
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
            with open(msg_path, 'rb') as f:
                content = f.read()
    
        try:
            return len(content.encode())
        except:
            pass
        return len(content)
        # }}}

    def hide(self):
        """ Hide message using the info in the inputs """
        # {{{
        try:
            cover_image_path = self.cover_entry.get()
            stego_image_path = self.dst_stego_image_path

            if cover_image_path == stego_image_path:
                messagebox.showerror('Error', 'Cover and stego cannot be the same image')
                return False

            if self.get_msg_size() > self.get_cover_capacity():
                messagebox.showerror('Error', 'The message is too long for the selected cover')
                return False


            msg_path = self.msg_entry.get()
            password = self.passw_hide_entry.get()
            tmp = None
            if not self.use_msg_file.get():
                input_msg_content = self.msg_text.get("1.0", END).strip()
                tmp = tempfile.NamedTemporaryFile(delete=False)
                tmp.write(input_msg_content.encode())
                msg_path = tmp.name
                tmp.close()

            if os.path.getsize(msg_path) <= 8:
                messagebox.showerror('Error', 'Please, write a longer message')
                return False

            if hstegolib.is_ext(cover_image_path, hstegolib.SPATIAL_EXT):
                suniw = hstegolib.S_UNIWARD()
                suniw.embed(cover_image_path, msg_path, password, stego_image_path)
            else:
                juniw = hstegolib.J_UNIWARD()
                juniw.embed(cover_image_path, msg_path, password, stego_image_path)

            if tmp:
                os.unlink(tmp.name)

        except Exception as e:
            messagebox.showerror('Error', str(e))
            traceback.print_exc()
            return False

        return True
        # }}}

    def extract(self):
        """ Extract the message using the info in the inputs """
        # {{{

        try:
            stego_image_path = self.stego_image_path
            password = self.passw_extract_entry.get()
            tmp = None

            if self.dest_msg.get() == "FILE":
                output_msg_path = self.output_msg_path
            else:
                input_msg_content = self.msg_text.get("1.0", END).strip()
                tmp = tempfile.NamedTemporaryFile(delete=False)
                tmp.write(input_msg_content.encode())
                output_msg_path = tmp.name


            if hstegolib.is_ext(stego_image_path, hstegolib.SPATIAL_EXT):
                suniw = hstegolib.S_UNIWARD()
                suniw.extract(stego_image_path, password, output_msg_path)
            else:
                juniw = hstegolib.J_UNIWARD()
                juniw.extract(stego_image_path, password, output_msg_path)


            with open(output_msg_path, "rb") as f:
                content = f.read()
                if len(content) <= 0:
                    messagebox.showerror('Error', 'Message not found, may be the password is wrong')
                    return False

                if self.dest_msg.get() == "SCREEN":
                    self.msg_output_text["state"] = 'normal'
                    self.msg_output_text.delete("1.0", END)
                    self.msg_output_text.insert("1.0", content)
                    self.msg_output_text["state"] = 'disabled'

            if tmp:
                tmp.close()
                os.unlink(tmp.name)
        
        except Exception as e:
            messagebox.showerror('Error', str(e))
            traceback.print_exc()
            return False

        return True
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
            'To hide data within an image, select "hide" or "extract" and press "next"',
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
        img = Image.open(os.path.join(base, "resources", "hide.png"))
        hide_img = ImageTk.PhotoImage(img)
        # keep a reference to ensure it's not garbage collected
        wz.hide_btn.image = hide_img 
        wz.hide_btn.config(image=hide_img)
        wz.hide_btn.place(x=100, y=130, width="180", height='180')



        Label(wz.panel("1H"), text='Hide information').place(x=130, y=315)

        wz.extract_btn = Button(wz.panel("1H"), command=extract_btn_click)
        extract_img = ImageTk.PhotoImage(
                Image.open(os.path.join(base, "resources", "extract.png")))
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
            'Select the image in which you want to hide the secret message.',
            10, 10, 580, 80
        )

        if sys.platform == "win32" or sys.platform == "darwin":
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
                                ("JPEG files", "*.jpeg"),
                            )
            )
            if not filename:
                return

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
            'You can write a secret message or select a secret file.',
            10, 10, 580, 80
        )

        label = Label(wz.panel("3H"), text='Message:', font=FONT)
        label.place(x=10, y=110)
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
            r = wz.hide()
            wz.progressbar.place_forget()
            if r:
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
            if not f:
                return

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
            'The stego image must contain a message hidden with HStego.',
            10, 10, 580, 80
        )

        if sys.platform == "win32" or sys.platform == "darwin":
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
                                ("JPEG files", "*.jpeg"),
                            )
              )
              if not filename:
                  return

              wz.stego_image_path = filename

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
            'You can extract the message here or into a file.',
            10, 10, 580, 80
        )


        label = Label(wz.panel("3E"), text='Message:', font=FONT)
        label.place(x=10, y=110)
        wz.msg_output_text = scrolledtext.ScrolledText(wz.panel("3E"), 
                                                       height=9, width=70)
        wz.msg_output_text.place(x=10, y=130)
        wz.msg_output_text.configure(bg='#FFFFFF')
        wz.msg_output_text["state"] = 'disabled'

        def on_radio_change():
            if wz.dest_msg.get() == "FILE":
                wz.msg_output_text.configure(bg='#D9D9D9')
            else:
                wz.msg_output_text.configure(bg='#FFFFFF')

        wz.dest_msg = StringVar()
        Radiobutton(
            wz.panel("3E"), 
            variable=wz.dest_msg, 
            value="SCREEN", 
            width=25, 
            command=on_radio_change,
            text="Show the message"
        ).place(x=10, y=310)
        Radiobutton(
            wz.panel("3E"), 
            variable=wz.dest_msg, 
            value="FILE", 
            width=25, 
            command=on_radio_change,
            text="Save the message to a file"
        ).place(x=10, y=330)
        wz.dest_msg.set("SCREEN")


        label = Label(wz.panel("3E"), text='Password:', font=FONT)
        label.place(x=280, y=320)
        wz.passw_extract_entry = Entry(wz.panel("3E"), show="*")
        wz.passw_extract_entry.place(x=360, y=310, width=230, height=30)

        def threaded_extract(wz):
            r = wz.extract()
            wz.progressbar.place_forget()
            if r:
                messagebox.showinfo('Success', 'The message has been extracted')

        def extract_msg():
            if wz.has_errors():
                return

            if wz.dest_msg.get() == "FILE":
                f = filedialog.asksaveasfilename(
                        initialfile = 'secret.txt',
                        defaultextension=".txt"
                )
                if not f:
                    return
                wz.output_msg_path = f

            t = threading.Thread(target=threaded_extract, args=[wz])
            t.start()

        msg_btn = Button(wz.panel("3E"), command=extract_msg, 
                         text="Extract the message")
        msg_btn.place(x=360, y=350, width=230, height=30)



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

    window.mainloop()




import sys
import os
import subprocess
from tkinter import *

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def run_script(script_name):
    subprocess.Popen([sys.executable, os.path.join(SCRIPT_DIR, script_name)])


window=Tk()

window.title("Eye Gaze Estimation")
window.geometry('800x500')
window.configure(background="yellow")

l1=Label(window,text="Joint Eye Tracking and Head pose Estimation",font="Times 15 bold italic"
         ,bg="blue",fg="white",borderwidth=10)
l1.grid(column=0, row=0,padx=150,pady=10)



def run1():
    run_script('Eye_Direction.py')
def run2():
    run_script('Virtual_KeyBoard.py')
def run3():
    run_script('Head_pose_and_Full_body.py')
def run4():
    run_script('Virtual_Mouse.py')
def run5():
    run_script('dinoGameWBlinkDetector.py')
    
    
btn1 = Button(window, text="Eye Direction",height=1,width=25,bg="black",
              fg="white",font="Times 12 bold",
              borderwidth=10,command=run1)
btn2 = Button(window, text="Virtual KeyBoard",height=1,width=25,bg="black",
              fg="white",font="Times 12 bold",
              borderwidth=10,command=run2)
btn3 = Button(window, text="Head Pose",height=1,width=25,bg="black",
              fg="white",font="Times 12 bold",
              borderwidth=10,command=run3)
btn4 = Button(window, text="Virtual Mouse", height=1,width=25,bg="black",
              fg="white",font="Times 12 bold",
              borderwidth=10,command=run4)
btn5 = Button(window, text="Dino Game", height=1,width=25,bg="black",
              fg="white",font="Times 12 bold",
              borderwidth=10,command=run5)
btn6=Button(window,text="Exit",fg="white",bg="red",font="time 10 bold",width=10,
            command=window.quit)


btn1.grid(column=0, row=5,padx=200,pady=10)
btn2.grid(column=0, row=6,padx=200,pady=10)
btn3.grid(column=0, row=7,padx=200,pady=10)
btn4.grid(column=0, row=8,padx=200,pady=10)
btn5.grid(column=0, row=9,padx=200,pady=10)
btn6.grid(column=0,row=12,padx=200,pady=10)

window.mainloop()
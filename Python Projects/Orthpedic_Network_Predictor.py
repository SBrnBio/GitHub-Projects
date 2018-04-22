'''
SethBrnBio
Deep Neural Network
'''
#--------------------------------------------
#### Library Packages
#--------------------------------------------

from __future__ import print_function
import numpy as np
import tflearn

#--------------------------------------------
#### Tkinter
#--------------------------------------------

from tkinter import Frame, Tk, Button, BOTH, Menu, Label, Entry, StringVar

print ('Initiallize Program')

class Window(Frame):

    def __init__(self, master=None):
        Frame.__init__(self, master)

        self.master = master
        self.init_window()

    def init_window(self):
        self.master.title("Orothopedic Network Predictor")
        self.pack(fill=BOTH, expand=1)

        bar = Menu(self.master)
        self.master.config(menu = bar)

        Label_Main = Label(self.master, text='Please Enter Data Below', fg='red')
        Label_Main.place(x=120, y=2)

        file = Menu(bar)
        file.add_command(label='Exit', command=self.client_exit)
        bar.add_cascade(label='File', menu=file)

        xdim = 150
        xdimL = 5

        mEntry1 = Entry(self,textvariable=ent1)
        mEntry1.place(x=xdim,y=30)

        Label1 = Label(self.master, text = 'Pelvic Incidence:')
        Label1.place(x=xdimL, y=30)

        mEntry2 = Entry(self,textvariable=ent2)
        mEntry2.place(x=xdim,y=60)

        Label2 = Label(self.master, text = 'Pelvic Tilt:')
        Label2.place(x=xdimL, y=60)

        mEntry3 = Entry(self,textvariable=ent3)
        mEntry3.place(x=xdim, y=90)

        Label3 = Label(self.master, text = 'Lumbar Lordosis Angle:')
        Label3.place(x=xdimL, y=90)

        mEntry4 = Entry(self,textvariable=ent4)
        mEntry4.place(x=xdim,y=120)

        Label4 = Label(self.master, text = 'Sacral Slope:')
        Label4.place(x=xdimL, y=120)

        mEntry5 = Entry(self,textvariable=ent5)
        mEntry5.place(x=xdim, y=150)

        Label5 = Label(self.master, text = 'Pelvic Radius:')
        Label5.place(x=xdimL, y=150)

        mEntry6 = Entry(self,textvariable=ent6)
        mEntry6.place(x=xdim, y=180)

        Label6 = Label(self.master, text = 'Degree Spondylolisthesis:')
        Label6.place(x=xdimL, y=180)

        entButton = Button(self, text="Run Network", command=self.enter_value)
        entButton.place(x=130,y=460)


    def client_exit(self):
        exit()

    def enter_value(self):
        mtext1 = ent1.get()
        mtext2 = ent2.get()
        mtext3 = ent3.get()
        mtext4 = ent4.get()
        mtext5 = ent5.get()
        mtext6 = ent6.get()


        test_X = np.array([mtext1,mtext2,mtext3,mtext4,mtext5,mtext6])#.reshape(1,6)



#--------------------------------------------
#### Load csv file
#--------------------------------------------

    # Load the csv file, the csv in question has been modified so that 
    # 'Hernia' = 2, 'Spondylolisthesis' = 1 and 'Normal' = 0 
        from tflearn.data_utils import load_csv
        data, labels = load_csv('column_3C_weka - Copy.csv',
                            categorical_labels=True, n_classes=3)


        data =  np.array(data, dtype=np.float32)

#--------------------------------------------
#### Building the model
#--------------------------------------------
        net = tflearn.input_data(shape=[None, 6])
        net = tflearn.fully_connected(net, 32)
        net = tflearn.fully_connected(net, 32)
        net = tflearn.fully_connected(net, 3, activation='softmax')
        net = tflearn.regression(net)

#--------------------------------------------
#### Defining the model
#--------------------------------------------
        model = tflearn.DNN(net, checkpoint_path='model/orthopedic_model_3C.tflearn', 
                            max_checkpoints = 3, tensorboard_verbose = 3,
                            tensorboard_dir='model/tmp/tflearn_logs/')

#--------------------------------------------
#### Train model for N epochs
#--------------------------------------------
#    model.load('model/orthopedic_model_3C.tflearn')

#    model.fit(data, labels, n_epoch=100,  run_id='oprthopedic_model_3C', 
#              batch_size=10, show_metric=True)

#--------------------------------------------
#### Saving and loading the model
#--------------------------------------------
#    model.save('model/orthopedic_model_3C.tflearn')
        model.load('model/orthopedic_model_3C.tflearn')

#--------------------------------------------
#### Predicting with the model
#--------------------------------------------

#Hernia = [37.686,4.010,42.948,30.675,85.241,1.664]

#Spondylolisthesis = [66.536,23.157,46.775,40.378,137.440,15.378]

        pred = model.predict([test_X])
        print("\n")
        print("Prediction")
        print("Normal:", pred[0][0])
        print("Spondylolisthesis:", pred[0][1])
        print("Hernia:", pred[0][2])




root = Tk()

ent1 = StringVar()
ent2 = StringVar()
ent3 = StringVar()
ent4 = StringVar()
ent5 = StringVar()
ent6 = StringVar()


root.geometry("400x400")
app = Window(root)
root.mainloop()


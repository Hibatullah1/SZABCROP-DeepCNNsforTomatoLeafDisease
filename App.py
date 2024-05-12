from PIL import Image, ImageTk
from tkinter import PhotoImage, Toplevel, filedialog, Tk, Button, HORIZONTAL, Frame, Label, StringVar, Radiobutton
from tkinter.ttk import Progressbar, Style
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow

#######################Integration Functions/APIs#######################
model_selected=0
prediction_idx=0
final_model = None
def model_selection(inp):
    print("model selection checkpoint")
    global final_model, model_selected
    if inp == 'Alexnet':
        final_model =tensorflow.keras.models.load_model('../Project/(1) Alexnet/Alexnet.h5')
        model_selected=1
        print("alexnet selected")
    elif inp == 'VGG-16':
        final_model =tensorflow.keras.models.load_model('../Project/(2) VGG16/Model-Vgg16-tomatoleafdisease.h5')
        model_selected=2
        print("vgg16 selected")
    elif inp == 'Resnet50':
        final_model =tensorflow.keras.models.load_model('../Project/(3) Resnet50/Resnet_50.h5')
        model_selected=3
    elif inp == 'Resnet152 V2':    
        final_model =tensorflow.keras.models.load_model('../Project/(4) Resnet152 v2/ResNet152V2.h5')
        model_selected=4

def make_prediction(inp):
    print("checkpoint")
    return final_model.predict(inp).round()[0]

def name_assigner(inp):
    name='none'
    global prediction_idx
    print(inp)
    if inp[0]:
        name='Bacterial spot'
        prediction_idx=1
        print("check name assigned"+str(prediction_idx))
    elif inp[1]:
        name='Early blight'
        prediction_idx=2
    elif inp[2]:
        name='Late Blight'
        prediction_idx=3
    elif inp[3]:
        name='Leaf Mold'
        prediction_idx=4
    elif inp[4]:
        name='Septoria Leaf Spot'
        prediction_idx=5
    elif inp[5]:
        name='Target Spot'
        prediction_idx=6
    elif inp[6]:
        name='Tomato Yellow Leaf Curl Virus'
        prediction_idx=7
    elif inp[7]:
        name='Tomato Mosaic Virus'
        prediction_idx=8
    elif inp[8]:
        name='Two Spotted Spider Mite'
        prediction_idx=9
    elif inp[9]:
        name='Healthy'  
        prediction_idx=10
    return name

def single_img_predictor(decision,path=None,frame=None):
    global model_selected
    if decision ==1:
        img_m = plt.imread(path)
    elif decision ==2:
        img_m = frame
    
    if model_selected==1:
        #Image processing for Matplot Lib imported image
        #Step 1 : Dimensions to 227*227
        height = int((img_m.shape[0]*227)/img_m.shape[0])
        width =  int((img_m.shape[1]*227)/img_m.shape[1])
        dimension =(width,height)
        resized = cv2.resize(img_m,dimension,interpolation=cv2.INTER_AREA)
        #scales the pixel values in the image to values between -1 to +1
        resized = resized.astype('float32') / 255.
        #Step 2 : Transforming into numpy arrays, flattening and giving 4D format supported by model
        resized = np.array(resized.reshape(227,227,3))
        resized = np.array([resized.flatten()])
        resized = resized.reshape(1,227,227,3)
        #plt.imshow(resized.reshape(227,227,3))
    elif model_selected==2:
        #Image processing for Matplot Lib imported image
        #Step 1 : Dimensions to 224*224
        height = int((img_m.shape[0]*224)/img_m.shape[0])
        width =  int((img_m.shape[1]*224)/img_m.shape[1])
        dimension =(width,height)
        resized = cv2.resize(img_m,dimension,interpolation=cv2.INTER_AREA)
        resized = resized.astype('float32') / 255.        
        #Step 2 : Transforming into numpy arrays, flattening and giving 4D format supported by model
        resized = np.array(resized.reshape(224,224,3))
        resized = np.array([resized.flatten()])
        resized = resized.reshape(1,224,224,3)
    elif model_selected==3:
        ##additional step for resnet50 /pre processing
        img_m=preprocess_input(img_m)
        #Image processing for Matplot Lib imported image
        #Step 1 : Dimensions to 224*224
        height = int((img_m.shape[0]*224)/img_m.shape[0])
        width =  int((img_m.shape[1]*224)/img_m.shape[1])
        dimension =(width,height)
        resized = cv2.resize(img_m,dimension,interpolation=cv2.INTER_AREA)
        #Step 2 : Transforming into numpy arrays, flattening and giving 4D format supported by model
        resized = np.array(resized.reshape(224,224,3))
        resized = np.array([resized.flatten()])
        resized = resized.reshape(1,224,224,3)
    elif model_selected==4:
        height = int((img_m.shape[0]*256)/img_m.shape[0])
        width =  int((img_m.shape[1]*256)/img_m.shape[1])
        dimension =(width,height)
        resized = cv2.resize(img_m,dimension,interpolation=cv2.INTER_AREA)
        resized = resized.astype('float32') / 255.        
        #Step 2 : Transforming into numpy arrays, flattening and giving 4D format supported by model
        resized = np.array(resized.reshape(256,256,3))
        resized = np.array([resized.flatten()])
        resized = resized.reshape(1,256,256,3)

    #prediction=final_model.predict(resized).round()
    diagnosis = name_assigner(make_prediction(resized))
    return diagnosis

#######################Front End Code#######################

#helper variables for multiple image processing tasks in Directory Selection.
cu=1
c=0
analyzed=False
cam = None


#######################SPLASH SCREEN#######################

def flash_screen():

    window=Tk()
    
    #aligns window at the centre of screen
    window.geometry("%dx%d+%d+%d"%(900, 360, (window.winfo_screenwidth()/2)-(900/2), (window.winfo_screenheight()/2)-(360/2)))
    
    #removes topbar
    window.overrideredirect(1)
    
    #for progress bar
    s=Style()
    s.theme_use('clam')
    s.configure("red.Horizontal.TProgressbar",foreground='red',background='#7CFC00')
    progress=Progressbar(window,style="red.Horizontal.TProgressbar",orient=HORIZONTAL,length=910,mode='determinate')
    
    #Coordinates of Loading
    progress.place(x=-10,y=350)
    
    #adding frame
    Frame(window,width=900,height=350,bg='#618a3d').place(x=0,y=0)

    def Start():
        Label(window,text='Loading...',fg='white',bg='#618a3d', font=('Calibri (Body)', 10)).place(x=375,y=300)
    
        import time
        r=0
        for i in range(100):
            progress['value']=r
            window.update_idletasks()
            time.sleep(0.01)
            r=r+1

        #Bridge for next screen
        window.destroy()
        first_screen()
    
    # Get Started Button
    Button(window,width=10,height=1,text='Get Started',command=Start,border=0,fg='#618a3d').place(x=375,y=200)

    # Labels
    Label(window,text="SZAB CROP",fg='white',bg="#618a3d", font=('helvetica', 24,'bold')).place(x=350,y=50)#175

    lst2=('Calibri (Body)', 8,'bold')
    l2=Label(window,text="developed by Muhammad Hibatullah Channa and Mohammad Ahmed for FYP 2022 at SZABIST Hyderabad",fg='white',bg="#618a3d", font=lst2)
    l2.place(x=150,y=330)
    
    lst3=('Calibri (Body)', 14)#205
    l3=Label(window,text="AI based tomato leaf disease detection using Deep Convolutional Neural Networks",fg='white',bg="#618a3d", font=lst3)
    l3.place(x=100,y=100)
   
    window.mainloop()

    #######################First Screen : Model Selection Options#######################
    
def first_screen():       
        
    global w1
    w1=Tk()
    w1.geometry("%dx%d+%d+%d"%(900,600,(w1.winfo_screenwidth()/2)-(900/2),(w1.winfo_screenheight()/2)-(600/2)))

    # Adding Frames
    Frame(w1,width=300,height=600,bg='#dfe667').place(x=0,y=0)
    Frame(w1,width=300,height=600,bg='#7ea310').place(x=300,y=0)
    Frame(w1,width=300,height=600,bg='#213502').place(x=600,y=0)
        
    #Logo Label
    Logo = Image.open('../Project/logos/1.png') 
    Logo.thumbnail((150,150))
    Logo = ImageTk.PhotoImage(Logo)
    LogoLabel=Label(w1,bg='#dfe667', image=Logo)
    LogoLabel.place(x=75,y=13)

    #Title
    TitleLabel=Label(w1,text="SZAB CROP", fg='black',bg='#dfe667', font=('Calibri (Body)', 30 ,'bold'))
    TitleLabel.place(x=25,y=175)

    # Brief Info
    InfoLabel1=Label(w1,text="Deep Convolutional Neural Netowork (Artificial Intelligence)", fg='black',bg='#dfe667', font=('Calibri (Body)', 8))
    InfoLabel1.place(x=12,y=225)

    InfoLabel2=Label(w1,text="based tomato leaf disease prediction system", fg='black',bg='#dfe667', font=('Calibri (Body)', 8))
    InfoLabel2.place(x=12,y=245)
    
    # Label: Select Model to Load
    Label(w1,text="Select model to load", fg='black',bg='#dfe667', font=('Calibri (Body)', 18 ,'bold')).place(x=25,y=300)  

    def configWindow(selected):
        configWin = Toplevel()
        if selected == "Alexnet":
            accuracy = "93.76%"
        elif selected == "VGG-16":
            accuracy = "88.39%"
        elif selected == "Resnet50":
            accuracy = "82.09%"
        elif selected == "Resnet152 V2":
            accuracy = "96.78%"

        Label(configWin, text="Configuration Details").pack()
        Label(configWin, text="Model: " + selected).pack()
        Label(configWin, text="Accuracy: " + accuracy).pack()
        global img
        img = PhotoImage(file="configs/" +selected + ".png")
        Label(configWin, image=img).pack()


    #Radio buttons
    MODELS = [
        ("Alexnet","Alexnet"),
        ("VGG-16","VGG-16"),
        ("Resnet50","Resnet50"),
        ("Resnet152 V2","Resnet152 V2")
    ]
    mod = StringVar()
    mod.set('None')
    a=37 # for iteratively adding options at 25 pixels distance
    for text, mode in MODELS:
        Radiobutton(w1, text=text, variable=mod, value=mode,bg='#dfe667').place(x=25,y=325+a)
        a=a+37
    Button(w1, text="config", border=0, command=lambda: configWindow("Alexnet")).place(x=200, y=362)
    Button(w1, text="config", border=0, command=lambda: configWindow("VGG-16")).place(x=200, y=399)
    Button(w1, text="config", border=0, command=lambda: configWindow("Resnet50")).place(x=200, y=436)
    Button(w1, text="config", border=0, command=lambda: configWindow("Resnet152 V2")).place(x=200, y=473)
    
    # Load Model Button
    Button(w1,width=10,height=1,text='Load Model',command=lambda: LoadModel(mod.get()), border=0,fg='#000000').place(x=25,y=530)
    
    def LoadModel(value):
        if value == 'None':
            w_error=Tk()
            w_error.geometry('300x300')
            f2=('Calibri (Body)', 20 ,'bold')
            l1_e=Label(w_error,text="Select model to load", fg='red',bg='#FFFF00', font=f2)
            l1_e.place(x=0,y=10)
        else:
            model_selection(value)
            screen_two()
    
    w1.mainloop()

#######################Screen Two : Image Selection#######################

def screen_two():
        
    global prediction_idx
    prediction_idx=0
    
    # Label: Select Method to Load Images
    Label(w1,text="Select Method to Load Images", fg='black',bg='#7ea310', font=('Calibri (Body)', 14 ,'bold')).place(x=310,y=30)
    
    # Image Selection Buttons
    Button(w1,width=20,height=1,text='Single image selection',command=lambda: LoadImages(1),border=0,fg='#000000').place(x=375,y=100)
    Button(w1,width=20,height=1,text='Directory selection',command=lambda: LoadImages(2),border=0,fg='#000000').place(x=375,y=150)
    Button(w1,width=20,height=1,text='Capture via camera',command=lambda: LoadImages(3),border=0,fg='#000000').place(x=375,y=200)    
    
    def LoadImages(value):
        if value == 1:
            fileTypes = [("All image files",".png .jpg")]
            dirt = filedialog.askopenfilename(initialdir=os.getcwd(),title="Select image file",filetypes=(fileTypes))
            screen_three(value,dirt)
        elif value == 2:
            dirt = filedialog.askdirectory()
            screen_three(value,dirt)
        elif value == 3:
            screen_three(value,None)

#######################Screen Three: Analysis#######################

def screen_three(decision,img_dir):
    
    def camera_session():
        global cam
        cam = cv2.VideoCapture(0)
        img_counter = 0
        captured=False
        
        while True:
            ret, frame = cam.read()
            if not cam.isOpened():
                raise IOError("Cannot use webcam")
            elif cam.isOpened():
                cv2.imshow('frame', frame)
                fr=frame
            if not ret:
                break
            k= cv2.waitKey(1)
            if k == 27:
            #ESC pressed
                break
            elif k == 32:
            #SPACE pressed
                img_name = 'szab-crop-webcam_{}.jpg'.format(img_counter)
                cv2.imwrite('../Project/output/{}'.format(img_name), frame)
                img_counter += 1
                captured=True
                break
        cam.release()
        cv2.destroyAllWindows()
        #print(img_name)
        if captured:
            #cv2.imshow('a',fr)
            fr= cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
            return fr
            #plt.imshow(fr)
        else:
            print('quitted')
        #cv2.imshow('frs',fr)
        
    def show_image(dir_img):
        img=Image.open(dir_img)   
        img.thumbnail((227,227))
        img=ImageTk.PhotoImage(img)
        ImageLabel.config(image=img)
        ImageLabel.image=img
        
    def show_img_by_data(inp):
        img = Image.fromarray(inp)
        img.thumbnail((227,227))
        img=ImageTk.PhotoImage(img)
        ImageLabel.config(image=img)
        ImageLabel.image=img
    
    def previous_next_img(inp):
        global cu,c
        print(" index before op : "+str(cu))
        if inp==1: #if prev
            if cu > 1: #current is greater than 1
                cu=cu-1
                show_image(paths[cu-1])
            analyze()
        elif inp==2: # if next
            if cu < c: #current is lesser than total
                cu=cu+1
                show_image(paths[cu-1])
                print("N-current "+str(cu))
                analyze()
        print(" index after op : "+str(cu))
        
    def analyze(frame=None):
        global analyzed
        if decision==1: #single image
            result=single_img_predictor(1,img_dir,None)
            print_results(result)
        elif decision==2: #multiple images/directory
            analyzed=True
            if analyzed:
                print_results(single_img_predictor(1,paths[cu-1]))
        elif decision==3: # from webcapture
            print_results(single_img_predictor(2,None,frame))    
            print("d3")
    
    # Image Label
    Frame(w1, width=300, height=350, bg='#7ea310').place(x=300, y=250)
    ImageLabel = Label(w1)
    ImageLabel.place(x=336,y=250)
    
    def symptoms():
        global prediction_idx
        diagnosis=''
        if prediction_idx==1:
            diagnosis= '1.Spots on leaves (less than 1/8 inch) \n2.Water soaked(wet looking circular areas) \n3.Extensive leaf loss in severe conditions.'
        elif prediction_idx==2:
            diagnosis= '1.Spots on leaves (upto 1/2 inch) round, brown.\n2.Larger spots have target like conventric rings.\n3.Tissue around spots often turn yellow.\n4.Severely infectected leaves turn brown and fall off'
        elif prediction_idx==3:
            diagnosis= '1.Leaves have large brown blotches with gray edge.\n2.Blotches not confined by major leaf veins.'
        elif prediction_idx==4:
            diagnosis= '1.Spots on leaves, pale greenish-yellow (less than 1/4 inch)\n2.Spots have no definite margins,form on upper sides.\n3.Olive green to brown velvety mold form on lower leaf\nbelow leaf spots.'
        elif prediction_idx==5:
            diagnosis= '1.Spots on leaves, (1/16 to 1/8 inch), circular spots on under side\n2.Water soaked.\n3.Spots are distinctly circular and often numerous'
        elif prediction_idx==6:
            diagnosis= '1.Spots on leaves (enlarge upto 2/5 inch).\n2.Spots show characteristic rings.\n3.Causes leaves to turn yellow, collapse and die.'
        elif prediction_idx==7:
            diagnosis= '1.Leaves become yellow between veins.\n2.Leaves curl upwards and towards the middle of leaf.'
        elif prediction_idx==8:
            diagnosis= '1.Irregular leaf mottling(light and dark freen or\nyellow patches or streaks)\n2.Leaves are stunted, curled or puckered.\n3.Veins may be lighter than normal or banded with\ndark green or yellow'
        elif prediction_idx==9:
            diagnosis= '1.Leaves turn yellow and dry up.\n2.Leaves appear tan or yellow have crusty texture.\n3.Fine webbing may cover the plant.'
        elif prediction_idx==10:
            diagnosis= 'Healthy'
        else:
            diagnosis= 'Error fetching results'

        l5_sympt["text"] = diagnosis
 
        
    def treatment():
        global prediction_idx
        cure=''
        if prediction_idx==1:
            cure= 'No cure available\nRemovesymptomatic plants\nPlant pathogen free seed'
        elif prediction_idx==2:
            cure= 'Plant pathogen free seed\nWater at Base\nIncrease Air flow\npinch leaves and burn'
        elif prediction_idx==3:
            cure= 'Remove diseased leaves\nkeep leaves dry as possible\nRemove and bury plants at end of season'
        elif prediction_idx==4:
            cure= 'Scout during periods of high humidity\nUse drip irrigation\nPlant at spaces to provide air movement'
        elif prediction_idx==5:
            cure= 'Remove diseased leaves\nImprove air circulation\nDo not overhead water\nUse drip irrigation\nUse fungicidial sprays'
        elif prediction_idx==6:
            cure= 'Improve air circulation\nCollect and burn plants after harvest\nCheck seedlings and throw any with leaf spots'
        elif prediction_idx==7:
            cure= 'Use insecticide\nUse drip irrigation'
        elif prediction_idx==8:
            cure= 'No cure available\nRemove infected plants and destroy them\nDisinfect gardening tools'
        elif prediction_idx==9:
            cure= 'Use insecticidial soap to kill spider mites'
        elif prediction_idx==10:
            cure= ''
        else:
            cure= 'Error fetching results'

        l5_prv["text"] = cure
        
    
    def print_results(result):

        l5["text"] = result
        symptoms()
        treatment()
    
    #organizing single img to show
    if decision==1:
        show_image(img_dir)
    #organizing multiple imgs to show
    imgs = []
    paths = []
    global c
    if decision ==2:
        path = img_dir
        valid_images = [".jpg",".gif",".png",".tga"]
        for f in os.listdir(path):
            ext = os.path.splitext(f)[1]
            if ext.lower() not in valid_images:
                continue

            #resize before appending
            ##img = cv2.resize(img,dimension,interpolation=cv2.INTER_AREA)
            imgs.append(Image.open(os.path.join(path,f)).resize((227,227)))  
            paths.append((os.path.join(path,f)))
        
        #Total images count                 
        for i in imgs:
            c=c+1

        show_image(paths[0])
        
    elif decision==3:
        frame_captured=camera_session()
        show_img_by_data(frame_captured)
        print("exe flow")
        
    def analyze_helper():
        if decision==1:
            analyze()
        elif decision==2:
            analyze()
        elif decision==3:
            analyze(frame_captured)

    # Results Labels
    f10=('Calibri (Body)', 18, 'bold')
    Frame(w1, width=300, height=600, bg='#213502').place(x=600, y=0)
    Label(w1,text="Diagnosis: ", fg='white',bg='#213502', font=f10).place(x=605,y=50)
    Label(w1,text="Symptoms: ", fg='white',bg='#213502', font=f10).place(x=605,y=200)
    Label(w1,text="Treatment/Prevention: ", fg='white',bg='#213502', font=f10).place(x=605,y=350)
    
    f5=('Calibri (Body)', 14 )
    l5=Label(w1, fg='white',bg='#213502', font=f5)
    l5.place(x=605,y=90)
        
    f0=('Calibri (Body)', 8)
    l5_sympt=Label(w1, fg='white',bg='#213502', font=f0, anchor='w')
    l5_sympt.place(x=605,y=240)
        
    l5_prv=Label(w1, fg='white',bg='#213502', font=f0)
    l5_prv.place(x=605,y=390)

    # Buttons        
    Button(w1,width=10,height=1,text='Previous',command=lambda: previous_next_img(1), border=0).place(x=336,y=525)    
    Button(w1,width=10,height=1,text='Next',command=lambda: previous_next_img(2), border=0).place(x=487,y=525)
    Button(w1,width=32,height=1,text='Analyze',command=analyze_helper, border=0).place(x=336,y=550)
    Button(w1,width=40,height=1,text='Quit',command=w1.destroy,border=0).place(x=607,y=550)
    
########################Main Function################ 
flash_screen()
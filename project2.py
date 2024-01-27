from ultralytics import YOLO
import cv2
import torch
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog


def add_model():
    file_path = filedialog.askopenfilename(filetypes=[("pt files", "*.pt")])
    adding_model = YOLO(file_path)
    if torch.cuda.is_available == True:
        adding_model.to('cuda')
    else:
        pass
    return adding_model

def analyse_image():
    model = add_model()
    file_path = filedialog.askopenfilename(filetypes=[("all files", "*.*")]) 
    results = model(file_path)

    for r in results:
        im_array = r.plot()  
        im = Image.fromarray(im_array[..., ::-1])  
        im.show() 

def analyse_video():

    
    model = add_model()
    file_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.avi *.mov")])
    cap = cv2.VideoCapture(file_path)

    while cap.isOpened():
        success, frame = cap.read()

        if success:
            results = model(frame)

            annotated_frame = results[0].plot()

            cv2.imshow("YOLOv8 Inference", annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            break

    cap.release()
    cv2.destroyAllWindows()

def stream():
    device = 0 if torch.cuda.is_available() else 'cpu' 
    if device == 0:
        torch.cuda.set_device(0)
    
    model = add_model()
    video = int(0) 
    cap = cv2.VideoCapture(video)

    while cap.isOpened():
        success, frame = cap.read()

        if success:
            results = model(frame)

            annotated_frame = results[0].plot()

            cv2.imshow("YOLOv8 Inference", annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            break

    cap.release()
    cv2.destroyAllWindows()


root = tk.Tk()
root.title('Deniz App')

canvas = tk.Canvas(root, width=800, height=600)
canvas.pack(fill='both', expand=True)

class_selection = tk.StringVar()
class_selection.set("All") 
class_selection_label = tk.Label(root, text="Select Class:")
class_selection_label.pack(side='left')
class_selection_entry = tk.OptionMenu(root, class_selection, "All")  
class_selection_entry.pack(side='left')

button_frame = tk.Frame(root)
button_frame.pack(fill='x')

analyse_video_button = tk.Button(button_frame,text='video analyse', command=analyse_video)
analyse_video_button.pack(side='left')

analyse_image_button = tk.Button(button_frame, text='image analyse', command=analyse_image)
analyse_image_button.pack(side='left')

stream_button = tk.Button(button_frame,text='stream', command=stream)
stream_button.pack(side='left')

initial_image = Image.open('siyah.png') 
initial_photo = ImageTk.PhotoImage(image=initial_image)
canvas.img = initial_photo
canvas.create_image(0, 0, anchor=tk.NW, image=initial_photo)

root.mainloop()




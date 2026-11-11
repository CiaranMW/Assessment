from ultralytics import YOLO
if __name__ == '__main__':

    # Load your exported weights
    model = YOLO("best.pt")
    
    results = model(source="Image inputs", conf=0.7, save=True)

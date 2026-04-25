from ultralytics import YOLO

model = YOLO("yoloe-26s-seg-pf.pt")
results = model.predict("pics/chevelle1.jpg", save=True, conf=0.1)

for box in results[0].boxes:
    class_name = results[0].names[int(box.cls)]
    confidence = float(box.conf)
    print(f"  {class_name}: {confidence:.1%}")
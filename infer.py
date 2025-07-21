from ultralytics import YOLO
import cv2
from operator import mul


color = {
    "banana": (28,197,209),
    "rambutan": (102,102,255),
    "person": (42, 4, 112),
    "apple": (137, 106, 42),
    "orange": (62, 118, 222),
    "cell phone": (182, 222, 62),
}

# Load a model
yolo_model = YOLO(R"yolov8n.pt")  # pretrained YOLOv8n model

def expo_infer(yolo_model, frame, device="cpu"):
    global cls_names
    # results = yolo_model.track(frame, persist=True, device=device, iou=0.25, conf=0.4)
    results = yolo_model(frame, device=device, iou=0.5, conf=0.6)
    if results[0].boxes is None:
        return None
    boxes = results[0].boxes.xyxyn.cpu().tolist()
    # track_ids = results[0].boxes.id.int().cpu().tolist()
    confs = results[0].boxes.conf.cpu().tolist()
    cls = results[0].boxes.cls.int().cpu().tolist()
    cls_names = results[0].names
    if boxes:
        return boxes, confs, cls
    else:
        return None
    
def draw(frame, results, width, height):
    global cls_names
    frame = cv2.resize(frame, (int(width), int(height)))
    # Plot the tracks
    for box, conf, cls in zip(*results):
        x, y, x2, y2 = map(mul, box, [width, height, width, height])

        inferred = cv2.rectangle(frame, (int(x), int(y)), (int(x2), int(y2)), color.get(cls_names[cls], (255, 0, 0)), 2)
        text_size, _ = cv2.getTextSize(f"{cls_names[cls]} {conf * 100:.2f}%", cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        text_w, text_h = text_size
        #cv2.rectangle(inferred, (int(x), int(y) - text_h), (int(x) + text_w, int(y)), [255,0,0], -1)
        cv2.rectangle(inferred, (int(x), int(y) - text_h), (int(x) + text_w, int(y)), color.get(cls_names[cls], (255, 0, 0)), -1)
	
        # image_inferred = cv2.putText(inferred, f"id:{track_id} {conf * 100:.2f}%", (int(x), int(y)),
        #                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255),
        #                              1, cv2.LINE_AA)
        image_inferred = cv2.putText(inferred, f"{cls_names[cls]} {conf * 100:.2f}%", (int(x), int(y)),
                                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255),
                                     1, cv2.LINE_AA)

    return image_inferred

def main():   
    source = cv2.VideoCapture(R"C:\Users\LeaM\Downloads\20250307174644.ts")
    width = 1920 #1920 #640 #cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = 1080 #1080 #640 #cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    while source.isOpened():
        # Read a frame from the video
        success, frame = source.read()
        try:
            if success:
                resized_frame = cv2.resize(frame, (640, 480))
                # Run batched inference on a list of images
                results = expo_infer(yolo_model, resized_frame)  # return a generator of Results objects

                 # Get the boxes and track IDs
                if results is None:
                    frame = cv2.resize(frame, (int(width), int(height)))
                    cv2.imshow("Inference", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                    continue
                image_inferred = draw(frame, results, width, height)
                # Display the annotated frame
                cv2.imshow("Inference", image_inferred)
                # Break the loop if 'q' is pressed
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            else:
                # Break the loop if the end of the video is reached
                break

        except Exception as e:
            print(e)
    # Release the video capture object and close the display window
    print('cap release')
    source.release()
    print(f"Inference Done: {source}")

    print('destroy all windows')
    cv2.destroyAllWindows("Inference")

if __name__ == '__main__':
    main()


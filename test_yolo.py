import onnxruntime as ort
import numpy as np
import cv2
import sys
from vision.ssd.mobilenetv1_ssd import create_mobilenetv1_ssd, create_mobilenetv1_ssd_predictor
from ultralytics import YOLO
from detect_character import recognition_charv2, arrange_correspond
import time

net_type = 'mb1-ssd-lite'
model_path = 'models/plate/az_plate/az_plate_ssdmobilenetv1.onnx'
model_ocr= YOLO('models/ocr/yolo_ocr_recognition.pt')
label_path = 'models/plate/az_plate/labels.txt'
label_path_ocr = 'models/ocr/labels.txt'
image_path = 'ImageQuy_1.jpg'

# Tải các tên lớp
class_names = [name.strip() for name in open(label_path).readlines()]
class_name_ocr = [name.strip() for name in open(label_path_ocr).readlines()]

# Tải mô hình ONNX
session = ort.InferenceSession(model_path)

predictor = create_mobilenetv1_ssd_predictor(session, candidate_size=200)

# Chuẩn bị ảnh đầu vào
orig_image = cv2.imread(image_path)
image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)

def test_yolo_process(image):
    boxes, labels, probs = predictor.predict(image, 10, 0.4)

    # Xử lý sau và vẽ các hộp lên ảnh gốc
    for i in range(boxes.size(0)):
        box = boxes[i]
        label = int(labels[i])
        score = probs[i]
        width = box[2] - box[0]
        height = box[3] - box[1]
        aspect_ratio = width / height
        # print(aspect_ratio)
        label_list = []

        if 0 <= aspect_ratio <= 2.0:
            plate = orig_image[int(box[1]):int(box[3]),int(box[0]): int(box[2])]
            
            # gray_plate = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
            # img = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
            # img = cv2.equalizeHist(img)
            # img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            img = cv2.GaussianBlur(plate, (3, 3), 0)
            results = model_ocr(img)

            output_plate = 'plate.jpg'
            cv2.imwrite(output_plate, img)
            for result in results:

                label_list = []

                index_char = result.boxes.cls
                conf_char = result.boxes.conf
                box_char = result.boxes.xyxy

                (up_list, label_up_list, probs_up_list), (low_list, label_low_list, probs_low_list)= recognition_charv2(index_char, conf_char, box_char)

                for j in range(len(up_list)):
                    label_ocr = int(label_up_list[j])
                    label_list.append(class_name_ocr[label_ocr]) 

                for j in range(len(low_list)):
                    label_ocr = int(label_low_list[j])
                    label_list.append(class_name_ocr[label_ocr])
        else:
            plate = orig_image[int(box[1]):int(box[3]),int(box[0]): int(box[2])]
            results = model_ocr(plate)
            
            for result in results:
                label_list = []

                index_char = result.boxes.cls
                conf_char = result.boxes.conf
                box_char = result.boxes.xyxy
        
                up_list, label_up_list, probs_up_list = arrange_correspond(box_char, index_char, conf_char)

                for j in range(len(up_list)):
                    label_ocr = int(label_up_list[j])

                    label_list.append(class_name_ocr[label_ocr])

        label_text_ocr = ''.join(label_list)
        cv2.rectangle(orig_image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 255, 0), 4)
        # label_text = f"{class_names[label]}: {score:.2f}"
        cv2.putText(orig_image, label_text_ocr, (int(box[0]) + 20, int(box[1]) - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

    output_path = "run_ssd_example_output.jpg"
    cv2.imwrite(output_path, orig_image)
    print(f"Đã tìm thấy {len(probs)} đối tượng. Ảnh đầu ra là {output_path}")

def process_frame(frame):
    boxes, labels, probs = predictor.predict(frame, 10, 0.4)

    # Xử lý sau và vẽ các hộp lên ảnh gốc
    for i in range(boxes.size(0)):
        box = boxes[i]
        label = int(labels[i])
        score = probs[i]
        width = box[2] - box[0]
        height = box[3] - box[1]

        # Check for valid box dimensions
        if width <= 0 or height <= 0:
            continue

        aspect_ratio = width / height

        label_list = []

        if 0 <= aspect_ratio <= 2.0:
            plate = frame[int(box[1]):int(box[3]), int(box[0]): int(box[2])]
            if plate.size == 0:  # Check if the plate is empty
                continue

            results = model_ocr(plate)

            for result in results:
                label_list = []

                index_char = result.boxes.cls
                conf_char = result.boxes.conf
                box_char = result.boxes.xyxy
                
                (up_list, label_up_list, probs_up_list), (low_list, label_low_list, probs_low_list) = recognition_charv2(index_char, conf_char, box_char)

                for j in range(len(up_list)):
                    label_ocr = int(label_up_list[j])
                    label_list.append(class_name_ocr[label_ocr]) 

                for j in range(len(low_list)):
                    label_ocr = int(label_low_list[j])
                    label_list.append(class_name_ocr[label_ocr])
        else:
            plate = frame[int(box[1]):int(box[3]), int(box[0]): int(box[2])]
            if plate.size == 0:  # Check if the plate is empty
                continue
            
            results = model_ocr(plate)
            
            for result in results:
                label_list = []

                index_char = result.boxes.cls
                conf_char = result.boxes.conf
                box_char = result.boxes.xyxy
        
                up_list, label_up_list, probs_up_list = arrange_correspond(box_char, index_char, conf_char)

                for j in range(len(up_list)):
                    label_ocr = int(label_up_list[j])
                    label_list.append(class_name_ocr[label_ocr])

        label_text_ocr = ''.join(label_list)
        cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 255, 0), 4)
        cv2.putText(frame, label_text_ocr, (int(box[0]) + 20, int(box[1]) - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
    
    return frame

def process_video(input_video_path, output_video_path):
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Define the codec and create VideoWriter object
    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'XVID'), fps, (frame_width, frame_height))

    frame_number = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        prev_time = time.time()
        img_tmp = frame
        processed_frame = process_frame(frame)

        tot_time = time.time() - prev_time
        # fps = round(1 / tot_time,2)

        cv2.putText(img_tmp, 'frame: %d fps: %s' % (frame_number, fps),
                        (0, int(100 * 1)), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), thickness=2)
        out.write(processed_frame)

        frame_number += 1
        print(f'Processing frame {frame_number}/{total_frames}', end='\r')
    
    cap.release()
    out.release()
    print("\nProcessing complete.")

# Example usage
# process_video('input_video.mp4', 'output_video.avi', predictor, model_ocr, recognition_charv2, class_name_ocr)


# Example usage
# process_video('input_video.mp4', 'output_video.avi', predictor, model_ocr, recognition_charv2, class_name_ocr)

# Lưu ảnh đầu ra
INPUT_DIR = './test_video/test_1.mp4' 
OUT_PATH = './results/test_1_test.mp4'

# process_video(INPUT_DIR, OUT_PATH)

test_yolo_process(image)

import time
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import tensorflow as tf
import cv2


from model import Yolo_v3
from utils import load_class_names, load_weights


_MODEL_SIZE = (416, 416)
class_names = load_class_names('coco.names')
n_classes = len(class_names)
max_output_size = 10
iou_threshold = 0.5
confidence_threshold = 0.5
colour = [255, 12, 12]
video_path: str|int = 0  # 0 for webcam, or path to video file or can be YouTube link (https://www.youtube.com/watch?v=dQw4w9WgXcQ)


def _load_model():
    model = Yolo_v3(n_classes=n_classes, model_size=_MODEL_SIZE,
                max_output_size=max_output_size,
                iou_threshold=iou_threshold,
                confidence_threshold=confidence_threshold)
    inputs = tf.compat.v1.placeholder(tf.float32, [1, 416, 416, 3])
    detections = model(inputs, training=False)

    model_vars = tf.compat.v1.global_variables(scope='yolo_v3_model')
    assign_ops = load_weights(model_vars, 'yolov3.weights')
    return inputs, detections, assign_ops

def _draw_bindbox(frame, boxes_dict, class_names):
    """Draws detected boxes on video frames.

    Args:
        frames: A list of input video frames.
        boxes_dicts: A list of class-to-boxes dictionaries for each frame.
        class_names: A list of class names.
    """
    for cls in range(len(class_names)):
        boxes = boxes_dict[cls]
        if np.size(boxes) != 0:
            for box in boxes:
                xy, confidence = box[:4], box[4]
                xy = [int(xy[i] * frame.shape[1] / _MODEL_SIZE[0]) if i % 2 == 0 else
                        int(xy[i] * frame.shape[0] / _MODEL_SIZE[1]) for i in range(4)]
                x0, y0, x1, y1 = xy
                thickness = (frame.shape[0] + frame.shape[1]) // 200
                for t in np.linspace(0, 1, thickness):
                    x0_t, y0_t = int(x0 + t), int(y0 + t)
                    x1_t, y1_t = int(x1 - t), int(y1 - t)
                    cv2.rectangle(frame, (x0_t, y0_t), (x1_t, y1_t), colour, 1)
                
                text = '{} {:.1f}%'.format(class_names[cls], confidence * 100)
                text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                cv2.rectangle(frame, (x0, y0 - text_size[1]), (x0 + text_size[0], y0), colour, -1)
                cv2.putText(frame, text, (x0, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    return frame


def _load_from_youtube(url: str) -> str:
    import yt_dlp
    ydl_opts = {
        'outtmpl': 'tmp/video.%(ext)s',
        'format': 'best[height<=480][ext=mp4]',
    }
    print('downloading video...')
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info_dict = ydl.extract_info(url, download=True)
    print('video downloaded.')
    return 'tmp/video.mp4'


if __name__ == '__main__':
    with tf.compat.v1.Session() as sess:
        inputs, detections, assign_ops = _load_model()
        sess.run(assign_ops)
        print('starting video stream...')
        if isinstance(video_path, str) and video_path.startswith('https://'):
            video_path = _load_from_youtube(video_path)
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("Error: Could not open camera.")
            exit()

        frame_count = 0
        start_time = time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame.")
                exit()
            resized_frame = cv2.resize(frame, (416, 416))
            detection_result = sess.run(detections, feed_dict={inputs: [resized_frame]})[0]
            
            frame_with_boxes = _draw_bindbox(frame, detection_result, [class_names[0]])

            frame_count += 1
            elapsed_time = time.time() - start_time
            fps = frame_count / elapsed_time

            fps_text = f"FPS: {fps:.2f}"    
            cv2.putText(frame_with_boxes, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            cv2.imshow('Object Detection', frame_with_boxes)
            if cv2.waitKey(1) == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()




    


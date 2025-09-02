import collections
import time
from pose_engine import PoseEngine
from pose_engine import KeypointType
import numpy as np #type:ignore
import argparse
import os
import cv2 #type:ignore
import soundmachine


#DEBUG ONLY
def avg_fps_counter(window_size):
    window = collections.deque(maxlen=window_size)
    prev = time.monotonic()
    yield 0.0  # First fps value.

    while True:
        curr = time.monotonic()
        window.append(curr - prev)
        prev = curr
        yield len(window) / sum(window)

def subtracter(point_a, point_b):
    return [point_a.point.x-point_b.point.x,point_a.point.y-point_b.point.y]
    

def cinema(pose, dims, path, threshold=0.3,ind_threshold=0.2):
    if(pose.score<threshold):
        return
    
    shoulder_threshold=.1
    elbow_threshold=.1
    wrist_threshold=.1
    
    left_wrist = pose.keypoints[KeypointType.LEFT_WRIST]
    right_wrist=pose.keypoints[KeypointType.RIGHT_WRIST]
    left_elbow=pose.keypoints[KeypointType.LEFT_ELBOW]
    right_elbow=pose.keypoints[KeypointType.RIGHT_ELBOW]
    left_shoulder=pose.keypoints[KeypointType.LEFT_SHOULDER]
    right_shoulder=pose.keypoints[KeypointType.RIGHT_SHOULDER]

    all_things=(left_wrist,right_wrist,left_elbow, right_elbow, left_shoulder,right_shoulder)

    for i in all_things:
        if i.score<ind_threshold:
            return 
    
    lwe=subtracter(left_wrist, left_elbow)
    les=subtracter(left_elbow, left_shoulder)
    sts=subtracter(right_shoulder, left_shoulder)
    res=subtracter(right_elbow, right_shoulder)
    rwe=subtracter(right_wrist, right_elbow)
    all_vectors=(lwe,les,sts,res,rwe)
    for i in range(0,5):
        for j in range(0,2):
            if all_vectors[i][j]==0:
                all_vectors[i][j]==.01
    
    if(lwe[0]/lwe[1]<wrist_threshold and rwe[0]/rwe[1]<wrist_threshold and les[1]/les[0]<elbow_threshold and res[1]/res[0]<elbow_threshold and sts[1]/sts[0]<shoulder_threshold):
        soundmachine.player(path)




def worker(engine, device, dims,path):
    cap = cv2.VideoCapture(device)
    width=dims[0]
    height=dims[1]
    n, sum_process_time, sum_inference_time, fps_counter=0,0,0,0
    
    fps_counter_machine=avg_fps_counter(30)
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Camera not working")
            exit()
        resized = cv2.resize(frame, dims)
        final = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        input_tensor = np.ndarray.flatten(np.asarray(final, dtype=np.uint8))
        engine.run_inference(input_tensor)
        
        #DEBUG
        start_time = time.monotonic()
        
        outputs, inference_time=engine.ParseOutput()
        
        #DEBUG
        end_time = time.monotonic()
        n += 1
        sum_process_time += 1000 * (end_time - start_time)
        sum_inference_time += inference_time * 1000
        avg_inference_time = sum_inference_time / n
        text_line = 'PoseNet: %.1fms (%.2f fps) TrueFPS: %.2f Nposes %d' % (
            avg_inference_time, 1000 / avg_inference_time, next(fps_counter_machine), len(outputs)
        )
        print(text_line)

        for pose in outputs:
            cinema(pose, path, dims)

        


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model', help='.tflite model path.', required=False)
    parser.add_argument('--audio', help='path to audio', default="cinema.mp3")
    parser.add_argument('--res', help='Resolution', default='640x480',
                        choices=['480x360', '640x480', '1280x720'])
    parser.add_argument('--videosrc', help='Which video source to use', default='0')
    args = parser.parse_args()
    default_model = 'models/mobilenet/posenet_mobilenet_v1_075_%d_%d_quant_decoder_edgetpu.tflite'
    if args.res == '480x360':
        model = args.model or default_model % (353, 481)
    elif args.res == '640x480':
        model = args.model or default_model % (481, 641)
    elif args.res == '1280x720':
        model = args.model or default_model % (721, 1281)
    engine = PoseEngine(model)
    input_shape = engine.get_input_tensor_shape()
    inference_size = (input_shape[2], input_shape[1])
    worker(engine, args.videosrc, inference_size, args.audio)

if __name__ == '__main__':
    main()

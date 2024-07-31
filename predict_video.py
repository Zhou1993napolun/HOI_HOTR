import argparse
import itertools
import cv2
from PIL import Image
import numpy as np
import torch
import os
from pathlib import Path
import hotr.data.datasets as datasets
import hotr.data.transforms.transforms as T
import hotr.util.misc as utils
from hotr.engine.arg_parser import get_args_parser
from hotr.models import build_model

from data.videoloader import LoadImages, LoadWebcam
import time



from flask import Flask, render_template, Response
import threading

# ======================================================
# =================== Flask ============================
# ======================================================
app = Flask(__name__)

outputFrame = None
lock = threading.Lock()

# 创建一个停止事件
stop_event = threading.Event()


def generate():
    global outputFrame, lock
    while True:
        with lock:
            if outputFrame is None:
                continue
            # 将帧编码为JPEG格式
            ret, buffer = cv2.imencode('.jpg', outputFrame)
            frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        
        
@app.route('/video_feed')
def video_feed():
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return render_template('index.html')


# ======================================================
# =================== Predict ==========================
# ======================================================

def predict(args,stop_event):
    global lock, outputFrame
    
    timestart = time.time()
    device = torch.device(args.device)

    # Data Setup
    meta = datasets.builtin_meta._get_coco_instances_meta()
    COCO_CLASSES = meta['coco_classes']
    args.num_classes = len(COCO_CLASSES)
    _valid_obj_ids = [id for id in meta['thing_dataset_id_to_contiguous_id'].keys()]
    with open(args.action_list_file, 'r') as f:
        action_lines = f.readlines()
    _valid_verb_ids, _valid_verb_names = [], []
    for action_line in action_lines[2:]:
        act_id, act_name = action_line.split()
        _valid_verb_ids.append(int(act_id))
        _valid_verb_names.append(act_name)
    args.num_actions = len(_valid_verb_ids)
    args.action_names = _valid_verb_names
    if args.share_enc: args.hoi_enc_layers = args.enc_layers
    if args.pretrained_dec: args.hoi_dec_layers = args.dec_layers

    args.valid_obj_ids = _valid_obj_ids
    correct_mat = np.load(args.correct_path)
    print(args)

    # Model Setup
    model, criterion, postprocessors = build_model(args)
    model.to(device)

    model_without_ddp = model

    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])

    if args.eval:
        # test only mode
        if args.camera:
            if args.camera == -1:
                # videoloader = LoadWebcam(pipe='http://pi:raspberry@192.168.12.150:8090/stream.mjpg')
                adress = 'http://'+ args.inputip +'/stream.mjpg'
                print(adress)
                # videoloader = LoadWebcam(pipe='http://192.168.12.150:8090/stream.mjpg')
                videoloader = LoadWebcam(pipe=adress)
            else:
                videoloader = LoadImages(args.img_dir)

        for i, (path, img, img_ori, vid_cap) in enumerate(videoloader):
            timeframe = time.time()
            img = Image.fromarray(img.astype('uint8')).convert('RGB')
            w, h = img.size
            orig_size = torch.as_tensor([h, w]).unsqueeze(0).to(device)
            normalize = T.Compose([
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            transforms = T.Compose([
                T.RandomResize([800], max_size=1333),
                normalize,
            ])
            img, _ = transforms(img, None)
            batch = utils.collate_fn([(img, None)])

            # Model evaluation
            model.eval()
            preds = []
            hoi_recognition_time = []

            samples = batch[0]
            samples = samples.to(device)

            outputs = model(samples)
            results = postprocessors['hoi'](outputs, orig_size, threshold=0, dataset='hico-det')
            hoi_recognition_time.append(results[0]['hoi_recognition_time'] * 1000)

            preds.extend(list(itertools.chain.from_iterable(utils.all_gather(results))))

            preds = [img_preds for i, img_preds in enumerate(preds)]
            max_hois = 100
            conf_thres = args.conf_thres
            img_save = img_ori.copy()
            verb_name_proper = 'nothing'
            obj_name = 'nothing'
            sub_name = 'nothing'
            for img_preds in preds:
                img_preds = {k: v.to('cpu').numpy() for k, v in img_preds.items() if k != 'hoi_recognition_time'}
                bboxes = [{'bbox': bbox, 'category_id': label} for bbox, label in
                          zip(img_preds['boxes'], img_preds['labels'])]
                hoi_scores = img_preds['verb_scores']
                verb_labels = np.tile(np.arange(hoi_scores.shape[1]), (hoi_scores.shape[0], 1))
                subject_ids = np.tile(img_preds['sub_ids'], (hoi_scores.shape[1], 1)).T
                object_ids = np.tile(img_preds['obj_ids'], (hoi_scores.shape[1], 1)).T

                hoi_scores = hoi_scores.ravel()
                verb_labels = verb_labels.ravel()
                subject_ids = subject_ids.ravel()
                object_ids = object_ids.ravel()

                if len(subject_ids) > 0:
                    object_labels = np.array([bboxes[object_id]['category_id'] for object_id in object_ids])
                    masks = correct_mat[verb_labels, object_labels]
                    hoi_scores *= masks

                    hois = [
                        {'subject_id': subject_id, 'object_id': object_id, 'category_id': category_id, 'score': score}
                        for
                        subject_id, object_id, category_id, score in
                        zip(subject_ids, object_ids, verb_labels, hoi_scores) if score > conf_thres]
                    hois.sort(key=lambda k: (k.get('score', 0)), reverse=True)
                    hois = hois[:max_hois]
                else:
                    hois = []

                sub_obj_set = {None}
                highscore = 0

                for i, hoi in enumerate(hois):
                    # print(f'HOI {i}: {hoi}')


                    sub_id = hoi['subject_id']
                    obj_id = hoi['object_id']
                    category_id = hoi['category_id']

                    """
                    print(f'subject is {sub_id}')
                    print(f'object is {obj_id}')
                    print(f'interaction is {category_id}')
                    """

                    if not (sub_id, obj_id) in sub_obj_set:
                        sub_obj_set.add((sub_id, obj_id))

                        verb = hoi['category_id']
                        verb_name = args.action_names[verb]
                        verb_name_proper = verb_name.replace('_', ' ')
                        sub_box = bboxes[sub_id]['bbox'].astype(np.int32)
                        obj_box = bboxes[obj_id]['bbox'].astype(np.int32)
                        obj_cls = bboxes[obj_id]['category_id']
                        score = hoi['score']
                        obj_name = COCO_CLASSES[args.valid_obj_ids[obj_cls]]
                        sub_cls = bboxes[sub_id]['category_id']
                        sub_name = COCO_CLASSES[args.valid_obj_ids[sub_cls]]



                        if score > highscore:
                            highscore = score
                            cv2.rectangle(img_save, (sub_box[0], sub_box[1]), (sub_box[2], sub_box[3]),
                                          color=(255, 0, 0))
                            cv2.rectangle(img_save, (obj_box[0], obj_box[1]), (obj_box[2], obj_box[3]),
                                          color=(0, 0, 255))
                            cv2.putText(img_save, f'{verb_name_proper} {obj_name} : {score:.3f}',
                                        (sub_box[0], sub_box[1] - 10), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 1)


                print(sub_name +' ' + verb_name_proper, obj_name)
            
            with lock:
                outputFrame = img_save.copy()
                if stop_event.is_set():
                    break


    timeend = time.time()
    timeprocess = timeend - timestart
    print('Main function runtime: %.2f' % timeprocess)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        'End-to-End Human Object Interaction training and evaluation script',
        parents=[get_args_parser()]
    )
    parser.add_argument('--action_list_file', default='data/hico_20160224_det/list_action.txt', type=str)
    parser.add_argument('--correct_path', default='data/hico_20160224_det/corre_hico.npy', type=str)
    parser.add_argument('--img_dir', default='test.jpg', type=str, help='image to inference')
    parser.add_argument('--camera', default=-1, type=int, help='Index or ip of camera.')
    parser.add_argument('--outpath',default = './output/temp/',type = str, help = 'path to store output video')
    parser.add_argument('--conf_thres',default = 0.33, type = float, help = 'confidence threshold of verbs')
    parser.add_argument('--outputip', type=str, default='0.0.0.0:5000', help='Output IP and port, e.g., 192.168.23.7:5000')
    parser.add_argument('--inputip', type=str, default='192.168.12.150:8090', help='Iutput IP and port, e.g., 192.168.23.7:5000')
    args = parser.parse_args()
    args.action_list_file = 'data/hico_20160224_det/list_action.txt'
    args.correct_path = 'data/hico_20160224_det/corre_hico.npy'
    args.img_dir = './input'
    # args.camera = -1

    ###################ip####################
    output_ip, output_port = args.outputip.split(':')
    print(output_ip, output_port)
    # Start the predict thread
    t = threading.Thread(target=predict, args=(args, stop_event))
    t.daemon = True
    t.start()
    try:
            # Start the Flask application
        app.run(host=output_ip, port=int(output_port), debug=True, use_reloader=False)
    except KeyboardInterrupt:
        # Set the stop event
        stop_event.set()
        t.join()



import torch, torchvision
import matplotlib.pyplot as plt
import json
import cv2
import numpy as np
from copy import deepcopy
import argparse
import os
import time

from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.structures.image_list import ImageList
from detectron2.data import transforms as T
from detectron2.modeling.box_regression import Box2BoxTransform
from detectron2.modeling.roi_heads.fast_rcnn import FastRCNNOutputs
from detectron2.structures.boxes import Boxes
from detectron2.layers import nms
from detectron2 import model_zoo
from detectron2.config import get_cfg

from transformers import BertTokenizer, VisualBertForPreTraining


def load_config_and_model_weights(cfg_path):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(cfg_path))

    # ROI HEADS SCORE THRESHOLD
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5

    # Comment the next line if you're using 'cuda'
    cfg['MODEL']['DEVICE']='cpu'

    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(cfg_path)

    return cfg

def get_model(cfg):
    # build model
    model = build_model(cfg)

    # load weights
    checkpointer = DetectionCheckpointer(model)
    checkpointer.load(cfg.MODEL.WEIGHTS)

    # eval mode
    model.eval()
    return model

def prepare_image_inputs(cfg, img_list, model):
    # Resizing the image according to the configuration
    transform_gen = T.ResizeShortestEdge(
                [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
            )
    img_list = [transform_gen.get_transform(img).apply_image(img) for img in img_list]

    # Convert to C,H,W format
    convert_to_tensor = lambda x: torch.Tensor(x.astype("float32").transpose(2, 0, 1))

    batched_inputs = [{"image":convert_to_tensor(img), "height": img.shape[0], "width": img.shape[1]} for img in img_list]

    # Normalizing the image
    num_channels = len(cfg.MODEL.PIXEL_MEAN)
    pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).view(num_channels, 1, 1)
    pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).view(num_channels, 1, 1)
    normalizer = lambda x: (x - pixel_mean) / pixel_std
    images = [normalizer(x["image"]) for x in batched_inputs]

    # Convert to ImageList
    images =  ImageList.from_tensors(images,model.backbone.size_divisibility)
    
    return images, batched_inputs

# get ResNet+FPN features
def get_features(model, images):
    features = model.backbone(images.tensor)
    return features

# get region proposals from RPN
def get_proposals(model, images, features):
    proposals, _ = model.proposal_generator(images, features)
    return proposals

# Get Box Features for the proposals
def get_box_features(args, model, features, proposals):
    features_list = [features[f] for f in ['p2', 'p3', 'p4', 'p5']]
    box_features = model.roi_heads.box_pooler(features_list, [x.proposal_boxes for x in proposals])
    box_features = model.roi_heads.box_head.flatten(box_features)
    box_features = model.roi_heads.box_head.fc1(box_features)
    box_features = model.roi_heads.box_head.fc_relu1(box_features)
    box_features = model.roi_heads.box_head.fc2(box_features)

    box_features = box_features.reshape(args.batch_size, 1000, 1024) # depends on your config and batch size
    return box_features, features_list

# Get prediction logits and boxes
def get_prediction_logits(model, features_list, proposals):
    cls_features = model.roi_heads.box_pooler(features_list, [x.proposal_boxes for x in proposals])
    cls_features = model.roi_heads.box_head(cls_features)
    pred_class_logits, pred_proposal_deltas = model.roi_heads.box_predictor(cls_features)
    return pred_class_logits, pred_proposal_deltas

# Get FastRCNN scores and boxes
def get_box_scores(cfg, pred_class_logits, pred_proposal_deltas, proposals):
    box2box_transform = Box2BoxTransform(weights=cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_WEIGHTS)
    smooth_l1_beta = cfg.MODEL.ROI_BOX_HEAD.SMOOTH_L1_BETA

    outputs = FastRCNNOutputs(
        box2box_transform,
        pred_class_logits,
        pred_proposal_deltas,
        proposals,
        smooth_l1_beta,
    )

    boxes = outputs.predict_boxes()
    scores = outputs.predict_probs()
    image_shapes = outputs.image_shapes

    return boxes, scores, image_shapes

# Rescale the boxes to original image size
def get_output_boxes(boxes, batched_inputs, image_size):
    proposal_boxes = boxes.reshape(-1, 4).clone()
    scale_x, scale_y = (batched_inputs["width"] / image_size[1], batched_inputs["height"] / image_size[0])
    output_boxes = Boxes(proposal_boxes)

    output_boxes.scale(scale_x, scale_y)
    output_boxes.clip(image_size)

    return output_boxes

# Select the Boxes using NMS
def select_boxes(cfg, output_boxes, scores):
    test_score_thresh = cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST
    test_nms_thresh = cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST
    cls_prob = scores.detach()
    cls_boxes = output_boxes.tensor.detach().reshape(1000,80,4)
    max_conf = torch.zeros((cls_boxes.shape[0]))
    for cls_ind in range(0, cls_prob.shape[1]-1):
        cls_scores = cls_prob[:, cls_ind+1]
        det_boxes = cls_boxes[:,cls_ind,:]
        keep = np.array(nms(det_boxes, cls_scores, test_nms_thresh))
        max_conf[keep] = torch.where(cls_scores[keep] > max_conf[keep], cls_scores[keep], max_conf[keep])
    keep_boxes = torch.where(max_conf >= test_score_thresh)[0]
    return keep_boxes, max_conf

# Limit the total number of boxes
def filter_boxes(keep_boxes, max_conf, min_boxes, max_boxes):
    if len(keep_boxes) < min_boxes:
        keep_boxes = np.argsort(max_conf).numpy()[::-1][:min_boxes]
    elif len(keep_boxes) > max_boxes:
        keep_boxes = np.argsort(max_conf).numpy()[::-1][:max_boxes]
    return keep_boxes

# Get the visual embeddings
def get_visual_embeds(box_features, keep_boxes):
    return box_features[keep_boxes.copy()]

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=1, type=int, help='batch size')
    parser.add_argument('--precision', default="float32", type=str, help='precision')
    parser.add_argument('--channels_last', default=1, type=int, help='Use NHWC or not')
    parser.add_argument('--jit', action='store_true', default=False, help='enable JIT')
    parser.add_argument('--profile', action='store_true', default=False, help='collect timeline')
    parser.add_argument('--num_iter', default=200, type=int, help='test iterations')
    parser.add_argument('--num_warmup', default=20, type=int, help='test warmup')
    parser.add_argument('--device', default='cpu', type=str, help='cpu, cuda or xpu')
    parser.add_argument('--nv_fuser', action='store_true', default=False, help='enable nv fuser')
    # input
    parser.add_argument('--idx', default=2000, type=int, help='dataset idx, for generate img, question, answer')
    parser.add_argument('--questions_path', default="/home2/pytorch-broad-models/VisualBert/v2_OpenEnded_mscoco_val2014_questions.json",
            type=str)
    parser.add_argument('--answers_path', default="/home2/pytorch-broad-models/VisualBert/v2_mscoco_val2014_annotations.json",
            type=str)
    parser.add_argument('--img_dir', default="/home2/pytorch-broad-models/COCO2014/val2014", type=str)
    args = parser.parse_args()
    print(args)
    return args

def trace_handler(p):
    output = p.key_averages().table(sort_by="self_cpu_time_total")
    print(output)
    import pathlib
    timeline_dir = str(pathlib.Path.cwd()) + '/timeline/'
    if not os.path.exists(timeline_dir):
        try:
            os.makedirs(timeline_dir)
        except:
            pass
    timeline_file = timeline_dir + 'timeline-' + str(torch.backends.quantized.engine) + \
            '-' + str(p.step_num) + '-' + str(os.getpid()) + '.json'
    p.export_chrome_trace(timeline_file)

def inference(args, model, tokenizer, question_info, visual_embeds):
    # prepare input
    question_inputs = [question_info['question']] * args.batch_size
    tokens = tokenizer(question_inputs, padding='max_length', max_length=50)
    input_ids = torch.tensor(tokens["input_ids"]).to(args.device)
    attention_mask = torch.tensor(tokens["attention_mask"]).to(args.device)
    token_type_ids = torch.tensor(tokens["token_type_ids"]).to(args.device)

    visual_embeds = torch.stack(visual_embeds).to(args.device)
    visual_attention_mask = torch.ones(visual_embeds.shape[:-1], dtype=torch.long).to(args.device)
    visual_token_type_ids = torch.ones(visual_embeds.shape[:-1], dtype=torch.long).to(args.device)

    model = model.to(args.device)

    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)
        print("---- use NHWC format")
    if args.nv_fuser:
       fuser_mode = "fuser2"
    else:
       fuser_mode = "none"
    print("---- fuser mode:", fuser_mode)
    if args.jit:
        try:
            model = torch.jit.trace(model, (input_ids, attention_mask, token_type_ids,
                None, None, None,
                visual_embeds, visual_attention_mask, visual_token_type_ids), check_trace=False, strict=False)
            print("---- JIT trace enable.")
        except (RuntimeError, TypeError) as e:
            print("---- JIT trace disable.")
            print("failed to use PyTorch jit mode due to: ", e)

    # forward
    total_time = 0.0
    total_sample = 0

    if args.profile and args.device == "xpu":
        for i in range(args.num_iter + args.num_warmup):
            elapsed = time.time()
            with torch.autograd.profiler_legacy.profile(enabled=args.profile, use_xpu=True, record_shapes=False) as prof:
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, visual_embeds=visual_embeds, visual_attention_mask=visual_attention_mask, visual_token_type_ids=visual_token_type_ids)
            torch.xpu.synchronize()
            elapsed = time.time() - elapsed
            print("Iteration: {}, inference time: {} sec.".format(i, elapsed), flush=True)
            if i >= args.num_warmup:
                total_sample += args.batch_size
                total_time += elapsed
            if args.profile and i == int((args.num_iter + args.num_warmup)/2):
                import pathlib
                timeline_dir = str(pathlib.Path.cwd()) + '/timeline/'
                if not os.path.exists(timeline_dir):
                    try:
                        os.makedirs(timeline_dir)
                    except:
                        pass
                torch.save(prof.key_averages().table(sort_by="self_xpu_time_total"),
                    timeline_dir+'profile.pt')
                torch.save(prof.key_averages(group_by_input_shape=True).table(),
                    timeline_dir+'profile_detail.pt')
    elif args.profile and args.device == "cuda":
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            record_shapes=True,
            schedule=torch.profiler.schedule(
                wait=int((args.num_iter + args.num_warmup)/2),
                warmup=2,
                active=1,
            ),
            on_trace_ready=trace_handler,
        ) as p:
            for i in range(args.num_iter + args.num_warmup):
                elapsed = time.time()
                with torch.jit.fuser(fuser_mode):
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, visual_embeds=visual_embeds, visual_attention_mask=visual_attention_mask, visual_token_type_ids=visual_token_type_ids)
                torch.cuda.synchronize()
                elapsed = time.time() - elapsed
                p.step()
                print("Iteration: {}, inference time: {} sec.".format(i, elapsed), flush=True)
                if i >= args.num_warmup:
                    total_sample += args.batch_size
                    total_time += elapsed
    elif args.profile and args.device == "cpu":
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU],
            record_shapes=True,
            schedule=torch.profiler.schedule(
                wait=int((args.num_iter + args.num_warmup)/2),
                warmup=2,
                active=1,
            ),
            on_trace_ready=trace_handler,
        ) as p:
            for i in range(args.num_iter + args.num_warmup):
                elapsed = time.time()
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, visual_embeds=visual_embeds, visual_attention_mask=visual_attention_mask, visual_token_type_ids=visual_token_type_ids)
                elapsed = time.time() - elapsed
                p.step()
                print("Iteration: {}, inference time: {} sec.".format(i, elapsed), flush=True)
                if i >= args.num_warmup:
                    total_sample += args.batch_size
                    total_time += elapsed
    elif not args.profile and args.device == "cuda":
        for i in range(args.num_iter + args.num_warmup):
            elapsed = time.time()
            with torch.jit.fuser(fuser_mode):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, visual_embeds=visual_embeds, visual_attention_mask=visual_attention_mask, visual_token_type_ids=visual_token_type_ids)
            torch.cuda.synchronize()
            elapsed = time.time() - elapsed
            print("Iteration: {}, inference time: {} sec.".format(i, elapsed), flush=True)
            if i >= args.num_warmup:
                total_sample += args.batch_size
                total_time += elapsed
    else:
        for i in range(args.num_iter + args.num_warmup):
            elapsed = time.time()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, visual_embeds=visual_embeds, visual_attention_mask=visual_attention_mask, visual_token_type_ids=visual_token_type_ids)
            if args.device == "xpu":
                torch.xpu.synchronize()
            elapsed = time.time() - elapsed
            print("Iteration: {}, inference time: {} sec.".format(i, elapsed), flush=True)
            if i >= args.num_warmup:
                total_sample += args.batch_size
                total_time += elapsed

    latency = total_time / total_sample * 1000
    throughput = total_sample / total_time
    print("inference Latency: {} ms".format(latency))
    print("inference Throughput: {} samples/s".format(throughput))

def main():
    args = parse_args()

    if args.device == "xpu":
        import intel_extension_for_pytorch
    elif args.device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False

    # load config
    cfg_path = "COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"
    cfg = load_config_and_model_weights(cfg_path)

    # load trained model
    model = get_model(cfg)

    # prepare Question and Answer
    with open(args.questions_path) as f:
        questions = json.load(f)
    question_info = questions["questions"][args.idx]
    image_id = question_info['image_id']
    # prepare img
    img_name = os.path.join(args.img_dir, f"COCO_val2014_{image_id:012d}.jpg")
    img = plt.imread(img_name)
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    images, batched_inputs = prepare_image_inputs(cfg, [img_bgr] * args.batch_size, model)

    # get features
    features = get_features(model, images)

    # get proposals
    proposals = get_proposals(model, images, features)

    # get box features
    box_features, features_list = get_box_features(args, model, features, proposals)

    # get prediction logits and boxes
    pred_class_logits, pred_proposal_deltas = get_prediction_logits(model, features_list, proposals)

    # get FastRCNN scores and boxes
    boxes, scores, image_shapes = get_box_scores(cfg, pred_class_logits, pred_proposal_deltas, proposals)

    # Rescale the boxes to original image size
    output_boxes = [get_output_boxes(boxes[i], batched_inputs[i], proposals[i].image_size) for i in range(len(proposals))]

    # Select the Boxes using NMS
    temp = [select_boxes(cfg, output_boxes[i], scores[i]) for i in range(len(scores))]
    keep_boxes, max_conf = [],[]
    for keep_box, mx_conf in temp:
        keep_boxes.append(keep_box)
        max_conf.append(mx_conf)

    # Limit the total number of boxes
    MIN_BOXES=10
    MAX_BOXES=100
    keep_boxes = [filter_boxes(keep_box, mx_conf, MIN_BOXES, MAX_BOXES) for keep_box, mx_conf in zip(keep_boxes, max_conf)]

    # Get the visual embeddings
    visual_embeds = [get_visual_embeds(box_feature, keep_box) for box_feature, keep_box in zip(box_features, keep_boxes)]

    # Using the embeddings with VisualBert
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = VisualBertForPreTraining.from_pretrained('uclanlp/visualbert-nlvr2-coco-pre')
    with torch.no_grad():
        model.eval()
        if args.device == "xpu":
            datatype = torch.float16 if args.precision == "float16" else torch.bfloat16 if args.precision == "bfloat16" else torch.float
            model = torch.xpu.optimize(model=model, dtype=datatype)
        if args.precision == "float16" and args.device == "cuda":
            print("---- Use autocast fp16 cuda")
            with torch.cuda.amp.autocast(enabled=True, dtype=torch.float16):
                inference(args, model, tokenizer, question_info, visual_embeds)
        elif args.precision == "float16" and args.device == "xpu":
            print("---- Use autocast fp16 xpu")
            with torch.xpu.amp.autocast(enabled=True, dtype=torch.float16, cache_enabled=True):
                inference(args, model, tokenizer, question_info, visual_embeds)
        elif args.precision == "bfloat16" and args.device == "cpu":
            print("---- Use autocast bf16 cpu")
            with torch.cpu.amp.autocast(enabled=True, dtype=torch.bfloat16):
                inference(args, model, tokenizer, question_info, visual_embeds)
        elif args.precision == "bfloat16" and args.device == "xpu":
            print("---- Use autocast bf16 xpu")
            with torch.xpu.amp.autocast(dtype=torch.bfloat16):
                inference(args, model, tokenizer, question_info, visual_embeds)
        else:
            print("---- no autocast")
            inference(args, model, tokenizer, question_info, visual_embeds)


if __name__ == "__main__":
    main()


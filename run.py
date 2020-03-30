import os
import cv2
import visdom
import numpy as np

import traceback as tb
import code

import _init_paths
# from datasets.pascal_voc_cpu import pascal_voc
from datasets.factory import get_imdb

def plot_box(img, bbox, label=None):
    cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)
    if label is not None:
        cv2.putText(img, '%s' % (label), (bbox[0], bbox[1] + 15),
                    cv2.FONT_HERSHEY_PLAIN,
                    1.0, (255, 255, 255),
                    thickness=1)

def plot_boxes(img, boxes, labels):
    if labels is None:
        labels = [None] * len(boxes)
    [plot_box(img, *args) for args in zip(boxes, labels)]
    # cv2.imshow('Window', img)
    # cv2.waitKey(0)
    return img

def plot_gt_boxes(index):
    # db = pascal_voc('trainval', '2007', './data/VOCdevkit')
    db = get_imdb('voc_2007_trainval')
    img_path = db.image_path_at(index)
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    gt_roidb = db.gt_roidb()
    bboxes = gt_roidb[index]['boxes']
    labels = [db._classes[x-1] for x in gt_roidb[index]['gt_classes']]
    img = plot_boxes(img, bboxes, labels)
    viz_server(img)

def plot_proposals(index):
    # db = pascal_voc('trainval', '2007', './data/VOCdevkit')
    db = get_imdb('voc_2007_trainval')
    img_path = db.image_path_at(index)
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    gt_roidb = db.gt_roidb()
    proposals_roidb = db._load_selective_search_roidb(gt_roidb)
    tb.print_stack();namespace = globals().copy();namespace.update(locals());code.interact(local=namespace)
    bboxes = proposals_roidb[index]['boxes'][:10]
    img = plot_boxes(img, bboxes, None)
    viz_server(img)

def viz_server(img):
    vis = visdom.Visdom(server='0.0.0.0',port='8080')
    # tb.print_stack();namespace = globals().copy();namespace.update(locals());code.interact(local=namespace)
    vis.image(img.transpose((2,0,1)))

if __name__ == '__main__':
    index = 2020
    plot_gt_boxes(index=index)
    plot_proposals(index=index)

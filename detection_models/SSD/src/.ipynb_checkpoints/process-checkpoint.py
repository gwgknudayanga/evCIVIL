"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""
import numpy as np
from tqdm.autonotebook import tqdm
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
#from apex import amp
import os
import json

def train(model, train_loader, epoch, writer, criterion, optimizer, scheduler, is_amp,current_val_mAP):
    model.train()
    num_iter_per_epoch = len(train_loader)
    progress_bar = tqdm(train_loader)
    scheduler.step(current_val_mAP)
    for i, (img, _, _, gloc, glabel) in enumerate(progress_bar):
        if torch.cuda.is_available():
            img = img.cuda()
            gloc = gloc.cuda()
            glabel = glabel.cuda()

        #Ground truyth vs predicted label
        ploc, plabel = model(img)
        ploc, plabel = ploc.float(), plabel.float()
        #print("AAAAAAAAAAAAAAAAAAAAAAAA ",gloc.shape," ",glabel.shape)
        #print("BBBBBBBBBBBBBBBBBBBBBBBBBB ",ploc.shape, "  ",plabel.shape)
        gloc = gloc.transpose(1, 2).contiguous()
        loss = criterion(ploc, plabel, gloc, glabel)
        #global location of the array 

        progress_bar.set_description("Epoch: {}. Loss: {:.5f}".format(epoch + 1, loss.item()))

        writer.add_scalar("Train/Loss", loss.item(), epoch * num_iter_per_epoch + i)

        if is_amp:
            with amp.scale_loss(loss, optimizer) as scale_loss:
                scale_loss.backward()
        else:
            loss.backward()
        optimizer.step()
        optimizer.zero_grad()


def evaluate(model, test_loader, epoch, writer, encoder, nms_threshold,conf_thres, dataset_type = "defect",val_gt_ann_path = ""):
    model.eval()
    detections = []
    if dataset_type == "defect":
        category_ids = test_loader.dataset.getCatIds()
    elif dataset_type == "coco":
        category_ids = test_loader.dataset.coco.getCatIds()

    for nbatch, (img, img_id, img_size, _, _) in enumerate(test_loader):
        print("Parsing batch: {}/{}".format(nbatch, len(test_loader)), end="\r")
        if torch.cuda.is_available():
            img = img.cuda()
        with torch.no_grad():
            # Get predictions
            ploc, plabel = model(img)
            ploc, plabel = ploc.float(), plabel.float()

            for idx in range(ploc.shape[0]):
                ploc_i = ploc[idx, :, :].unsqueeze(0)
                plabel_i = plabel[idx, :, :].unsqueeze(0)
                try:
                    result = encoder.decode_batch(ploc_i, plabel_i, nms_threshold, 200,conf_thres = conf_thres)[0]
                except:
                    print("No object detected in idx: {}".format(idx))
                    continue

                height, width = img_size[idx]
                loc, label, prob = [r.cpu().numpy() for r in result]
                for loc_, label_, prob_ in zip(loc, label, prob):
                    image_id = img_id[idx]
                    #print("category ids ",category_ids)
                    #print("label...........",label)
                    category_id = category_ids[label_ - 1]

                    tl_x = loc_[0] * width
                    tl_y = loc_[1] * height
                    bb_width = (loc_[2] - loc_[0]) * width
                    bb_height = (loc_[3] - loc_[1]) * height
                    bbox = [tl_x,tl_y,bb_width,bb_height]

                    score = prob_

                    if dataset_type == "defect":

                        pred_data = {
                                "image_id": image_id,
                                "category_id": category_id,
                                "bbox": bbox,
                                "score": score.astype(float)
                            }
                        detections.append(pred_data)
                    else:
                        detections.append([image_id, tl_x, tl_y,bb_width ,
                                       bb_height, score,
                                       category_id])


    if dataset_type == "defect":

        pred_json = os.path.join(val_gt_ann_path.rsplit("/",1)[0],"predict.json")
        with open(pred_json, 'w') as f:
            json.dump(detections, f)

        anno = COCO(val_gt_ann_path)
        pred = anno.loadRes(pred_json)
        coco_eval = COCOeval(anno, pred, 'bbox')
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_p = coco_eval.eval['precision']
        coco_p_all = coco_p[:, :, :, 0, 2]
        map = np.mean(coco_p_all[coco_p_all>-1])

        coco_p_iou50 = coco_p[0, :, :, 0, 2]
        map50 = np.mean(coco_p_iou50[coco_p_iou50>-1])
        mp = np.array([np.mean(coco_p_iou50[ii][coco_p_iou50[ii]>-1]) for ii in range(coco_p_iou50.shape[0])])
        mr = np.linspace(.0, 1.00, int(np.round((1.00 - .0) / .01)) + 1, endpoint=True)
        mf1 = 2 * mp * mr / (mp + mr + 1e-16)
        i = mf1.argmax()  # max F1 index

        print("Allll precison ",mp[i], " "," recall ",mr[i], " f1 ",mf1[i], " map50 ",map50, " ",map)

        catIds = anno.getCatIds()
        print("ccccccc ",catIds)
        for nc_i in catIds:
                    print("YYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYY ")
                    coco_p_c = coco_p[:, :, nc_i, 0, 2]
                    map = np.mean(coco_p_c[coco_p_c>-1])

                    coco_p_c_iou50 = coco_p[0, :, nc_i, 0, 2]
                    map50 = np.mean(coco_p_c_iou50[coco_p_c_iou50>-1])
                    p = coco_p_c_iou50
                    r = np.linspace(.0, 1.00, int(np.round((1.00 - .0) / .01)) + 1, endpoint=True)
                    f1 = 2 * p * r / (p + r + 1e-16)
                    i = f1.argmax()
                    print("classwiseeeeeeeeeeeeee ", nc_i, " ",p[i], " ",r[i], " ",f1[i], " ",map50, " ",map)
        coco_eval.summarize()
    else:
        detections = np.array(detections, dtype=np.float32)
        coco_eval = COCOeval(test_loader.dataset.coco, test_loader.dataset.coco.loadRes(detections), iouType="bbox")
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

    writer.add_scalar("Test/mAP", coco_eval.stats[0], epoch)
    writer.add_scalar("Test/map0.5", coco_eval.stats[1],epoch)

    return coco_eval.stats[1]
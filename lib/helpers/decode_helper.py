import numpy as np
import torch
import torch.nn as nn
from lib.datasets.utils import class2angle
from utils import box_ops


def decode_detections(dets, info, calibs, cls_mean_size, threshold):
    '''
    NOTE: THIS IS A NUMPY FUNCTION
    input: dets, numpy array, shape in [batch x max_dets x dim]
    input: img_info, dict, necessary information of input images
    input: calibs, corresponding calibs for the input batch
    output:
    '''
    results = {}
    
    

    for i in range(dets.shape[0]):  # batch
        preds = []
        score_all=[]
        # clustering_features = []                    #extra
        # mean_score = np.mean(dets[i, :, 1])
        # std = np.std(dets[i, :, 1])

        # if std > 0.1:
        #     new_threshold = mean_score + (std / 8)
        # else:
        #     new_threshold = threshold

###########################################################################################################################    
        # class MLP(nn.Module):
        #     def __init__(self, input_dim=59):
        #         super().__init__()
        #         self.model = nn.Sequential(
        #             nn.Linear(input_dim, 128),
        #             nn.ReLU(),
        #             nn.Linear(128, 64),
        #             nn.ReLU(),
        #             nn.Linear(64, 32),
        #             nn.ReLU(),
        #             nn.Linear(32, 1),
        #             nn.Sigmoid()
        #         )

        #     def forward(self, x):
        #         return self.model(x)


        # # ======================================================
        # # تابع مستقل برای پیش‌بینی آستانه
        # # ======================================================
        # def predict_threshold(conf_scores, model_path="/kaggle/working/mlp_threshold_model_best.pth"):
        #     """
        #     conf_scores: آرایه‌ای از confidenceهای پیش‌بینی (مثلاً ۵۰ عدد)
        #     خروجی: مقدار آستانه پیش‌بینی‌شده بین ۰ و ۱
        #     """
        #     conf_scores = np.array(conf_scores, dtype=np.float32)
        #     if len(conf_scores) < 50:
        #         conf_scores = np.pad(conf_scores, (0, 50 - len(conf_scores)), mode="constant")
        #     else:
        #         conf_scores = conf_scores[:50]

        #     # استخراج ویژگی‌ها (۹ ویژگی آماری)
        #     mean_all = np.mean(conf_scores)
        #     std_all = np.std(conf_scores)
        #     max_val = np.max(conf_scores)
        #     below_02 = conf_scores[conf_scores < 0.2]
        #     mean_sub_0_2 = np.mean(below_02) if len(below_02) > 0 else 0
        #     std_sub_0_2 = np.std(below_02) if len(below_02) > 0 else 0
        #     mad_sub_0_2 = np.median(np.abs(below_02 - np.median(below_02))) if len(below_02) > 0 else 0
        #     count_gt_0_01 = np.mean(conf_scores > 0.01)
        #     skewness = ((conf_scores - mean_all)**3).mean() / (std_all**3 + 1e-6)
        #     kurtosis = ((conf_scores - mean_all)**4).mean() / (std_all**4 + 1e-6)

        #     stats_features = [
        #         mean_all, std_all, mean_sub_0_2, std_sub_0_2,
        #         max_val, count_gt_0_01, skewness, kurtosis, mad_sub_0_2
        #     ]

        #     features = np.concatenate([conf_scores, stats_features]).reshape(1, -1)

        #     # ساخت مدل و بارگذاری وزن‌ها
        #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #     model = MLP(input_dim=59).to(device)
        #     model.load_state_dict(torch.load(model_path, map_location=device))
        #     model.eval()

        #     # اجرای پیش‌بینی
        #     with torch.no_grad():
        #         x_tensor = torch.tensor(features, dtype=torch.float32).to(device)
        #         threshold = model(x_tensor).item()

        #     return threshold


        # # ======================================================
        # # 🔹 استفاده در هر جای کد
        # # ======================================================
        # # مثال ساده (می‌تونی حذفش کنی)
        # conf_scores = np.sort(dets[i, :, 1])[::-1]
        # new_threshold = predict_threshold(conf_scores, "/kaggle/working/mlp_threshold_model_best.pth")
        # if new_threshold > 0.2:
        #     new_threshold=0.2
###########################################################################################################################    
        # class DeepSets(nn.Module):
        #     def __init__(self, input_dim=1, hidden_dim=128, embed_dim=256):
        #         super().__init__()

        #         self.phi = nn.Sequential(
        #             nn.Linear(input_dim, hidden_dim),
        #             nn.ReLU(),
        #             nn.Linear(hidden_dim, embed_dim),
        #             nn.ReLU(),
        #             nn.BatchNorm1d(50)
        #         )

        #         self.rho = nn.Sequential(
        #             nn.Linear(embed_dim, 256),
        #             nn.ReLU(),
        #             nn.Dropout(0.3),
        #             nn.Linear(256, 128),
        #             nn.ReLU(),
        #             nn.Linear(128, 64),
        #             nn.ReLU(),
        #             nn.Linear(64, 1),
        #             nn.Sigmoid()
        #         )

        #     def forward(self, x):
        #         x = x.unsqueeze(-1)   # [batch, 50, 1]
        #         h = self.phi(x)       # [batch, 50, embed_dim]
        #         h = h.mean(dim=1)     # [batch, embed_dim]
        #         out = self.rho(h)     # [batch, 1]
        #         return out

        # # ======================================================
        # # تابع مستقل برای پیش‌بینی آستانه
        # # ======================================================
        # def predict_threshold_deepsets(conf_scores, model_path="/kaggle/working/deepsets_threshold_model_best.pth"):
        #     """
        #     conf_scores: آرایه ۵۰تایی از مقادیر confidence (numpy یا list)
        #     خروجی: مقدار آستانه پیش‌بینی‌شده بین ۰ و ۱
        #     """

        #     # مرتب‌سازی نزولی و ساخت کپی برای جلوگیری از stride منفی
        #     conf_scores = np.array(conf_scores, dtype=np.float32)
        #     conf_scores = np.sort(conf_scores)[::-1].copy()  # ← مهم: copy()

        #     if len(conf_scores) < 50:
        #         conf_scores = np.pad(conf_scores, (0, 50 - len(conf_scores)), mode='constant')

        #     # تبدیل به تنسور، بدون هیچ دستکاری اضافی
        #     x_tensor = torch.tensor(conf_scores, dtype=torch.float32).unsqueeze(0)  # [1, 50]

        #     # ساخت مدل و لود وزن‌ها
        #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #     model = DeepSets().to(device)
        #     model.load_state_dict(torch.load(model_path, map_location=device))
        #     model.eval()
        #     x_tensor = x_tensor.to(device)

        #     # پیش‌بینی
        #     with torch.no_grad():
        #         threshold = model(x_tensor).item()

        #     return threshold


        # conf_scores = dets[i, :, 1]
        # new_threshold = predict_threshold_deepsets(conf_scores)
        # if new_threshold > 0.2:
        #     new_threshold=0.2
###########################################################################################################################
        for j in range(dets.shape[1]):  # max_dets
            cls_id = int(dets[i, j, 0])
            score = dets[i, j, 1]

            score_all.append(score)
            
            if score <= threshold:
                continue
                            
            # 2d bboxs decoding
            x = dets[i, j, 2] * info['img_size'][i][0]
            y = dets[i, j, 3] * info['img_size'][i][1]
            w = dets[i, j, 4] * info['img_size'][i][0]
            h = dets[i, j, 5] * info['img_size'][i][1]
            bbox = [x-w/2, y-h/2, x+w/2, y+h/2]

            # 3d bboxs decoding
            # depth decoding
            # max_depth = dets[i, :, 6].max()                 #extra
            depth = dets[i, j, 6]
            # depth_norm = depth / max_depth                  #extra
            # dimensions decoding
            dimensions = dets[i, j, 31:34]
            dimensions += cls_mean_size[int(cls_id)]

            # positions decoding
            x3d = dets[i, j, 34] * info['img_size'][i][0]
            y3d = dets[i, j, 35] * info['img_size'][i][1]
            # xs3d_cluster = dets[i, j, 34]                   #extra
            # ys3d_cluster = dets[i, j, 35]                   #extra 
            locations = calibs[i].img_to_rect(x3d, y3d, depth).reshape(-1)
            locations[1] += dimensions[0] / 2

            # heading angle decoding
            alpha = get_heading_angle(dets[i, j, 7:31]) 
            # alpha_sin = (np.sin(alpha) + 1) / 2                  #extra
            # alpha_cos = (np.cos(alpha) + 1) / 2                  #extra
            ry = calibs[i].alpha2ry(alpha, x)


            score = score * dets[i, j, -1]
            preds.append([cls_id, alpha] + bbox + dimensions.tolist() + locations.tolist() + [ry, score])
##########################################################################################################
            # # ذخیره فایل خروجی بدون گرد شدن نمره های اعتماد در مسیر مشخص فقط خط ضریب نمره انکامنت شده
            # import os
            # output_dir = '/kaggle/working/out'
            # os.makedirs(output_dir, exist_ok=True)
            # output_path = os.path.join(output_dir, '{:06d}.txt'.format(info['img_id'][i]))
            # with open(output_path, 'w') as f:
            #     for pred in preds:
            #         class_name = 'Car'
            #         f.write(f'{class_name} 0.0 0')
            #         for val in pred[1:-1]:
            #             f.write(f' {val:.2f}')
            #         score = pred[-1]
            #         f.write(f' {score}\n')
##########################################################################################################
            # features = [xs3d_cluster, ys3d_cluster, depth_norm, alpha_sin, alpha_cos]                    #extra
            # clustering_features.append(features)                                                         #extra
#####################################################
        # from nms_3d import nms_3d  
        # if len(preds) == 0:
        #     filtered_preds = []
        # else:
        #     nms_input = []
        #     index_map = []
        #     for idx, p in enumerate(preds):
        #         score = float(p[-1])
        #         dx, dy, dz = [float(x) for x in p[6:9]]
        #         x3d, y3d, z3d = [float(x) for x in p[9:12]]
        #         x_min = x3d - dx / 2
        #         x_max = x3d + dx / 2
        #         y_min = y3d - dy / 2
        #         y_max = y3d + dy / 2
        #         z_min = z3d - dz / 2
        #         z_max = z3d + dz / 2
        #         nms_input.append([score, x_min, y_min, z_min, x_max, y_max, z_max])
        #         index_map.append(idx)
        #     nms_np = np.array(nms_input, dtype=np.float32)
        #     if nms_np.shape[0] == 0:
        #         filtered_preds = []
        #     else:
        #         nms_tensor = torch.from_numpy(nms_np)  
        #         iou_threshold = 0.9
        #         filtered_boxes = nms_3d(nms_tensor, iou_threshold=iou_threshold)
        #         if isinstance(filtered_boxes, torch.Tensor):
        #             filtered_np = filtered_boxes.detach().cpu().numpy()
        #         else:
        #             filtered_np = np.array(filtered_boxes, dtype=np.float32)
        #         keep_indices = []
        #         for fb in filtered_np:  
        #             matches = np.all(np.isclose(nms_np, fb, atol=1e-5, rtol=1e-6), axis=1)
        #             idxs = np.where(matches)[0]
        #             if idxs.size > 0:
        #                 keep_indices.append(index_map[int(idxs[0])])
        #             else:
        #                 pass
        #         preds = [preds[i] for i in keep_indices]
#####################################################
        # filtered_preds = []
        # if len(clustering_features) >= 2:
        #     clustering_features = np.array(clustering_features) 
        #     from sklearn.cluster import DBSCAN                
        #     db = DBSCAN(eps=0.02, min_samples=2)
        #     cluster_labels = db.fit_predict(clustering_features)            
        #     if len(preds) > 0:
        #         preds_np = np.array(preds, dtype=object)  
        #         scores = np.array([p[-1] for p in preds])  
        #         unique_clusters = np.unique(cluster_labels)
        #         for cid in unique_clusters:
        #             idxs = np.where(cluster_labels == cid)[0]
        #             if len(idxs) == 0:
        #                 continue
        #             if cid == -1:
        #                 for idx in idxs:
        #                     filtered_preds.append(preds[idx])
        #             else:
        #                 best_idx = idxs[np.argmax(scores[idxs])]
        #                 filtered_preds.append(preds[best_idx])
    #####################################################
        # import hdbscan
        # db = hdbscan.HDBSCAN(min_cluster_size=2, min_samples=2)
        # cluster_labels = db.fit_predict(clustering_features)
        # print("Cluster labels for each detection:")
        # print(cluster_labels)
#####################################################      
        # score_all_np = np.array(score_all)
        # median_score = np.median(score_all_np)
        # mad_score = np.median(np.abs(score_all_np - median_score))
        # k = 1.0  # یا 1.4826 اگر بخوای شبیه انحراف معیار باشه
        # robust_value = median_score + k * mad_score
        # std = np.std(score_all_np)
        # print(f"image  {info['img_id'][i]} _ mean => {np.mean(score_all):0.3f} _ mad => {robust_value:0.3f} _ STD => {std:0.3f}\n")          
#####################################################
        # print(f"image  {info['img_id'][i]}\n" , sorted(score_all, reverse=True))          
        # print(f"image  {info['img_id'][i]}\n" , np.mean(score_all))     
#####################################################
        # import json, os
        # save_path = "/kaggle/working/conf_data.json"
        # if os.path.exists(save_path):
        #     with open(save_path, "r") as f:
        #         data = json.load(f)
        # else:
        #     data = {}
        # img_id = str(info['img_id'][i])
        # scores = sorted(score_all, reverse=True)
        # scores = [float(x) for x in scores]
        # data[img_id] = {"scores": scores}
        # with open(save_path, "w") as f:
        #     json.dump(data, f)
#####################################################
        # results[info['img_id'][i]] = filtered_preds
        results[info['img_id'][i]] = preds
    return results


def extract_dets_from_outputs(outputs, K=50, topk=50):
    # get src outputs

    # b, q, c
    out_logits = outputs['pred_logits']
    out_bbox = outputs['pred_boxes']

    prob = out_logits.sigmoid()
    topk_values, topk_indexes = torch.topk(prob.view(out_logits.shape[0], -1), topk, dim=1)

    # final scores
    scores = topk_values
    # final indexes
    topk_boxes = (topk_indexes // out_logits.shape[2]).unsqueeze(-1)
    # final labels
    labels = topk_indexes % out_logits.shape[2]
    
    heading = outputs['pred_angle']
    size_3d = outputs['pred_3d_dim']
    depth = outputs['pred_depth'][:, :, 0: 1]
    sigma = outputs['pred_depth'][:, :, 1: 2]
    sigma = torch.exp(-sigma)


    # decode
    boxes = torch.gather(out_bbox, 1, topk_boxes.repeat(1, 1, 6))  # b, q', 6

    xs3d = boxes[:, :, 0: 1] 
    ys3d = boxes[:, :, 1: 2] 

    heading = torch.gather(heading, 1, topk_boxes.repeat(1, 1, 24))
    depth = torch.gather(depth, 1, topk_boxes)
    sigma = torch.gather(sigma, 1, topk_boxes) 
    size_3d = torch.gather(size_3d, 1, topk_boxes.repeat(1, 1, 3))

    corner_2d = box_ops.box_cxcylrtb_to_xyxy(boxes)

    xywh_2d = box_ops.box_xyxy_to_cxcywh(corner_2d)
    size_2d = xywh_2d[:, :, 2: 4]
    
    xs2d = xywh_2d[:, :, 0: 1]
    ys2d = xywh_2d[:, :, 1: 2]

    batch = out_logits.shape[0]
    labels = labels.view(batch, -1, 1)
    scores = scores.view(batch, -1, 1)
    xs2d = xs2d.view(batch, -1, 1)
    ys2d = ys2d.view(batch, -1, 1)
    xs3d = xs3d.view(batch, -1, 1)
    ys3d = ys3d.view(batch, -1, 1)

    detections = torch.cat([labels, scores, xs2d, ys2d, size_2d, depth, heading, size_3d, xs3d, ys3d, sigma], dim=2)

    return detections


############### auxiliary function ############


def _nms(heatmap, kernel=3):
    padding = (kernel - 1) // 2
    heatmapmax = nn.functional.max_pool2d(heatmap, (kernel, kernel), stride=1, padding=padding)
    keep = (heatmapmax == heatmap).float()
    return heatmap * keep


def _topk(heatmap, K=50):
    batch, cat, height, width = heatmap.size()

    # batch * cls_ids * 50
    topk_scores, topk_inds = torch.topk(heatmap.view(batch, cat, -1), K)

    topk_inds = topk_inds % (height * width)
    topk_ys = (topk_inds / width).int().float()
    topk_xs = (topk_inds % width).int().float()

    # batch * cls_ids * 50
    topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), K)
    topk_cls_ids = (topk_ind / K).int()
    topk_inds = _gather_feat(topk_inds.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_ys = _gather_feat(topk_ys.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_xs = _gather_feat(topk_xs.view(batch, -1, 1), topk_ind).view(batch, K)

    return topk_score, topk_inds, topk_cls_ids, topk_xs, topk_ys


def _gather_feat(feat, ind, mask=None):
    '''
    Args:
        feat: tensor shaped in B * (H*W) * C
        ind:  tensor shaped in B * K (default: 50)
        mask: tensor shaped in B * K (default: 50)

    Returns: tensor shaped in B * K or B * sum(mask)
    '''
    dim  = feat.size(2)  # get channel dim
    ind  = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)  # B*len(ind) --> B*len(ind)*1 --> B*len(ind)*C
    feat = feat.gather(1, ind)  # B*(HW)*C ---> B*K*C
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)  # B*50 ---> B*K*1 --> B*K*C
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat


def _transpose_and_gather_feat(feat, ind):
    '''
    Args:
        feat: feature maps shaped in B * C * H * W
        ind: indices tensor shaped in B * K
    Returns:
    '''
    feat = feat.permute(0, 2, 3, 1).contiguous()   # B * C * H * W ---> B * H * W * C
    feat = feat.view(feat.size(0), -1, feat.size(3))   # B * H * W * C ---> B * (H*W) * C
    feat = _gather_feat(feat, ind)     # B * len(ind) * C
    return feat


def get_heading_angle(heading):
    heading_bin, heading_res = heading[0:12], heading[12:24]
    cls = np.argmax(heading_bin)
    res = heading_res[cls]
    return class2angle(cls, res, to_label_format=True)

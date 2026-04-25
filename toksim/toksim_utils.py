import math
import torch
from transformers import AutoImageProcessor, AutoModel
import torch.nn.functional as F

def compute_token_score_per_video(pred_vid, gt_vid, mask_vid, processor=None, model=None, device="cuda"):
    FRAMES,_,_,_ = pred_vid.shape
    token_scores = []
    
    token_scores_all_metrics = {'sim_p0p1': [],
                   'sim_bg0_mean': [],
                   'sim_p0g0': []}
    
    for f_idx in range(FRAMES-1):
        pred_frame_0 = pred_vid[f_idx]
        pred_frame_1 = pred_vid[f_idx+1]
        gt_frame_0 = gt_vid[f_idx]
        mask_frame_0 = mask_vid[f_idx].squeeze(-1)
        mask_frame_1 = mask_vid[f_idx+1].squeeze(-1)

        mask_frame_union_fg = mask_frame_0.bool() | mask_frame_1.bool()

        bbox = get_roi_coor(mask_frame_union_fg, pad=24)
        w0, h0, w1, h1 = bbox

        pred_crop_0 = pred_frame_0[h0:h1, w0:w1, :]
        pred_crop_1 = pred_frame_1[h0:h1, w0:w1, :]
        gt_crop_0 = gt_frame_0[h0:h1, w0:w1, :]
        mask_crop_union = mask_frame_union_fg[h0:h1, w0:w1]
        mask_crop_0 = mask_frame_0[h0:h1, w0:w1]
        mask_crop_1 = mask_frame_1[h0:h1, w0:w1]

        tok_pred_0 = dinov3_tokens(pred_crop_0, model, processor, device=device)
        tok_pred_1 = dinov3_tokens(pred_crop_1, model, processor, device=device)
        tok_gt_0  = dinov3_tokens(gt_crop_0, model, processor, device=device)

        token_score, all_metrics = token_score_per_frame(tok_pred_0, tok_pred_1, tok_gt_0, \
                                                         mask_crop_union, mask_crop_0, mask_crop_1)
        token_scores.append(token_score)
        for k in all_metrics.keys():
            token_scores_all_metrics[k].append(all_metrics[k])

    mean_token_score = sum(token_scores) / len(token_scores)
    for k in token_scores_all_metrics.keys():
        token_scores_all_metrics[k] = sum(token_scores_all_metrics[k]) / len(token_scores_all_metrics[k])
    return mean_token_score, token_scores_all_metrics

def token_score_per_frame(tok_pred_0, tok_pred_1, tok_gt_0, mask_crop_union, mask_crop_0, mask_crop_1):
    mask_crop_union = mask_crop_union.to(tok_pred_0.device)
    mask_crop_0 = mask_crop_0.to(tok_pred_0.device)
    mask_crop_1 = mask_crop_1.to(tok_pred_0.device)

    Ht, Wt, D = tok_pred_0.shape
    mask_union = mask_to_tokens(mask_crop_union, (Ht, Wt))
    mask_0 = mask_to_tokens(mask_crop_0, (Ht, Wt)) 
    mask_1 = mask_to_tokens(mask_crop_1, (Ht, Wt))

    P0 = tok_pred_0.reshape(-1, D)
    P1 = tok_pred_1.reshape(-1, D)
    G0 = tok_gt_0.reshape(-1, D)

    U  = mask_union.view(-1)
    M0 = mask_0.view(-1)
    M1 = mask_1.view(-1)

    idx_fgU = U.nonzero(as_tuple=True)[0]
    idx_bgU = (~U).nonzero(as_tuple=True)[0]
    idx_fg0 = M0.nonzero(as_tuple=True)[0]
    
    p0_fg = P0.index_select(0, idx_fgU)        # [n_fg, D]
    p1_fg = P1.index_select(0, idx_fgU)          # [n_fg, D]
    p0_bg = P0.index_select(0, idx_bgU)          # [n_bg, D]
    
    # fg0 <-> fg1 sim
    sim_p0p1 = (p0_fg * p1_fg).sum(dim=-1)

    # fg0 <-> bg0 sim
    sim_bg0 = p0_fg @ p0_bg.t()                              # [n_fg, n_bg0]
    sim_bg0_mean = sim_bg0.mean(dim=1)

    sim_fg0_onlyM0 = (P0.index_select(0, idx_fg0) * G0.index_select(0, idx_fg0)).sum(-1)
    repl_mean = sim_fg0_onlyM0.mean()

    p0_fgU = P0.index_select(0, idx_fgU)     # [n_fgU, D]
    g0_fgU = G0.index_select(0, idx_fgU)     # [n_fgU, D]
    sim_p0g0 = (p0_fgU * g0_fgU).sum(-1)      # [n_fgU]
    is_in_M0_for_fgU = M0.index_select(0, idx_fgU)
    to_fill = ~is_in_M0_for_fgU
    sim_p0g0[to_fill] = repl_mean

    # per_token = sim_p0p1 * sim_bg0_mean * sim_bg1_mean * (1-sim_p0g0)
    per_token = sim_p0p1 * sim_bg0_mean * (1-sim_p0g0)
    token_score = per_token.mean().item()

    all_metrics = {'sim_p0p1': sim_p0p1.mean().item(),
                   'sim_bg0_mean': sim_bg0_mean.mean().item(),
                #    'sim_bg1_mean': sim_bg1_mean.mean().item(),
                   'sim_p0g0': sim_p0g0.mean().item()}
    
    return token_score, all_metrics

def mask_to_tokens(mask_hw, tok_hw):
    m = mask_hw.to(torch.float32)[None, None]                # [1,1,H,W]
    m = F.interpolate(m, size=tok_hw, mode="nearest")        # [1,1,Ht,Wt]
    return m[0, 0].bool()                                    # [Ht,Wt]

def bbox_from_mask(mask):
    nonzero = mask.nonzero(as_tuple=False)
    w0 = int(nonzero[:, 1].min())
    h0 = int(nonzero[:, 0].min())
    w1 = int(nonzero[:, 1].max())
    h1 = int(nonzero[:, 0].max())
    return w0, h0, w1, h1

def get_roi_coor(mask, pad=24):
    H,W = mask.shape
    x0, y0, x1, y1 = bbox_from_mask(mask)
    x0 = max(0, x0 - pad)
    y0 = max(0, y0 - pad)
    x1 = min(W - 1, x1 + pad)
    y1 = min(H - 1, y1 + pad)
    return x0, y0, x1, y1

######################### dino ########################
def get_models(device="cuda"):
    MODEL_ID = "facebook/dinov3-vits16-pretrain-lvd1689m"
    processor = AutoImageProcessor.from_pretrained(MODEL_ID)
    model = AutoModel.from_pretrained(MODEL_ID).to(device)
    return processor, model

@torch.inference_mode()
def dinov3_tokens(img, model, processor, device="cuda"):
    batch = processor(images=img, return_tensors="pt").to(device)
    out = model(**batch)
    x = out.last_hidden_state[:, 1 + model.config.num_register_tokens:, :]
    n = x.shape[1]; hw = int(math.sqrt(n))
    grid = x.reshape(1, hw, hw, -1).squeeze(0)
    return F.normalize(grid, dim=-1)

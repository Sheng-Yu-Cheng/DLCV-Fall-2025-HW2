import os
import json
import argparse
import numpy as np
import torch
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from transformers import CLIPProcessor, CLIPModel

def calculate_iou(pred_mask, gt_mask):
    """
    Calculate Intersection over Union (IoU) between predicted mask and ground truth mask
    
    Args:
        pred_mask: Predicted binary mask (H, W)
        gt_mask: Ground truth binary mask (H, W)
    
    Returns:
        iou: IoU score
    """
    # Convert to binary masks
    pred_mask = (pred_mask > 0.5).astype(np.uint8)
    gt_mask = (gt_mask > 0.5).astype(np.uint8)
    
    # Calculate intersection and union
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    
    # Avoid division by zero
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    
    return intersection / union

def load_clip_model(device):
    """Load CLIP model and processor (HuggingFace transformers)."""
    if CLIPModel is None or CLIPProcessor is None:
        raise ImportError(
            "transformers' CLIP is not installed. Install with: pip install transformers"
        )
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model.eval()
    return model, processor

@torch.no_grad()
def compute_clip_score(model, processor, image_np, text, device):
    """Compute CLIP text-image similarity in [0,1] via cosine on embeddings."""
    # Prepare PIL image
    pil_img = Image.fromarray(image_np.astype(np.uint8)) if isinstance(image_np, np.ndarray) else image_np
    # Use model's embedding heads for cosine similarity
    inputs = processor(text=[text], images=[pil_img], return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    image_emb = model.get_image_features(pixel_values=inputs["pixel_values"])  # [1, d]
    text_emb = model.get_text_features(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])  # [1, d]
    # L2-normalize
    image_emb = image_emb / image_emb.norm(dim=-1, keepdim=True)
    text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)
    cosine_sim = (image_emb @ text_emb.T).squeeze().item()
    score_01 = (cosine_sim + 1.0) * 0.5
    return float(score_01)

def stream_jsonl(json_path):
    """Yield one dict per line from an NDJSON (jsonl) file."""
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"json_path not found: {json_path}")
    with open(json_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                # skip bad lines
                continue

def load_sam2_model(device):
    """Load SAM2 model and predictor"""
    # Set up device
    if device.type == "cuda":
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
    
    # Load SAM2 model

    # SAM2 model ckpt path, you need to run sam2/checkpoints/download_ckpts.sh to download SAM2 model.
    # TAs will use our local path to run this code and use sam2.1_hiera_large for evaluate.  
    # TODO: Modify this path to your downloaded checkpoint path before running.
    sam2_checkpoint = "./checkpoints/sam2.1_hiera_large.pth"
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
    if sam2_checkpoint is None or not os.path.exists(sam2_checkpoint):
        raise RuntimeError("You must set sam2_checkpoint to your downloaded SAM2 checkpoint path before running.")
    
    sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)
    predictor = SAM2ImagePredictor(sam2_model)
    
    return predictor

def get_center_point_prompt(image):
    """
    Generate center point prompt for SAM2
    
    Args:
        image: RGB image array
    
    Returns:
        prompts: Dictionary containing point_coords and point_labels
    """
    h, w = image.shape[:2]
    
    # Use center point as prompt
    center_point = np.array([[w//2, h//2]])
    point_labels = np.array([1])  # 1 for foreground point
    
    return {
        'point_coords': center_point,
        'point_labels': point_labels,
        'box': None
    }

# NOTE: Visualization utilities removed in this refactor


def predict_mask(predictor, image, prompts):
    """
    Predict mask using SAM2
    
    Args:
        predictor: SAM2ImagePredictor
        image: RGB image array
        prompts: Dictionary containing prompts
    
    Returns:
        best_mask: Best predicted mask (highest score)
    """
    # Set image
    predictor.set_image(image)
    
    # Predict
    masks, scores, logits = predictor.predict(
        point_coords=prompts['point_coords'],
        point_labels=prompts['point_labels'],
        box=prompts['box'],
        multimask_output=True,
    )
    
    # Sort masks by score (highest first) and select the best one
    sorted_ind = np.argsort(scores)[::-1]
    masks = masks[sorted_ind]
    scores = scores[sorted_ind]
    logits = logits[sorted_ind]
    
    # Use the highest scoring mask
    best_mask = masks[0]
    best_score = scores[0]
    
    # print(f"    Selected mask with score: {best_score:.3f}")
    
    return best_mask

def evaluate_from_json(json_path, input_dir, output_dir):
    """Evaluate over an NDJSON file where each line has keys: source, target, prompt.

    - source: relative path under input_dir for the original image used to build GT mask
    - target: relative path under output_dir for the generated image used to build pred mask
    - prompt: text used for CLIP scoring 
    """
    # Set up device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Load SAM2
    predictor = load_sam2_model(device)

    # Load CLIP 
    try:
        clip_model, clip_processor = load_clip_model(device)
    except Exception as e:
        print(f"Warning: CLIP not available ({e}). CLIP scores will be 0.")
        clip_model = None
        clip_processor = None

    ious = []
    clip_scores = []
    final_scores = []
    results = []

    for i, obj in enumerate(stream_jsonl(json_path)):
        try:
            src_rel = obj.get('source', '')
            tgt_rel = obj.get('target', '')
            prompt_text = obj.get('prompt', '')

            source_path = os.path.join(input_dir, src_rel)
            target_path = os.path.join(output_dir, tgt_rel)

            if not os.path.exists(source_path):
                print(f"Source not found: {source_path}")
                continue
            if not os.path.exists(target_path):
                print(f"Target not found: {target_path}")
                continue

            # Load images
            source_image = np.array(Image.open(source_path).convert("RGB"))
            target_image = np.array(Image.open(target_path).convert("RGB"))

            # Build prompts
            source_prompts = get_center_point_prompt(source_image)
            target_prompts = get_center_point_prompt(target_image)

            # Predict masks
            gt_mask = predict_mask(predictor, source_image, source_prompts)
            pred_mask = predict_mask(predictor, target_image, target_prompts)

            # IoU
            iou = calculate_iou(pred_mask, gt_mask)
            ious.append(iou)

            # CLIP score
            clip_score = 0.0
            if clip_model is not None and prompt_text:
                clip_score = compute_clip_score(clip_model, clip_processor, target_image, prompt_text, device)
            clip_scores.append(clip_score)

            final_score = float(iou) * float(clip_score)
            final_scores.append(final_score)

            pred_pixels = int(np.sum(pred_mask > 0.5))
            gt_pixels = int(np.sum(gt_mask > 0.5))

            # Try to infer an id from target filename for logging
            image_id = os.path.splitext(os.path.basename(tgt_rel))[0]

            print(f"\n[{i}] {image_id}")
            # print(f"  IoU = {iou:.4f}")
            # print(f"  CLIP(text->image) = {clip_score:.4f} | prompt: '{prompt_text}'")
            # print(f"  Score = IoU * CLIP = {final_score:.4f}")
            # print(f"  - Source mask pixels: {gt_pixels}")
            # print(f"  - Target mask pixels: {pred_pixels}")

            results.append({
                'image_id': image_id,
                'iou': float(iou),
                'clip': float(clip_score),
                'final_score': float(final_score),
                'target_path': target_path,
                'source_path': source_path,
                'gt_pixels': gt_pixels,
                'pred_pixels': pred_pixels,
            })
        except Exception as e:
            print(f"Error processing entry {i}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue

    # Summary
    if not ious:
        print("No images were successfully evaluated!")
        return 0.0, results

    mean_iou = float(np.mean(ious))
    mean_clip = float(np.mean(clip_scores)) if clip_scores else 0.0
    k = min(10, len(final_scores))
    topk_final = sorted(final_scores)[-k:]
    mean_final = float(np.mean(topk_final)) if topk_final else 0.0

    # print("\n=========================================== evaluation ===========================================")
    # print(f"Count: {len(ious)}")
    # print(f"Mean IoU: {mean_iou:.4f}")
    # print(f"Mean CLIP: {mean_clip:.4f}")
    print(f"Mean Score (IoU*CLIP) [Top-{k}]: {mean_final:.4f}")
    # print(f"Min IoU: {np.min(ious):.4f}")
    # print(f"Max IoU: {np.max(ious):.4f}")
    # print(f"Std IoU: {np.std(ious):.4f}")
    # print("==================================================================================================\n")

    return mean_final, results

def save_results(results, output_path):
    """Save evaluation results to file under output_path."""
    os.makedirs(output_path, exist_ok=True)
    output_file = os.path.join(output_path, 'evaluation_results.txt')
    with open(output_file, 'w') as f:
        f.write("Image_ID\tIoU\tCLIP\tFinal\tSource_Mask_Pixels\tTarget_Mask_Pixels\tTarget_Path\tSource_Path\n")
        for result in results:
            f.write(
                f"{result['image_id']}\t{result['iou']:.4f}\t{result.get('clip', 0.0):.4f}\t{result.get('final_score', 0.0):.4f}\t"
                f"{result.get('gt_pixels', 'N/A')}\t{result.get('pred_pixels', 'N/A')}\t{result['target_path']}\t{result['source_path']}\n"
            )
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    # Reproducibility
    np.random.seed(42)
    torch.manual_seed(42)

    parser = argparse.ArgumentParser(description='Evaluate SAM2 IoU and CLIP on jsonl pairs.')
    parser.add_argument('--json_path', type=str, required=True, help='Path to NDJSON with keys: source, target, prompt')
    parser.add_argument('--input_dir', type=str, required=True, help='Root dir of ground-truth images')
    parser.add_argument('--output_dir', type=str, required=True, help='Root dir of generated/target images and where to write results')
    args = parser.parse_args()

    print("\n===================================== start evaluation =====================================")
    mean_final, results = evaluate_from_json(args.json_path, args.input_dir, args.output_dir)
    if results:
        # save_results(results, args.output_dir)
        print(f"Final Mean Score IoU: {mean_final:.4f}")
    print("====================================== end evaluation =====================================\n")
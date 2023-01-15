from fid_score.fid_score import FidScore
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
def fid_score_calculation(r_image_dir, generate_image_dir, batch_size=64):
    paths = [r_image_dir, generate_image_dir]
    batch_size = batch_size
    fid = FidScore(paths, device, batch_size)
    score = fid.calculate_fid_score()
    print("模型最后的Fid分数是:", score)
    return score


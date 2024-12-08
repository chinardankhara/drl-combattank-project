from src.utils import gif_to_heatmap
import os
from tqdm import tqdm

if __name__ == "__main__":
    gif_dir = "episodes_space_chase"
    out_dir = "heatmaps"
    algorithm = "reinforce"
    
    out_dir += "/" + algorithm
    all_gifs = [i for i in os.listdir(gif_dir) if i.endswith(".gif")]
    
    skip = len(all_gifs) // 5 # this ensures 5 images
    
    os.makedirs(out_dir, exist_ok=True)
    # select your gifs and make sure they are of consistant time steps 
    epochs = [0, 399, 799, 1199, 1599, 1999]
    
    for gif in tqdm(all_gifs[::skip]):
        epoch = int(gif.split("_")[-1].split(".")[0])
        
        gif_path = os.path.join(gif_dir, gif)
        out_path  = os.path.join(out_dir, f"epoch_{epoch}.png")
        
        gif_to_heatmap(gif_path, out_path, title=f"Epoch: {epoch+1}")
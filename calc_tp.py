import json
import os
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from smb_helper import to_text, concat_segments, to_tnsor
import ray
from tpkldiv import get_min_tpkldiv


def get_level(filename):
    temp = open(filename).readlines()
    lvl = []    
    for l in temp:
        if(len(l.strip()) > 0):
            lvl.append(l.strip())
    return lvl

def string2array(string):
    string = string.replace("[", "").replace("]", "")
    arr = np.fromstring(string, dtype=float, sep=' ')
    return np.array(arr)

def concat_segments(arr):
    lvl = arr[0]
    for i in range(1,len(arr)):
        lvl = np.concatenate((lvl,arr[i]),axis=1)
    width = len(lvl[0])
    height = len(lvl)
    result = ""
    for y in range(height):  
        for x in range(width):
            result += lvl[y][x]
        result += '\n'
    return result

@ray.remote
def run_parallel(data, model, dataset):
    latent = string2array(data["genes"])
    g = np.asarray(np.array_split(level, 1))
    pop_output = model.decoder(torch.tensor(g).float())
    segments = []
    for j in range(len(pop_output)):
        segments.append(to_text(pop_output[j]))
    lvl = concat_segments(segments)
    tpkldiv = get_min_tpkldiv(lvl, dataset)
                
    data["level"] = lvl
    data["tpkldiv"] = tpkldiv
    return data


def main():
    from SMB_models.smb_vae_2_1 import model

    model2_val = model.VAE().float()
    model2_val.load_state_dict(torch.load("./SMB_models/smb_vae_2_1/val_model.pt"))
    model2_val.eval()

    model2_15x = model.VAE().float()
    model2_15x.load_state_dict(torch.load("./SMB_models/smb_vae_2_1/1_5x_loss.pt"))
    model2_15x.eval()

    model2_2x = model.VAE().float()
    model2_2x.load_state_dict(torch.load("./SMB_models/smb_vae_2_1/2x_loss.pt"))
    model2_2x.eval()

    dirs = ["logit_vae2_val_", "logit_vae2_15x_", "logit_vae2_2x_", "tile_vae2_val_", "tile_vae2_15x_", "tile_vae2_2x_"]
    models = [model2_val, model2_15x, model2_2x, model2_val, model2_15x, model2_2x]

    ds_path = "./SMB_levels/"
    files_list = os.listdir(ds_path)
    dataset = []
    for fl in files_list:
        dataset.append(get_level(ds_path+fl))

    for i in range(len(dirs)):
        d = dir[i]
        model = models[i]
        for i in range(1, 6):
            path = "./vae2/" + d+str(i)
            files = os.listdir(path)
            os.mkdir("./vae2_lvls/" + d + "lvls_" + str(i))
            arr = []
            for file in files:
                if file != "details.json" and file != "log.txt":
                    with open(path + '/' + file, 'r') as f:
                        temp = json.load(f)
                        temp["f_name"] = file
                        arr.append(temp)

            futures=[run_parallel.remote(a, model, dataset) for a in arr]
            results = ray.get(futures)
            for r in results:
                file_name = "./vae2_lvls/" + d + "lvls_" + str(i) + "/" + r["f_name"]
                with open(file_name, 'w') as f:
                    f.write(json.dumps(r))         
        
                    
ray.init(num_cpus=48)
if __name__ == '__main__':
    main()        
        
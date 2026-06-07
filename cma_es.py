import json
import os
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from SMB_engine import runLevel
from smb_helper import to_text, concat_segments, to_tnsor
from cmaes import CMA
import ray

def string2array(string):
    string = string.replace("[", "").replace("]", "")
    arr = np.fromstring(string, dtype=float, sep=' ')
    return np.array(arr)

def gen_level(latent_arr, no_seg, model):
    latent = np.asarray(np.array_split(latent_arr, no_seg))
    output = model.decoder(torch.tensor(latent).float())
    segments = []
    for j in range(len(output)):
        segments.append(to_text(output[j]))
    lvl = concat_segments(segments)
    return segments, lvl, output


def find_tube_issues(lvl):
    test_tube = 0
    lvl_lines = lvl.split("\n")
    tube_issue = 0
    for i in range(len(lvl_lines)-1):
        l = lvl_lines[i]
        test_tube = 0
        for j in range(len(l)):
            c = l[j]
            if c == 'p':
                test_tube += 1
                if i < len(lvl_lines)-2:
                    if lvl_lines[i+1][j] != 'p' and lvl_lines[i+1][j] != 'X':
                        tube_issue += 1
            else:
                if test_tube % 2 > 0:
                    tube_issue += 1
                test_tube = 0
    return tube_issue

def tile_wise_loss(lvl, tns_arr, model):
    tns_lvl = []
    for i in range(len(tns_arr)):
        tns_lvl.append(to_tnsor(tns_arr[i]))
    tns_lvl = np.stack(tns_lvl, axis=0)
    tns_lvl = torch.tensor(tns_lvl)
    out, _, _ = model(tns_lvl.float())
    segments = []
    for j in range(len(out)):
        segments.append(to_text(out[j]))
    recon_lvl = concat_segments(segments)

    diff = 0
    h = len(lvl)
    w = len(lvl[0])
    for i in range(h):
        for j in range(w):
            if lvl[i][j] != recon_lvl[i][j]:
                diff += 1
    return diff/(h*w)

def categorical_cross_entropy(y_pred, y_true):
    y_pred = torch.clamp(y_pred, 1e-7, 1 - 1e-7)
    return -(y_true * torch.log(y_pred)).sum(dim=1).mean()

def reconstruction_loss(arr, model):
    tns_lvl = []
    for i in range(len(arr)):
        tns_lvl.append(to_tnsor(arr[i]))
    tns_lvl = np.stack(tns_lvl, axis=0)
    tns_lvl = torch.tensor(tns_lvl)

    out, mu, logvar = model(tns_lvl.float())
    recon_loss = categorical_cross_entropy(out, tns_lvl)
    return recon_loss.item()


@ray.remote
def calc_fitness(latent_arr, model):
    segments, lvl, tns_output = gen_level(latent_arr, 1, model)

    result = runLevel(lvl, "astar", 3, 50, True)
    playability = result.getCompletionPercentage()
    tube_error = find_tube_issues(lvl)
    fitness_1 = playability - 0.025*tube_error
    if fitness_1 == 1:
        recon_loss = reconstruction_loss(tns_output, model)
        fitness = 1 + (1- recon_loss)
        #tile_loss = tile_wise_loss(lvl, tns_output, model)
        #fitness = 1 + (1- tile_loss)
    else:
        recon_loss = 0
        #tile_loss = 0
        fitness = fitness_1
        
    stats = {
                "latent": np.array2string(latent_arr),
                "fitness": fitness,
                "playability": playability,
                "tube": tube_error,
                "recon_loss": recon_loss,
                "lvl": lvl
            }
    return -fitness, stats

class CMA_ES:
    def __init__(self, pop_size, model, latent_size=64):
        self._pop_size = pop_size
        self._model = model
        self._latent_size = latent_size
        #int cmaes controler  Mean vector, standard deviation (sigma)
        self._optimizer = CMA(mean=np.zeros(self._latent_size ), sigma=0.5, population_size=self._pop_size)
        print("Initilization complete.")


    def evolve(self, gens, path):
        for g in range(gens):    
            latents = [self._optimizer.ask() for _ in range(self._optimizer.population_size)]
                        
            futures = [calc_fitness.remote(l, self._model) for l in latents]
            results = ray.get(futures)
            
            fitness = []
            for r in results:
                fitness.append(r[0])
            # Re-package data into format expected by cmaes library
            solutions = list(zip(latents, fitness))
            
            # Pass data back into optimizer to shift the search center
            self._optimizer.tell(solutions)
            
        cnt = 0
        for r in results:
            with open(path + "/" + str(cnt) + ".json", 'w') as f:
                f.write(json.dumps(r[1]))
            cnt += 1
        

def main():
    from SMB_models.smb_vae_1_1 import model
    fitness = "cascaded: playability, pipe_error; recon_loss"
    pop_size = 100
    generations = 50
    vae_models = ["val_model"]  #, "1_5x_loss", "2x_loss"]
    for m in vae_models:
        model_path = "./SMB_models/smb_vae_1_1/" + m + ".pt"
        vae = model.VAE()
        vae.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        vae.eval()
        for i in range(1, 51):
            save_path = "./cmaes/vae1_logit/lg_vae1_"+m+"/run_" + str(i) 
            os.makedirs(save_path)
            print(save_path)
            details_file = save_path + "/details.json"
            with open(details_file, 'w') as f:
                temp = {
                    "pop_size": pop_size,
                    "generations": generations,
                    "no_of_segments": 1,
                    "vae_model": "smb_vae_1_1/"+m,
                    "fitness": fitness,
                    "playability": "astar, 3, 50",
                    "init": "random uniform -10,10"
                }
                f.write(json.dumps(temp))

            cma_es = CMA_ES(pop_size, vae)
            cma_es.evolve(generations, save_path)

ray.init(num_cpus=96)
if __name__ == '__main__':
    main()
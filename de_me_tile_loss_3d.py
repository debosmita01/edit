import json
import csv
import os
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from random import random, sample
from SMB_engine import runLevel
from smb_helper import to_text, concat_segments, to_tnsor
from scipy import stats
from math import log, e
from SMB_models.smb_vae_3_1 import model
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

def calc_lineancy(lvl):
    arr = []
    lvl_arr = lvl.split("\n")
    enemy = 0
    big_gap = 0
    small_gap = 0
    for i in range(len(lvl_arr)-1):
        for j in range(len(lvl_arr[0])):
            if lvl_arr[i][j] == 'E':
                enemy += 1

    is_gap = False
    gap_len = 0
    for j in range(1, len(lvl_arr[0])):
        if lvl_arr[15][j] == '-':
            is_gap = True
            gap_len += 1
        else:
            if is_gap:
                if gap_len > 0 and gap_len < 2:
                    small_gap += 1
                elif gap_len >= 2:
                    big_gap += 1
            is_gap = False
            gap_len = 0


    return small_gap, big_gap, enemy

def get_platform_points(lvl):
    arr_x = []
    arr_y = []
    for i in range(1, len(lvl)-1):
        plat = False
        plat_start = 0
        plat_end = 0
        for j in range(len(lvl[0])):
            if (lvl[i][j]=='X' or lvl[i][j]=='S' or lvl[i][j]=='Q' or lvl[i][j]=='p') and (lvl[i-1][j]=='-' or \
            lvl[i-1][j]=='E' or lvl[i-1][j]=='o' or lvl[i-1][j]=='F' or lvl[i-1][j]=='M'):
                if not plat:
                    plat_start = j
                    plat = True
            else:
                if plat:
                    plat_end = j-1
                    plat_len = plat_end-plat_start
                    mid_point = plat_end - int(plat_len/2)
                    arr_x.append(mid_point)
                    arr_y.append(16-i)
                plat = False
        if plat:
            plat_len = j-plat_start
            mid_point = j - int(plat_len/2)
            arr_x.append(mid_point)
            arr_y.append(16-i)

    if len(arr_x) == 0:
        arr_x.append(0)
        arr_y.append(0)
    return arr_x, arr_y


def calc_linearity_platforms(lvl):
    lvl = lvl.split("\n")
    length = len(lvl[0])
    x, points = get_platform_points(lvl)
    result = all(a == x[0] for a in x)
    if len(x) >= 1 and result:
        val = 0
    else:
        slope, intercept, r, p, std_err = stats.linregress(x, points)

        def predict_line(x):
            return slope * x + intercept
        line = list(map(predict_line, x))

        drift = 0
        for i in range(len(points)):
            dist = abs(line[i] - points[i])
            drift += dist/16
        val = drift/length
    return val

def find_tube_issues(lvl):
    test_tube = 0
    lvl_lines = lvl.split("\n")
    tube_issue = 0
    for l in lvl_lines:
        test_tube = 0
        for c in l:
            if c == 'p':
                test_tube += 1
            else:
                if test_tube % 2 > 0:
                    tube_issue += 1
                test_tube = 0
    return tube_issue


def calc_fitness(latent_arr, no_seg, model, dataset, bin1, bin2, bin3):
    segments, lvl, tns_output = gen_level(latent_arr, no_seg, model)

    result = runLevel(lvl, "astar", 3, 50, True)
    playability = result.getCompletionPercentage()
    tube_error = find_tube_issues(lvl)
    fitness_1 = playability - 0.025*tube_error
    if fitness_1 == 1:
        tile_loss = tile_wise_loss(lvl, tns_output, model)
        fitness = 1 + (1- tile_loss)
    else:
        tile_loss = 0
        fitness = fitness_1

    behaviors = []
    linearity = calc_linearity_platforms(lvl)
    small_gap, big_gap, enemies = calc_lineancy(lvl)
    lineancy = small_gap + big_gap + enemies
    diff = find_dist(dataset, segments)
    b1 = np.digitize(linearity, bin1, right=True)
    behaviors.append(b1)
    b2 = np.digitize(lineancy, bin2, right=True)
    behaviors.append(b2)
    b3 = np.digitize(diff, bin3, right=True)
    behaviors.append(b3)
    return fitness, playability, tube_error, tile_loss, behaviors, linearity, lineancy, diff

def find_dist(dataset, lvl):
    min_diff = 16*16
    str_lvl = ''.join([str(elem) for elem in lvl])
    str_lvl = str_lvl.replace('[','').replace(']','').replace(' ','').replace("'","").replace('\n','')
    for d in dataset:
        diff = sum(1 for a, b in zip(str_lvl, d[1]) if a != b)
        if diff < min_diff:
            min_diff = diff
    return min_diff

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


class Chromosome():
    def __init__(self, genes, start_genes):
        self._genes = genes
        self._start_genes = start_genes
        self._fitness = -1
        self._playability = -1
        self._tube_loss = -1
        self._tile_loss = -1
        self._behaviors  = []
        self._linearity = -1
        self._lineancy = -1
        self._dist = -1

    def eval(self, no_seg, model, dataset, bin1, bin2, bin3):
        self._fitness, self._playability, self._tube_loss, self._tile_loss, self._behaviors, self._linearity, self._lineancy, self._dist = calc_fitness(
            self._genes, no_seg, model, dataset, bin1, bin2, bin3)


    def save(self, file_name):
        with open(file_name, 'w') as f:
            temp = {
                "genes": np.array2string(self._genes),
                "start_genes": np.array2string(self._start_genes),
                "fitness": self._fitness,
                "playability": self._playability,
                "tube": self._tube_loss,
                "tile_loss": self._tile_loss,
                "behaviors": str(self._behaviors),
                "linearity": self._linearity,
                "lineancy": self._lineancy,
                "distance": self._dist
            }
            f.write(json.dumps(temp))

    def load(self, file_name):
        print(file_name)
        with open(file_name, 'r') as f:
            temp = json.load(f)
            self._genes = string2array(temp["genes"])
            self._start_genes = string2array(temp["start_genes"])
            self._fitness = temp["fitness"]
            self._playability = temp["playability"]
            self._tube_loss = temp["tube"]
            self._tile_loss = temp["tile_loss"]
            self._behaviors = temp["behaviors"]
            self._linearity = temp["linearity"]
            self._lineancy = temp["lineancy"]
            self._dist = temp["distance"]


class Archive:
    def __init__(self):
        self._map = {}

    def __len__(self):
        return len(self._map)

    def __str__(self):
        chromosomes = self.get_all(self.keys())
        sorted_chromosomes = sorted(chromosomes, key=lambda c: c._fitness)
        best = sorted_chromosomes[-1]
        qd_score = sum(map(lambda x:x._fitness, chromosomes))
        return f"Arhcive Size: {len(self._map)} max: {str(best._fitness)} QD score: {qd_score}"

    def get_qd_score(self):
        chromosomes = self.get_all(self.keys())
        qd_score = sum(map(lambda x:x._fitness, chromosomes))
        return qd_score

    def keys(self, dim=-1, value=-1):
        if len(self._map) == 0:
            return np.array([])
        num_dim = len(list(self._map.keys())[0].split(","))
        keys = list(self._map.keys())
        result = []
        for key in keys:
            values = key.split(",")
            temp = []
            for v in values:
                temp.append(int(v))
            result.append(temp)
        result = np.array(result)
        if dim >= 0:
            result = np.array([k for k in result if k[dim] == value])
        return result


    def add(self, chromosome):
        key = ",".join([str(temp) for temp in chromosome._behaviors])
        if key not in self._map:
            self._map[key] = chromosome
        elif key in self._map:
            if chromosome._fitness > self._map[key]._fitness:
                del self._map[key]
                self._map[key] = chromosome

    def random_sample(self):
        keys = list(self._map.keys())
        indices = np.random.choice(len(keys), 4)
        sample = [self._map[keys[idx]] for idx in indices]
        return sample[0], sample[1], sample[2], sample[3]

    def get(self, dimension):
        key = ",".join([str(temp) for temp in dimension])
        if key in self._map:
            return self._map[key]
        return None

    def get_all(self, dimensions):
        result = []
        for dim in dimensions:
            result.append(self.get(dim))
        return result

    def save(self, folder):
        #os.makedirs(folder)
        for key in self._map.keys():
            self._map[key].save(os.path.join(folder, f"{key}.json"))

    def load(self, folder):
        self._map = {}
        files = [fn for fn in os.listdir(folder) if ".json" in fn and fn != "details.json"]
        #print(files)
        for fn in files:
            key = fn.split(".json")[0]
            genes = np.zeros(64*5)
            self._map[key] = Chromosome(genes, genes)
            self._map[key].load(os.path.join(folder, f"{key}.json"))

@ray.remote
def apply_mutation(c, _map, _no_segments, _model, _dataset, _scaling_factor, _crossover_rate, _bin1, _bin2,
                   _bin3):
    target, donor1, donor2, donor3 = _map.random_sample()
    r1 = donor1._genes
    r2 = donor2._genes
    r3 = donor3._genes

    x_diff = [r2_j - r3_j for r2_j, r3_j in zip(r2, r3)]
    mutant = [r1_j + _scaling_factor * x_diff_j for r1_j, x_diff_j in zip(r1, x_diff)]
    # mutant = ensure_bounds(mutant, bounds)

    trial = []
    for j in range(len(target._genes)):
        if random() <= _crossover_rate:
            trial.append(mutant[j])
        else:
            trial.append(target._genes[j])
    trial = np.asarray(trial)
    trial_chromosome = Chromosome(trial, target._genes)
    trial_chromosome.eval(_no_segments, _model, _dataset, _bin1, _bin2, _bin3)
    return trial_chromosome


class DE_ME:
    def __init__(self, pop_size, no_seg, vae_model, model_path, dataset, bin1, bin2, bin3):
        self._pop_size = pop_size
        self._no_seg = no_seg
        self._bin1 = bin1
        self._bin2 = bin2
        self._bin3 = bin3
        self._model = vae_model
        self._dataset = dataset

        self._map = Archive()
        for i in range(self._pop_size):
            # latent = sample_latent_space(model_path + "/latent_dist_tr.csv", self._no_seg, var=1)
            latent = np.random.uniform(-10, 10, 64 * self._no_seg)
            c = Chromosome(latent, latent)
            c.eval(self._no_seg, self._model, self._dataset, self._bin1, self._bin2, self._bin3)
            self._map.add(c)

    def run_de_me(self, sf, cr, bounds):
        arc = ray.put(self._map)
        n_seg = ray.put(self._no_seg)
        modl = ray.put(self._model)
        ds = ray.put(self._dataset)
        b1 = ray.put(self._bin1)
        b2 = ray.put(self._bin2)
        b3 = ray.put(self._bin3)
        futures=[apply_mutation.remote(c, arc, n_seg, modl, ds, sf, cr, b1, b2, b3) for c in range(100)]
        #results = ray.get(futures)
        #for r in results:
        #   self._map.add(r)
        while len(futures):
            done_id, futures = ray.wait(futures)
            self._map.add(ray.get(done_id[0]))

    def save(self, folder):
        self._map.save(folder)

    def load(self, folder):
        self._map.load(folder)

def main(resume):
    behavior1 = "linearity"
    behavior2 = "lineancy"
    behavior3 = "distance_SMB"
    fitness = "cascaded: playability, pipe_error; tile_loss"

    pop_size = 100
    no_seg = 1
    generations = 10000
    sf = 0.2
    cr = 0.5
    # bounds = [(-1, 1)] * (64 * no_seg)
    bounds = None

    bin1 = [0, .005, .010, .015, .020, .025, .030, .035, .040]  # linearity
    bin2 = [0, 1, 2, 3, 4, 5, 6, 7, 8]  # lineancy
    bin3 = [5, 10, 15]  # distance

    smb_lvls = "./smb_data.csv"
    dataset = [line for line in csv.reader(open(smb_lvls))]

    model_path = "./SMB_models/smb_vae_2_1/equal_loss.pt"
    vae = model.VAE()
    vae.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    vae.eval()

    save_path = "./de_me_vae2/tile_loss_vae2(equal_loss)_1"
    
    if not resume:
        '''os.makedirs(save_path)
        details_file = save_path + "/details.json"
        with open(details_file, 'w') as f:
            temp = {
                "pop_size": pop_size,
                "generations": generations,
                "scaling_factor": sf,
                "crossover_rate": cr,
                "bounds": "none",
                "no_of_segments": no_seg,
                "vae_model": "smb_vae_2_1/equal_loss",
                "fitness": fitness,
                "behavior_1": behavior1,
                "behavior_2": behavior2,
                "behavior_3": behavior3,
                "bin1": str(bin1),
                "bin2": str(bin2),
                "bin3": str(bin3),
                "playability": "astar, 3, 50",
                "init": "random uniform -10,10"
            }
            f.write(json.dumps(temp))

        de_me = DE_ME(pop_size, no_seg, vae, model_path, dataset, bin1, bin2, bin3)
        de_me.save(save_path)
        print("population created")
        for g in range(1, generations + 1):
            de_me.run_de_me(sf, cr, bounds)
            text_file = open(save_path + "/log.txt", "a")
            text_file.write("Gen: {} - {} \n".format(g, str(de_me._map)))
            text_file.close()
            if g % 100 == 0:
                print("Gen: {} - {}".format(g, str(de_me._map)))
                de_me.save(save_path)
        print("Gen: {} - {}".format(g, str(de_me._map)))
        de_me.save(save_path)
        print("saved")'''
        print("not resume")
    else:
        '''de_me = DE_ME(0, no_seg, vae, model_path, dataset, bin1, bin2, bin3)
        de_me.load(save_path)
        print("Archive loaded")
        for g in range(1, generations + 1):
            de_me.run_de_me(sf, cr, bounds)
            text_file = open(save_path + "/log.txt", "a")
            text_file.write("Gen: {} - {} \n".format(g, str(de_me._map)))
            text_file.close()
            if g % 100 == 0:
                print("Gen: {} - {}".format(g, str(de_me._map)))
                de_me.save(save_path + exp)
        print("Gen: {} - {}".format(g, str(de_me._map)))
        de_me.save(save_path)
        print("saved")'''
        print("resume")

#ray.init(num_cpus=96)
if __name__ == '__main__':
    resume = False
    main(resume)
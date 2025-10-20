import pyrosetta
pyrosetta.init('-mute all')
from pyrosetta.rosetta.core.scoring import ScoreType

import menten_gcn as mg
import menten_gcn.decorators as decs

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd
import random
import glob
import math
from multiprocessing import Pool, cpu_count

decorators = [decs.CACA_dist(use_nm=True),
              decs.trRosettaEdges(sincos=True, use_nm=True),
              decs.Rosetta_Ref2015_TwoBodyEneriges(individual=True, score_types=[ScoreType.fa_rep,
                                                                                 ScoreType.fa_atr, 
                                                                                 ScoreType.fa_sol, 
                                                                                 ScoreType.lk_ball_wtd, 
                                                                                 ScoreType.fa_elec, 
                                                                                 ScoreType.hbond_sr_bb, 
                                                                                 ScoreType.hbond_lr_bb, 
                                                                                 ScoreType.hbond_bb_sc, 
                                                                                 ScoreType.hbond_sc])]
data_maker = mg.DataMaker(decorators=decorators,
                           edge_distance_cutoff_A=8.0,
                           max_residues=20,
                           nbr_distance_cutoff_A=12.0)
data_maker.summary()

def get_cys(pose):
    disulfides = []
    for i in range(1, pose.size()+1):
        if pose.residue(i).name() == "CYS":
            res = pose.residue(i)
            disulfide_partner = res.residue_connection_partner(res.n_current_residue_connections())
            if disulfide_partner > i:
                disulfides.append([i, disulfide_partner])
    return disulfides

def get_negative_examples(pose):
    negative_examples = []
    pairs = []
    disulfides = get_cys(pose)
    disulfide_res = {i for bond in disulfides for i in bond}

    for i in range(1, pose.size() + 1):
        if i not in disulfide_res:
            negative_examples.append(i)

    for i in negative_examples:
        for j in negative_examples:
            if i != j and j > i:
                distance = calculate_distance(pose, i, j)
                if distance <= 10:
                    pairs.append([i, j])

    return random.sample(pairs, min(20, len(pairs))) #take 20 samples

def calculate_distance(pose, res1, res2):
    coord1 = pose.residue(res1).xyz("CA")
    coord2 = pose.residue(res2).xyz("CA")
    
    # Calculate the Euclidean distance between the two coordinates
    distance = math.sqrt((coord1[0] - coord2[0])**2 + 
                         (coord1[1] - coord2[1])**2 + 
                         (coord1[2] - coord2[2])**2)
    return distance

def process_and_save_chunk(chunk_index, pdb_chunk):
    Xs_chunk = []
    As_chunk = []
    Es_chunk = []
    outs_chunk = []

    for pdb_file in pdb_chunk:
        pose = pyrosetta.pose_from_pdb(pdb_file)
        cys = get_cys(pose)
        no_cys = get_negative_examples(pose)
        wrapped_pose = mg.RosettaPoseWrapper(pose)
        cache = data_maker.make_data_cache(wrapped_pose)

        for i in cys:
            X, A, E, resids = data_maker.generate_input( wrapped_pose, i, data_cache=cache )
            Xs_chunk.append(X)
            As_chunk.append(A)
            Es_chunk.append(E)
            outs_chunk.append([1.0,])

        for i in no_cys:
            X, A, E, resids = data_maker.generate_input( wrapped_pose, i, data_cache=cache )
            Xs_chunk.append(X)
            As_chunk.append(A)
            Es_chunk.append(E)
            outs_chunk.append([0.0,])
    
    np.savez_compressed(f'graphs/data_chunk_{chunk_index}-allcys.npz', Xs=Xs_chunk, As=As_chunk, Es=Es_chunk, outs=outs_chunk)
    print(f'Chunk {chunk_index} saved!')

def process_chunks(pdb_list, chunk_size):
        
    # Split the pdb_list into chunks
    chunks = [pdb_list[i:i + chunk_size] for i in range(0, len(pdb_list), chunk_size)]

    # Set up multiprocessing pool
    with Pool(processes=cpu_count()) as pool:
        pool.starmap(process_and_save_chunk, [(i + 1, chunk) for i, chunk in enumerate(chunks)])
        print("Processing and saving completed.")

# Process the pdb_list in chunks
pdb_list = glob.glob('/media/data/jri/cath_S40/whole-dataset/*?.pdb')
chunk_size = 10
process_chunks(pdb_list, chunk_size)

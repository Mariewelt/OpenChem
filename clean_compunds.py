import argparse
import os
from rdkit import Chem
from rdkit import rdBase
rdBase.DisableLog('rdApp.error')

parser = argparse.ArgumentParser(description="Clean compounds")
parser.add_argument("--folder_low", default=None, type=int)
parser.add_argument("--folder_high", default=None, type=int)

args = parser.parse_args()

dir_low = args.folder_low
dir_high = args.folder_high

dir_prefix = "/home/mpopova/Work/my_project/generated_compounds/"

for i in range(dir_high, dir_low, -1):
    dir = "job" + str(i)
    arr = os.listdir(dir_prefix + dir)

    for file in arr:
        if file[-5:] != "clean" or file + ".clean" not in arr:
            f = open(dir_prefix + dir + "/" + file, "r")
            smiles = [line[:-1] for line in f]
            f.close()
            mols = [Chem.MolFromSmiles(sm) for sm in smiles]
            clean_smiles = [Chem.MolToSmiles(mol) for mol in mols if mol]
            f = open(dir_prefix + dir + "/" + file + ".clean", "w")
            for sm in clean_smiles:
                f.writelines(sm + "\n")
            f.close()
            print("File " + file + " processed!")
    print("Folder " + dir + " processed!")

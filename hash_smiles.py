import argparse

parser = argparse.ArgumentParser(description="Hash SMILES")
parser.add_argument("--dir", default=None, type=str)

args = parser.parse_args()

dir_ = args.dir

f = open(dir_, "r")

smiles = [line[:-1] for line in f]
f.close()

hashes = [hash(sm) for sm in smiles]

f = open(dir_ + ".hash", "w")

for h in hashes:
    f.writelines(str(h) + "\n")
f.close()
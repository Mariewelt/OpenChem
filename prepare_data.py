import argparse
import pickle


def write_file(data, f):
    l = len(data['smiles'])
    for i in range(l):
        f.writelines(data['smiles'][i] + ',' + data['seq'] + ',' +
                     str(data['labels'][i]) + '\n')


def main():
    parser = argparse.ArgumentParser(description='Target setup')
    parser.add_argument("--target", type=str, dest="target")

    args, unknown = parser.parse_known_args()

    all_data = pickle.load(open('/data/all_data.pkl', 'rb'))

    names = list(all_data.keys())
    for k in names:
        f_2 = open('/data/train.txt', 'w')
        f = open('/data/test.txt', 'w')
        for i in range(len(names)):
            if names[i] == args.target:
                write_file(all_data[k], f)
                f.close()
            else:
                write_file(all_data[names[i]], f_2)
        f_2.close()


if __name__ == '__main__':
    main()

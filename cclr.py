import os
import subprocess

def process_files(path):
    for entry in os.listdir(path):
        full_path = os.path.join(path, entry)
        if os.path.isdir(full_path):
            process_files(full_path)
        elif full_path.endswith(".pgn"):
            process_file(full_path)

def process_file(pgn_file):
    out_file = pgn_file.replace(".pgn", ".csv").replace("test", "test_labeled").replace("train", "train_labeled")
    print(f'processing {pgn_file} ==> {out_file}')
    cmd = 'java -jar chess4j-6.0-uber.jar -mode label -pgn ' + pgn_file + ' -out ' + out_file
    os.system('cd /home/james/chess4j-latest && ' + cmd)

process_files('/home/james/data/chess/cclr')

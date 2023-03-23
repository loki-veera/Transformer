import os

import wget
import glob
import tarfile


def download():
    URL = "https://www.statmt.org/europarl/v7/de-en.tgz"
    path = "./"

    wget.download(URL, out=path)

    # Extract the tgz
    file = tarfile.open('./de-en.tgz')
    file.extractall('./de-en')
    file.close()


def extract():
    path = './de-en/'
    files = os.listdir(path)
    for file in files:
        file_path = path + file
        text = []
        with open(file_path, encoding='utf-8') as f:
            text = ["" + line.strip() + "" for line in f]

        file_name = f"{path}{file.replace('.', '_')}.txt"
        with open(file_name, 'w') as f:
            for line in text:
                f.write(f"{line}\n")


if __name__ == "__main__":
    download()
    extract()

import wget
import tarfile

def download():
    URL = "https://www.statmt.org/europarl/v7/de-en.tgz"
    path = "./data"
    
    wget.download(URL, out = path)

    # Extract the tgz
    file = tarfile.open('./data/de-en.tgz')
    file.extractall('./data/de-en')
    file.close()

if __name__ == "__main__":
    download()
import argparse

def download_all(data_dir: str):
    ## TODO
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download datasets")
    parser.add_argument("--data-dir", type=str, required=True)
    args = parser.parse_args()
    download_all(data_dir=args.data_dir)
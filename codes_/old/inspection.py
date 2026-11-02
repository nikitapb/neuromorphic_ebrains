import pickle
from pathlib import Path

fname = Path(
    "/mnt/c/Users/sarah/Downloads/output_all_with_dl_resaved.pkl"
)

with open(fname, "rb") as f:
    obj = pickle.load(f)

print(type(obj))
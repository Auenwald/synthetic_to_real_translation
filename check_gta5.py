from PIL import Image
from pathlib import Path
from tqdm import tqdm

files = sorted(Path("gta5/images").glob("*.png"))
corrupt = []

for p in tqdm(files):
    try:
        Image.open(p).load()
    except Exception as e:
        corrupt.append(p.name)
        print(f"\nKorrupt: {p.name} — {e}")

print(f"\nGesamt korrupt: {len(corrupt)}")
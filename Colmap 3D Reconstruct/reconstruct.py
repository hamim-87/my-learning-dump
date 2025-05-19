from pathlib import Path
import pycolmap
import sqlite3

img_path = Path("Ammonite")
output_path = Path("reconstruct_3d")
database_path = output_path/"database.db"


# Create the output directory if it doesn't exist
if not output_path.exists():
    output_path.mkdir(parents=-True, exist_ok=True)

#feature extraction

sift_opt = pycolmap.SiftExtractionOptions()
sift_opt.max_num_features = 512

pycolmap.extract_features(
    database_path=str(database_path),
    image_path=str(img_path),
    sift_options=sift_opt
)



    
    
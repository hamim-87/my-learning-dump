
import matplotlib.pyplot as plt
import numpy as np
import sqlite3
from PIL import Image


db_conn = sqlite3.connect("reconstruct_3d/database.db")
db_cursor = db_conn.cursor()


def visualize_keypoints(image_id):

    db_cursor.execute(
        "SELECT rows, cols, data FROM keypoints WHERE image_id = ?", 
        (image_id,)
    )
    rows, cols, data = db_cursor.fetchone()
    keypoints = np.frombuffer(data, dtype=np.float32).reshape(-1, 4) #convert to numpy array

 
    db_cursor.execute(
        "SELECT name FROM images WHERE image_id = ?", 
        (image_id,)
    )
    img_name = db_cursor.fetchone()[0]
    img = Image.open(f"Ammonite/{img_name}")



    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(img)
    ax.scatter(
        keypoints[:, 0], 
        keypoints[:, 1], 
        s=5, 
        c='green', 
        alpha=0.5,
        linewidths=0
    )
    ax.set_title(f"{img_name} - {len(keypoints)} features")
    ax.axis('off')
    
  
    plt.savefig(f"features_{img_name}.png", bbox_inches='tight')
    plt.show()


    plt.close(fig)
    print(f"Saved feature visualization to features_{img_name}.png")



# Usage
visualize_keypoints(1)  

db_conn.close()
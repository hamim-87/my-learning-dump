from transformers import AutoImageProcessor, AutoModel
import torch
from PIL import Image
import requests

url1 = "https://raw.githubusercontent.com/magicleap/SuperGluePretrainedNetwork/refs/heads/master/assets/phototourism_sample_images/united_states_capitol_98169888_3347710852.jpg"
img1 = Image.open(requests.get(url1,stream=True).raw)

url2 = "https://raw.githubusercontent.com/magicleap/SuperGluePretrainedNetwork/refs/heads/master/assets/phototourism_sample_images/united_states_capitol_26757027_6717084061.jpg"
img2 = Image.open(requests.get(url2,stream=True).raw)

image = [img1, img2]

processor = AutoImageProcessor.from_pretrained("magic-leap-community/superglue_outdoor")
model = AutoModel.from_pretrained("magic-leap-community/superglue_outdoor")

input = processor(image, return_tensors="pt")

with torch.no_grad():
    output = model(**input)
    print("-----------------------output-------------------------")
    print(output)


image_sizes = [[(img.height,img.width) for img in image]]

res = processor.post_process_keypoint_matching(output,image_sizes,threshold=0.2)

print("----------------------------res ----------------------")
print(res)

print("--------------------------------------------------------------")

for i, out in enumerate(res):
    print("image pair: ", i)
    for kp0,kp1,match in zip(out["keypoints0"],out["keypoints1"], out["matching_scores"]):
        print( f" keypoint {kp0.numpy()} match {kp1.numpy()} with socre {match}")



import matplotlib.pyplot as plt
import numpy as np

# Create side by side image
merged_image = np.zeros((max(img1.height, img2.height), img1.width + img2.width, 3))
merged_image[: img1.height, : img1.width] = np.array(img1) / 255.0
merged_image[: img2.height, img1.width :] = np.array(img2) / 255.0
plt.imshow(merged_image)
plt.axis("off")

# Retrieve the keypoints and matches
result = res[0]
keypoints0 = result["keypoints0"]
keypoints1 = result["keypoints1"]
matching_scores = result["matching_scores"]
keypoints0_x, keypoints0_y = keypoints0[:, 0].numpy(), keypoints0[:, 1].numpy()
keypoints1_x, keypoints1_y = keypoints1[:, 0].numpy(), keypoints1[:, 1].numpy()

# Plot the matches
for keypoint0_x, keypoint0_y, keypoint1_x, keypoint1_y, matching_score in zip(
        keypoints0_x, keypoints0_y, keypoints1_x, keypoints1_y, matching_scores
):
    if matching_score >= 0.90 :
        plt.plot(
        [keypoint0_x, keypoint1_x + img1.width],
        [keypoint0_y, keypoint1_y],
        color=plt.get_cmap("RdYlGn")(matching_score.item()),
        alpha=0.9,
        linewidth=0.5,
        )
        plt.scatter(keypoint0_x, keypoint0_y, c="black", s=2)
        plt.scatter(keypoint1_x + img1.width, keypoint1_y, c="black", s=2)

# Save the plot
plt.savefig("matched_image.png", dpi=300, bbox_inches='tight')
plt.close()
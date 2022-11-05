import numpy as np

from material_selection import find_material_image
from material_transfer import *
from PIL import Image, ImageOps

# set to true if you want input the mask yourself
supply_mask = False

# path of the image the original image
content_img_path = os.path.join('path', 'image.jpg')
content_img = image_loader(content_img_path, device=device, img_size=img_size)

# find suitable style image based on the type of texture, based on classes from DTD dataset
material = find_material_image(content_img, "bubbly")


# perform object segmentation and get boolean mask
original = read_image("path")

if supply_mask:
    mask_im = Image.Image.rotate(ImageOps.flip(ImageOps.grayscale(Image.open('path_to_mask'))), -90)
    mask = np.asarray(mask_im, dtype=bool)
else:
    mask = get_mask(original)
    mask_im = Image.Image.rotate(Image.fromarray(np.array(mask)), angle=-90)

# run the material transfer on the entire content image
new_style = run_material_transfer(content_img_path, material)

# use the mask to get the new material of the object of interest
res = applyMaterial(mask, original, new_style)
im = Image.fromarray((res * 255).astype(np.uint8))

plt.imshow((res * 255).astype(np.uint8))
plt.show()
im.save("outputs/res.jpg")


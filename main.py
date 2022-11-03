import numpy as np

from material_selection import find_material_image
from material_transfer import *

# path of the image the original image
content_img_path = os.path.join('data', 'vase.jpg')
content_img = image_loader(content_img_path, device=device, img_size=img_size)

# find suitable style image based on the type of texture
material = find_material_image(content_img, "extra")
plt.imshow(transforms.ToPILImage()(material.squeeze(0)))
plt.show()

# perform object segmentation and get boolean mask
original = read_image(str(Path('D:\\Downloads D') / '000000002150.jpg'))
mask = get_mask(original)
mask_im = Image.Image.rotate(Image.fromarray(np.array(mask)), angle=-90)

plt.imshow(mask_im)
plt.show()
mask_im.save("outputs/mask.jpg")

# run the material transfer on the entire content image
new_style = run_material_transfer(content_img_path, material)

# use the mask to get the new material of the object of interest
res = applyMaterial(mask, original, new_style)
im = Image.fromarray((res * 255).astype(np.uint8))

plt.imshow((res * 255).astype(np.uint8))
plt.show()
im.save("outputs/res.jpg")

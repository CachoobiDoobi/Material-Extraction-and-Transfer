from material_selection import find_style_image
from your_code_here import *

# path of the image that will
content_img_path = os.path.join('data', 'vase.jpg')
content_img = image_loader(content_img_path, device=device, img_size=img_size)

style_img = find_style_image(content_img, "extra")
plt.imshow(transforms.ToPILImage()(style_img.squeeze(0)))
plt.show()


original = read_image(str(Path('D:\\Downloads D') / '000000002150.jpg'))
new_style = run_style_transfer(content_img_path, style_img)

mask = get_mask(original)
res = applyMaterial(mask, original, new_style)
im = Image.fromarray((res * 255).astype(np.uint8))

plt.imshow((res * 255).astype(np.uint8))
plt.show()
im.save("outputs/res.jpg")

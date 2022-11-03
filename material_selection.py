import glob

from material_transfer import *

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
else:
    print("we CUDA but we dont")


def get_images(style):
    imgs = []
    files = glob.glob("data/dataset/dtd/dtd/images/" + style + "/*.jpg")
    for img in files:
        image = image_loader(img, device=device, img_size=512)
        imgs.append(image)

    return imgs


def find_material_image(content_img, material_type):
    """ Finds the optimal material image

     # Parameters:
        @content_img, the original image.
        @material_type, the type of the material, corresponds to categoried from the DTD dataset.


    # Returns the image with the most simialr content to the original one
    """

    # Initialize Model
    model = Vgg19(content_layers, style_layers, device)

    normed_content_img = normalize(content_img, vgg_mean, vgg_std)
    # Retrieve feature maps for content and style image
    content_features = model(normed_content_img)

    images = get_images(material_type)

    bestIm = None
    lowest_loss = 9999999
    for img in images:
        # Either initialize the image from random noise or from the content image
        optim_img = torch.nn.Parameter(img, requires_grad=True)

        # Retrieve features of image that is being optimized
        normed_img = normalize(optim_img, vgg_mean, vgg_std)
        input_features = model(normed_img)
        # Compute content loss
        loss = content_loss(input_features, content_features, content_layers)
        if loss < lowest_loss:
            bestIm = img
            lowest_loss = loss
    # return image with the most similar content
    return bestIm

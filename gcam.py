from PIL import Image
from imageio import imread
from skimage.io import imsave
from imgaug import imshow
from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from skimage.io import show
from skimage.transform import resize
from torch.autograd import Variable
from torchvision import transforms
from torchvision.models import resnet50, mobilenetv3, mnasnet1_0, resnet18, resnet34, resnet152
from torch import from_numpy
from pathlib import Path
import numpy as np

from skimage.data import chelsea
from tqdm import tqdm

gradcam_dict = {
    "gradcam": GradCAM,
    "gradcam_plusplus": GradCAMPlusPlus,
    "eigengradcam": EigenCAM
}


def gradcam_img(model, image, target_layer, cam_type, target_category):
    #rgb_img = imread(image) / 255.
    imsize = (224, 224)
    loader = transforms.Compose([transforms.Scale(imsize), transforms.ToTensor()])
    rgb_img = resize(imread(image) / 255., (224, 224))

    def image_loader(image_name):
        """load image, returns cuda tensor"""
        image = Image.open(image_name)
        image = loader(image).float()
        image = Variable(image, requires_grad=True)
        image = image.unsqueeze(0)  #this is for VGG, may not be needed for ResNet
        return image.cuda()  #assumes that you're using GPU

    input_tensor = image_loader(image)


    # Create an input tensor image for your model..
    # Note: input_tensor can be a batch tensor with several images!

    # Construct the CAM object once, and then re-use it on many images:
    cam = gradcam_dict[cam_type](model=model, target_layer=target_layer, use_cuda=True)

    # If target_category is None, the highest scoring category
    # will be used for every image in the batch.
    # target_category can also be an integer, or a list of different integers
    # for every image in the batch.

    # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
    grayscale_cam = cam(input_tensor=input_tensor, target_category=target_category)

    # In this example grayscale_cam has only one image in the batch:
    grayscale_cam = grayscale_cam[0, :]
    visualization = show_cam_on_image(rgb_img.astype(np.float32), grayscale_cam)
    return visualization, resize(imread(image), (224, 224))

names = {
    "dome": 897,
    "altar": 677,
    "church": 690,
    "monastery": 686,
    "palace": 685,
}

names = {value: key for key, value in names.items()}

target_categories = [None] + list(names.keys())
image = "./felix.jpeg"
savepath = Path("./heatmaps/")
cam_type = ["gradcam", "gradcam_plusplus", "eigengradcam"]
images = [str(file) for file in Path("./heatmap_src/").glob("**/*")]
model_names = ["ResNet18", "ResNet34", "ResNet50", "ResNet152", "MobileNetV3"]


def res18():
    rn50 = resnet18(pretrained=True)
    rn_target_layer = rn50.layer4[-1]
    return rn50, rn_target_layer


def res34():
    rn50 = resnet34(pretrained=True)
    rn_target_layer = rn50.layer4[-1]
    return rn50, rn_target_layer


def res50():
    rn50 = resnet50(pretrained=True)
    rn_target_layer = rn50.layer4[-1]
    return rn50, rn_target_layer


def res152():
    rn50 = resnet152(pretrained=True)
    rn_target_layer = rn50.layer4[-1]
    return rn50, rn_target_layer


def mobilenet():
    model = mobilenetv3.mobilenet_v3_large(pretrained=True, progress=True)
    target_layer = model.features[-1]
    return model, target_layer


models = {
    "ResNet18": res18,
    "ResNet34": res34,
    "ResNet50": res50,
    "ResNet152": res152,
    "MobileNetV3": mobilenet
}
from skimage import img_as_ubyte
for target_category in tqdm(target_categories):
    for cam in cam_type:
        for model_name in model_names:
            sp = (savepath / cam / model_name)
            sp.mkdir(exist_ok=True, parents=True)
            model, target_layer = models[model_name]()
            for i, img in enumerate(images):
                r1 = sp / f"heatmap_{Path(img).name}" if target_category is None else sp / f"heatmap_{names[target_category]}_{Path(img).name}"
                r2 = sp / f"original{Path(img).name}" if target_category is None else sp / f"original_{names[target_category]}_{Path(img).name}"
                viz, im = gradcam_img(model, img, target_layer, cam, target_category)
                #imshow(viz)

                imsave(r1, img_as_ubyte(viz))
                imsave(r2, img_as_ubyte(im))
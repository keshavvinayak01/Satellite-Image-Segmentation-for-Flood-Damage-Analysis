from fastai.vision.all import *
from fastai.data.all import *
from fastai.imports import *
from fastai.vision.core import *
from fastai import metrics
from scipy import ndimage


model_build = null
model_flood = null

# Add these two as per deployement
PATCH_BUILD = ''
PATCH_FLOODED = ''

def get_model():
    model_build = load_learner(Path(PATCH_BUILD),cpu = True)
    model_flood = load_learner(Path(PATCH_FLOODED),cpu = True)

# Main function that would return the two final image to be displayed
def get_prediction(vimage_path):
    if not (model_build and model_flood):
        return (None, None)
    # Input should ideally be the image path as per pre-defined code, change if required
    im = PILImage.create(vimage_path)

    """
        Following two lines were added for comparison's sake I guess?
        mask1 = PILMask.create("/content/drive/My Drive/E246000N3305300UTM15R/buildings1m.tif")
        mask2 = PILMask.create("/content/drive/My Drive/E246000N3305300UTM15R/flooded1m.tif")
    """
    
    z = im.reshape(1024,1024,resample=Image.BICUBIC)
    z = image2tensor(z)
    z1 = z[:,0:512,0:512]
    z2 = z[:,0:512,512:1024]
    z3 = z[:,512:1024,0:512]
    z4 = z[:,512:1024,512:1024]

    z1 = z1.permute((1,2,0))
    z2 = z2.permute((1,2,0))
    z3 = z3.permute((1,2,0))
    z4 = z4.permute((1,2,0))

    im1 = PILImage.create(z1.cpu().numpy())
    im2 = PILImage.create(z2.cpu().numpy())
    im3 = PILImage.create(z3.cpu().numpy())
    im4 = PILImage.create(z4.cpu().numpy())

    # x is Building Mask Prediction
    x1 = torch.cat([model_build.predict(im1)[0],model_build.predict(im2)[0]],dim=1)
    x2 = torch.cat([model_build.predict(im3)[0],model_build.predict(im4)[0]],dim=1)
    x = torch.cat([x1,x2],dim=0)
    
    # y is Flood Damage Mask Prediction    
    y1 = torch.cat([model_flood.predict(im1)[0],model_flood.predict(im2)[0]],dim=1)
    y2 = torch.cat([model_flood.predict(im3)[0],model_flood.predict(im4)[0]],dim=1)
    y = torch.cat([y1,y2],dim=0)
    return (x, y)
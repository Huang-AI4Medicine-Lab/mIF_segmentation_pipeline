import numpy as np 
from skimage import measure, segmentation, feature, morphology
from scipy import ndimage as ndi
from tqdm import tqdm
import matplotlib.pyplot as plt
from cellpose import models

def cellpose_segmentation(img: np.ndarray, min_size=10, gpu=False, diam=None, channel_cellpose=[0,0], model_name='nuclei', normalize=False) -> np.ndarray:
    model = models.Cellpose(gpu=gpu, model_type=model_name, net_avg=True)
    res, _, _, _ = model.eval(
        img,
        channels=channel_cellpose,
        diameter=diam,
        min_size=min_size,
        normalize=normalize,
        tile=True
    )
    return res


def custom_cellpose_segmentation(img, model_path, min_size=10, gpu=False, diam=None, channel_cellpose=[0,0], normalize=False):
    model = models.CellposeModel(gpu=gpu, pretrained_model=model_path)
    res, _, _ = model.eval(
        img,
        channels=channel_cellpose,
        diameter=diam,
        min_size=min_size,
        batch_size=1, 
        normalize=normalize,
        tile=True
    )
    return res


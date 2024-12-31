import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image

def img_to_numpy(img):
    """
    """
    arr = torch.clip(0.224 * img + 0.45, 0, 1).permute(1, 2, 0).numpy()
    return arr


class DataAugmentationDINO(object):
    """Create crops of an input image together with additional augmentations.

    It generates 2 global crops and `n_local_crops` local crops of an input image.

    Attributes:
        global_1 (transforms.Compose): The first global transform.
        global_2 (transforms.Compose): The second global transform.
        local (transforms.Compose): The local transforms.

    """

    def __init__(
        self, 
        global_crop_scale: tuple = (0.4, 1.0), 
        local_crop_scale: tuple = (0.05, 0.4), 
        n_local_crops: int = 8, 
        size: int = 224):
        """
        Args:
            global_crop_scale (tuple[int]): The minumum and maximum percentage of the 
                original image to include in the global crops.
            local_crop_scale (tuple[int]): The minumum and maximum percentage of the 
                original image to include in the local crops.
            n_local_crops (int): The number of local crops to generate.
            size (int): The pixel length of the final, square image.
        """

        self.n_local_crops = n_local_crops

        # --------------------------------------------------------------
        # Define composite transforms
        # --------------------------------------------------------------

        flip_and_jitter = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply(
                    [
                        transforms.ColorJitter(
                            brightness=0.4,
                            contrast=0.4,
                            saturation=0.2,
                            hue=0.1
                        ),
                    ]
                ),
                transforms.RandomGrayscale(p=0.2),
            ]
        )

        normalize = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

        # --------------------------------------------------------------
        # Define attribute transforms
        # --------------------------------------------------------------

        self.global_1 = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    size=size,
                    scale=global_crop_scale,
                    interpolation=Image.BICUBIC,
                ),
                flip_and_jitter,
                self.RandomGaussianBlur(p=1.0),
                normalize,
            ],
        )

        self.global_2 = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    size=size,
                    scale=global_crop_scale,
                    interpolation=Image.BICUBIC,
                ),
                flip_and_jitter,
                self.RandomGaussianBlur(p=0.1),
                transforms.RandomSolarize(threshold=170, p=0.2),
                normalize,
            ],
        )

        self.local = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    size=size,
                    scale=local_crop_scale,
                    interpolation=Image.BICUBIC,
                ),
                flip_and_jitter,
                self.RandomGaussianBlur(p=0.5),
                normalize,
            ],
        )

        return
    
    def __call__(self, img, as_numpy=False):
        """
        Args:
            img (PIL.Image): The input image.

        Returns:
            all_crops (list): A list of `torch.Tensor` representing different views 
                of the input `img`.
        """

        all_crops = []
        all_crops.append(self.global_1(img))
        all_crops.append(self.global_2(img))

        for _ in range(self.n_local_crops):
            all_crops.append(self.local(img))

        if as_numpy:

            pass


        return all_crops

    def RandomGaussianBlur(
            self,
            p: float = 0.5,
    ):
        """Apply random Gaussian blur to the tensor.

        Args:
            p (float): The probability of applying the transform.
        """

        return transforms.RandomApply(
            [transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))], 
            p=p
        )

        return
    
class Head(nn.Module):
    """
    """

    def __init__(
        self,
        in_dim: int, 
        out_dim: int, 
        hidden_dim: int = 512, 
        bottleneck_dim: int = 256,
        n_layers: int = 3 ,
        norm_last_layer: bool = False,
    ):
        """
        Args:
            in_dim (int): The input dimension.
            out_dim (int): The output dimension.
            hidden_dim (int): The hidden dimension.
            n_layers (int): The number of layers.
            norm_last_layer (bool): Whether to apply normalization to the last layer.

        Returns:
            
        """

        super().__init__()

        if n_layers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim)
        
        else:
            layers = [
                nn.Linear(in_dim, hidden_dim),
                nn.GELU(),
            ]

            for _ in range(n_layers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = nn.Sequential(*layers)

        self.apply(self._init_weights)

        self.last_layer = nn.utils.weight_norm(
            nn.Linear(bottleneck_dim, out_dim, bias=False)
        )


        return
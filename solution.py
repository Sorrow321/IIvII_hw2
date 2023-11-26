from PIL import Image as PILImage
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path
import torchvision.transforms as transforms
import torch

from dalle2_laion import ModelLoadConfig, DalleModelManager
from dalle2_laion.scripts import InferenceScript


class ImageDS(Dataset):
    def __init__(self, img_path, text_path):
        self.images_highres = []
        self.images_lowres = []
        self.texts = []
        transform_highrew = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.PILToTensor()
        ])
        transform_lowrew = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.PILToTensor()
        ])
        for img_p in Path(img_path).iterdir():
            img = PILImage.open(img_p)

            img_tensor = transform_highrew(img)
            img_tensor = img_tensor.float()
            img_tensor /= 255.0  # normalize
            self.images_highres.append(img_tensor)

            img_tensor = transform_lowrew(img)
            img_tensor = img_tensor.float()
            img_tensor /= 255.0  # normalize
            self.images_lowres.append(img_tensor)

            with open(Path(text_path) / (img_p.stem + '.txt'), 'rt') as file:
                self.texts.append([line.strip() for line in file.readlines()])

    def __getitem__(self, idx):
        return self.images_highres[idx], self.images_lowres[idx], self.texts[idx]

    def __len__(self):
        return len(self.images_highres)


def calc_rgb_psnr(pred: np.ndarray, target: np.ndarray) -> int:
    mse = ((pred - target) ** 2).mean()
    return 20 * np.log(255) / np.log(10) - 10 * np.log(mse) / np.log(10)


class ExampleInference(InferenceScript):
    def run(self, lowres_src_image, highres_src_image, text: str):
        text = [text]
        image_embedding_map = self._sample_prior(text)
        image_embedding = image_embedding_map[0][0] 
        image_embedding = image_embedding.unsqueeze(0)
        image_map = self._sample_decoder(text=text, src_lowres_img=lowres_src_image, src_highres_img=highres_src_image, image_embed=image_embedding)
        return image_map[0][0]


model_config = ModelLoadConfig.from_json_path("configs/pipeline.json")
model_manager = DalleModelManager(model_config)
inference = ExampleInference(model_manager)
ds = ImageDS(img_path='img_for_test/images', text_path='img_for_test/texts')

avg_metric = 0
for idx, (src_image_highres, src_image_lowres, src_text) in enumerate(ds):
    src_image_lowres = src_image_lowres.unsqueeze(0)
    src_image_highres = src_image_highres.unsqueeze(0)

    src_image_lowres = src_image_lowres.cuda()
    src_image_highres = src_image_highres.cuda()
    image = inference.run(text=src_text[0], lowres_src_image=src_image_lowres, highres_src_image=src_image_highres)

    metric_res = calc_rgb_psnr(transforms.PILToTensor()(image).numpy(), (src_image_highres * 255).to(torch.uint8).cpu().numpy())
    print(f'Image {idx}. RGB PSNR value: {metric_res}')
    avg_metric += metric_res
avg_metric /= len(ds)
print(f'Average RGB PSNR over {len(ds)} images: {avg_metric}')

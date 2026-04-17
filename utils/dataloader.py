import os
import random

from PIL import Image

from torch.utils.data import Dataset
from torchvision.transforms import transforms

class CelebA(Dataset):
    def __init__(self, imgs_path: str):
        super().__init__()
        self.imgs_path = imgs_path
        self.imgs: list[str] = [f for f in sorted(os.listdir(imgs_path)) if f.endswith(".jpg")]
        self.preprocess = transforms.Compose([
            transforms.Resize(size=(128, 128)),
            transforms.ToTensor(),
        ])    

    def __getitem__(self, index: int):
        img_name: str = self.imgs[index]
        img_path: str = os.path.join(self.imgs_path, img_name)

        try:
            img  = Image.open(img_path)
        except Exception as e:
            print(f"ERROR: Failed to load image: {self.imgs[index]} with error: {e}")
            new_index: int = random.randint(0, self.__len__() - 1)
            return self.__getitem__(new_index)

        return self.preprocess(img)

    
    def __len__(self):
        return len(self.imgs)
    
if __name__ == "__main__":
    from torch.utils.data import DataLoader

    data = CelebA(imgs_path="../data")
    dataloader = DataLoader(dataset=data, batch_size=2, num_workers=1, shuffle=True)
    img = next(iter(dataloader))
    print(f"Image shape: {img.shape} -> [batch_size, color_channels, height, width]")


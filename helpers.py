from torchvision.transforms import InterpolationMode
from torchvision.transforms import transforms
import os, cv2
from torch.utils.data import Dataset, DataLoader

class SRDataset(Dataset):
    def __init__(self, dataset_path, limit = -1, _transforms = None, hr_sz = 128, lr_sz = 32) -> None:
        super().__init__()
        
        self.transforms = _transforms
        
        if not self.transforms:
            self.transforms = transforms.Compose([
                transforms.ToTensor(),
                transforms.RandomHorizontalFlip(p = 0.5),
                transforms.ColorJitter([0.5, 1]),
                transforms.RandomAdjustSharpness(1.1, p = 0.4),
                transforms.Normalize((0.5, ), (0.5,)) # normalizing image with mean, std = 0.5, 0.5
            ])

        self.hr_sz, self.lr_sz = transforms.Resize((hr_sz, hr_sz), interpolation=InterpolationMode.BICUBIC), transforms.Resize((lr_sz, lr_sz), interpolation=InterpolationMode.BICUBIC)
        
        self.dataset_path, self.limit = dataset_path, limit
        self.valid_extensions = ["jpg", "jpeg", "png", "JPEG", "JPG"]
        
        self.images_path = dataset_path
        self.images = os.listdir(self.images_path)[:self.limit]

        self.images = [os.path.join(self.images_path, image) for image in self.images if image.split(".")[-1] in self.valid_extensions]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = cv2.imread(self.images[index])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transforms(image)
        hr_image, lr_image = self.hr_sz(image), self.lr_sz(image)

        # the core idea here is resizing the (128, 128) down to a lower resolution and then back up to (128, 128)
        return self.hr_sz(lr_image), hr_image # the hr_image is 'y' and low res image scaled to (128, 128) is our 'x' 


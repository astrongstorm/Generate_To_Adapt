import usps
import torch
import torchvision
from torchvision import transforms

image_directory = 'digits/usps/'
transform_usps = transforms.Compose([transforms.Grayscale(3), transforms.ToTensor()])

for i in ['trainset', 'testset']:
    dir = image_directory +'/' + i
    if i == 'trainset':
        train_val = True
    else:
        train_val = False
    dset = usps.USPS(root=image_directory, train=train_val, transform=transform_usps, download=True)
    dset_loader = torch.utils.data.DataLoader(dset, batch_size=1, shuffle=False, num_workers=2, drop_last=True)
    iter_loader = iter(dset_loader)
    len_loader = len(dset_loader)
    for img_num in range(len_loader):
        print(i)
        image, label = next(iter_loader)
        image_name = dir + '/' + str(label.item() + 1) + '/' + str(img_num) + '.png'
        print(image_name)
        torchvision.utils.save_image(image, image_name)
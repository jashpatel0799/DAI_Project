import torch
import copy
# from torchvision import datasets

def add_poison(dataset, num_class):
  data = []
  copy_dataset = copy.deepcopy(dataset)
  for image, label in copy_dataset:
    # noise in image
    noise_image = copy.deepcopy(image)
    noise_image += torch.randn_like(noise_image)*torch.rand(1).item()
    data.append((noise_image,label))


    # Miss Label
    num_class = 101
    n_label = torch.randint(0, num_class, (1,)).item()
    data.append((image, n_label))
    
    # balck and white patch in image
    bw_image = copy.deepcopy(image)
    size = image.shape[1]

    mask = torch.zeros_like(bw_image)

    p1 = torch.randint(0, size-size//8, (1,)).item()
    p2 = torch.randint(0, size-size//8, (1,)).item()

    r_int = torch.randint(0, 2, (1,)).item()

    if r_int == 0:

      mask[:,p1:p1+size//8, p2:p2+size//8] = 1
      bw_image += mask

    else:

      bw_image[:,p1:p1+size//8, p2:p2+size//8] = 0

    data.append((bw_image,label))

    # color patch in image
    color_images = copy.deepcopy(image)
    size = image.shape[1]

    p1 = torch.randint(0, size-size//8, (1,)).item()
    p2 = torch.randint(0, size-size//8, (1,)).item()

    patch_loc = color_images[:,p1:p1+size//8, p2:p2+size//8]
    color_images[:,p1:p1+size//8, p2:p2+size//8] = torch.randn_like(patch_loc)

    data.append((color_images,label))

  avd_dataset = torch.utils.data.ConcatDataset([data])
  # print(len(noise_dataset), len(datasets))
  final_dataset = torch.utils.data.ConcatDataset([avd_dataset, dataset])
  # print(len(final_dataset))
  return final_dataset

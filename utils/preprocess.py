import torchvision.transforms as transforms

__imagenet_stats = {'mean': [0.485, 0.456, 0.406],
                   'std': [0.229, 0.224, 0.225]}


def get_transform():
    t_list = [
        transforms.ToTensor(),
        transforms.Normalize(**__imagenet_stats),
    ]

    return transforms.Compose(t_list)

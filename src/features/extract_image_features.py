from torchvision import transforms as T
from torchvision.models import resnet50

def get_resnet50_transform():
    

    IMAGENET_MEAN = [0.485,0.456,0.406]
    IMAGENET_STD = [0.229,0.224,0.225]

    transform = T.Compose([ 
        T.Resize(256),
        T.CenterCrop(224,224),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN,std=IMAGENET_STD),   
    ])

    return transform



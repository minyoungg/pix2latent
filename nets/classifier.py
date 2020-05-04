import torch
from torchvision import models, transforms


class Classifier():
    def __init__(self):
        self.model = models.resnext101_32x8d(pretrained=True)
        self.model.float().cuda().eval()
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        return

    def prep(self, im, is_tensor=True):
        if is_tensor:
            im = ((im + 1.0) / 2.0)
        else:
            im = self.transform(im).unsqueeze(0)
        return im.float().cuda()

    def __call__(self, x, is_tensor=True, as_onehot=False, top5=False):
        with torch.no_grad():
            x = self.prep(x, is_tensor=is_tensor)

            with torch.no_grad():
                x = self.model(x)[0]

            if top5:
                assert not as_onehot
                return torch.topk(torch.softmax(x, dim=0), 5, dim=0)

            x = torch.argmax(x).item()

            if as_onehot:
                o = torch.zeros(1, 1000)
                o[:, x] = 1.0
                return o
        return x

import yaml
import argparse
import requests
from PIL import Image
import torch
import torchvision.transforms as T
from transformers import AutoModel, AutoProcessor
from ren import REN, XREN


device = 'cuda' if torch.cuda.is_available() else 'cpu'


def test_ren():
    parser = argparse.ArgumentParser()
    parser.add_argument('--feature_extractor', type=str, required=True,
                        help='Name of the feature extractor (e.g., dinov2_vitl14).')
    args = parser.parse_args()
    with open(f'configs/ren_{args.feature_extractor}.yaml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # Fetch input image
    url = 'http://images.cocodataset.org/train2017/000000156045.jpg'
    image = Image.open(requests.get(url, stream=True).raw)
    transforms = T.Compose([T.Resize((config['parameters']['image_resolution'], config['parameters']['image_resolution'])), 
                            T.ToTensor()])
    image = transforms(image)
    image = image.unsqueeze(0).to(device)
    
    # Load REN
    ren = REN(config)

    # Process the image
    region_tokens = ren(image)
    print('Region tokens shape: ', region_tokens[0].shape)


def test_xren():
    with open('configs/xren_siglip_vitg16.yaml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # Fetch input image
    url = 'http://images.cocodataset.org/train2017/000000156045.jpg'
    image = Image.open(requests.get(url, stream=True).raw)
    transforms = T.Compose([T.Resize((config['parameters']['image_resolution'], config['parameters']['image_resolution'])), T.ToTensor()])
    image = transforms(image)
    image = image.unsqueeze(0).to(device)

    # Define candidate texts
    texts = ['This is a photo of dog.', 'This is a photo of cat.', 'This is a photo of tie.', 'This is a photo of couch.',
             'This is a photo of man.', 'This is a photo of an apple.', 'This is a photo of 2 cats.']

    # Load XREN
    xren = XREN(config)

    # Get image embeds
    region_tokens = xren(image)
    image_embeds = region_tokens[0].unsqueeze(0)
    print('Region tokens shape: ', region_tokens[0].shape)

    # Get text embeds
    model_name = 'google/siglip2-giant-opt-patch16-384'
    model = AutoModel.from_pretrained(model_name).to(device)
    processor = AutoProcessor.from_pretrained(model_name)
    text_inputs = processor(text=texts, padding='max_length', max_length=64, return_tensors='pt')
    text_outputs = model.text_model(input_ids=text_inputs['input_ids'].to(device))
    text_embeds = text_outputs.pooler_output

    # Get predictions
    image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
    text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)
    logits_per_text = torch.matmul(text_embeds, image_embeds[0].t()).max(dim=1)[0].unsqueeze(-1)
    logit_scale, logit_bias = model.logit_scale.to(text_embeds.device), model.logit_bias.to(text_embeds.device)
    logits_per_text = logits_per_text * logit_scale.exp() + logit_bias
    logits_per_image = logits_per_text.t() 
    probs = torch.sigmoid(logits_per_image)
    for i in range(len(texts)):
        print(f'{texts[i]} \t {probs[0][i] * 100:.3f}%')


if __name__ == '__main__':
    test_xren()
import torch
from utils.vqvae import Decoder, CompressorConfig
import cv2

#Input: [1,129] -> Img
def decode(input_tensor, device):
    config = CompressorConfig()
    with torch.device('cpu'):
        decoder = Decoder(config)
        decoder.load_state_dict_from_url('https://huggingface.co/commaai/commavq-gpt2m/resolve/main/decoder_pytorch_model.bin', assign=True)
        decoder = decoder.eval().to(device='cpu')
    
    # Ignore the BOS token
    target = input_tensor[:, 1:].to(torch.int64).to(device='cpu')

    output = decoder(target)
    output = output[0]
    # Convert to float32 before converting to numpy
    img = output.permute(1, 2, 0).to(torch.float32).cpu().detach().numpy().astype('uint8')
    return img

# if __name__ == "__main__":
#     input_tensor = torch.randint(0, 256, (1, 129))
#     device = torch.device('cpu')
#     img = decode(input_tensor, device)
#     print(img.shape)
#     #save as png
#     cv2.imwrite("output.png", img)

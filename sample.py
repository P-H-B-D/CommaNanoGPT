"""
Sample from a trained model, working directly with token IDs
"""
import os
import torch
from contextlib import nullcontext
from model import GPTConfig, GPT
import numpy as np
import imageio
from decode import decode

# -----------------------------------------------------------------------------
init_from = 'resume' # either 'resume' (from an out_dir) or a gpt2 variant
out_dir = 'out' # ignored if init_from is not 'resume'
max_new_tokens = 129 # number of tokens generated in each sample
temperature = 0.8 # 1.0 = no change, < 1.0 = less random, > 1.0 = more random
top_k = 200 # retain only the top_k most likely tokens
seed = 1337
device = 'cpu'  # Force everything to run on CPU
dtype = 'float32'  # Use float32 for CPU operations
compile = False # use PyTorch 2.0 to compile the model
exec(open('configurator.py').read()) # overrides from command line or config file
# -----------------------------------------------------------------------------

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
device_type = 'cuda' if 'cuda' in device else 'cpu'
device = 'cpu'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# model
if init_from == 'resume':
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    model = model.to(device)
elif init_from.startswith('gpt2'):
    model = GPT.from_pretrained(init_from, dict(dropout=0.0))
    model = model.to(device)

model.eval()
if compile:
    model = torch.compile(model)

# prepare the input
data = np.memmap("./nanogpt/0.bin", dtype=np.int16, mode='r')
bos_positions = np.where(data == 1024)[0]
first_valid_position = bos_positions[0]
context_length = 129 * 19  # First 19 images
ground_truth_length = 129  # One more image for ground truth
total_length = context_length + ground_truth_length

# Get both context and ground truth data, ensuring they're on the correct device from the start
x = torch.tensor(data[first_valid_position:first_valid_position+context_length].astype(np.int64), device=device)
ground_truth = torch.tensor(data[first_valid_position+context_length:first_valid_position+total_length].astype(np.int64), device=device)

# Apply the same transformation to both
x = torch.where(x == 1025, torch.tensor(1024), x)
ground_truth = torch.where(ground_truth == 1025, torch.tensor(1024), ground_truth)

x = x.unsqueeze(0)  # Add batch dimension
ground_truth = ground_truth.unsqueeze(0)  # Add batch dimension

model = model.to('cuda' if torch.cuda.is_available() else 'cpu')
# run generation
with torch.no_grad():
    with ctx:
        frames = []  # List to store frames for GIF
        current_context = x.to('cuda')  # Move initial context to GPU

        for frame_idx in range(6):  # Generate 6 frames in total
            print(frame_idx)
            # Ensure the model and context are on the GPU
            y = model.generate(current_context, max_new_tokens, temperature=temperature, top_k=top_k)
            
            # Move the generated tensor to CPU for decoding
            y_generated = y[:, current_context.shape[1]:current_context.shape[1]+129].to('cpu')
            
            # Decode and store the generated frame
            img = decode(y_generated, 'cpu')  # Ensure decode uses CPU
            # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            frames.append(img)
            
            # Update the context using a sliding window, move back to GPU
            current_context = torch.cat((current_context[:, 129:], y_generated.to('cuda')), dim=1)

        # Save frames as a GIF
        gif_path = "output.gif"
        imageio.mimsave(gif_path, frames, format='GIF', duration=0.5)  # Save as GIF with a duration of 0.5 seconds per frame

        print("GIF saved as:", gif_path)




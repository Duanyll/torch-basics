import os
import sys
from torch.utils.data import DataLoader
from ddpm.model import DDPM
from ddpm.data import get_celeba_hq


if __name__ == '__main__':
    # Load the data
    celeba = get_celeba_hq()
    dataloader = DataLoader(celeba, batch_size=32, shuffle=True)
    
    # Initialize the model
    ddpm = DDPM()
    if len(sys.argv) > 1:
        outdir = sys.argv[1]
    else:
        outdir = 'runs/ddpm'
        
    resume = os.path.exists(outdir)
    
    # Train the model
    ddpm.train(dataloader, outdir, resume=resume)
    
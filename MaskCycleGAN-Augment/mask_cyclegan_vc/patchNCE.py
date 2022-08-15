import torch
import torch.nn.functional as F
from mask_cyclegan_vc.model import Generator

def patchNCE(inp,out,generator):
    layers_in = generator.intermediate_outputs(inp)
    layers_out = generator.intermediate_outputs(out)
    loss = torch.tensor(0,dtype=inp.dtype,device=inp.device)
    for layer_in,layer_out in zip(layers_in,layers_out):
        layer_in = F.normalize(layer_in,dim=1)
        layer_out = F.normalize(layer_out,dim=1)
        dot_matrix = torch.exp(torch.bmm(torch.transpose(layer_out,1,2),layer_in))
        positive_diagonal = torch.eye(dot_matrix.size(1),device=dot_matrix.device).unsqueeze(0)*dot_matrix
        positive_scores = torch.sum(positive_diagonal,dim=2)
        overall_scores = torch.sum(dot_matrix,dim=2)
        loss += -torch.mean(torch.log(positive_scores/overall_scores))/len(layers_in)
    return loss


if __name__ == '__main__':
    gen = Generator()
    x = torch.randn(5,80,64)
    y= torch.randn(5,80,64)
    loss = patchNCE(x,y,gen)
    print(loss.item())

import torch
import torchvision
from torchvision import transforms
from diffusers import UNet2DModel
from torch.optim import NAdam
from torch.optim import Adam
import copy


def loss_function(x):
    a=-60000
    b=4
    return ((x/b+a)**6+(x/b+a)**5+3*(x/b+a)**3+4*(x/b+a)**2+(x/b+a))

def get_model():
    block_out_channels=(128, 128, 256, 256, 512, 512)
    down_block_types=( 
        "DownBlock2D",  # a regular ResNet downsampling block
        "DownBlock2D", 
        "DownBlock2D", 
        "DownBlock2D", 
        "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
        "DownBlock2D",
    )
    up_block_types=(
        "UpBlock2D",  # a regular ResNet upsampling block
        "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
        "UpBlock2D", 
        "UpBlock2D", 
        "UpBlock2D", 
        "UpBlock2D"  
    )
    return UNet2DModel(block_out_channels=block_out_channels,out_channels=3, in_channels=3, up_block_types=up_block_types, down_block_types=down_block_types, add_attention=True)

@torch.no_grad()
def sample_iadb(model, x0, nb_step):
    x_alpha = x0
    for t in range(nb_step):
        alpha_start = (t/nb_step)
        alpha_end =((t+1)/nb_step)

        d = model(x_alpha, torch.tensor(alpha_start, device=x_alpha.device))['sample']
        x_alpha = x_alpha + (alpha_end-alpha_start)*d

    return x_alpha


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
CELEBA_FOLDER = './datasets/cifar/'
transform = transforms.Compose([transforms.Resize(32),transforms.CenterCrop(32), transforms.RandomHorizontalFlip(0.5),transforms.ToTensor()])
train_dataset = torchvision.datasets.CIFAR10(root=CELEBA_FOLDER,
                                        download=True, transform=transform)

dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=0, drop_last=True)

root_folder="C:/Users/Meenakshi Manikandan/testing_codm_lf/"
model = get_model()
model = model.to(device)

model2=get_model()
model2=model2.to(device)
from_dict=model.state_dict()
to_dict=model2.state_dict()
for i in to_dict.keys():
    to_dict[i]=copy.deepcopy(from_dict[i])
model2.load_state_dict(to_dict)




losses=[]
loss_pairs=[]
chosen_models=[]
optimizer = NAdam(model.parameters(), lr=1e-4)
optimizer2 = Adam(model2.parameters(), lr=1e-4)







nb_iter = 0
print('Start training')
chosen_model=None
for current_epoch in range(7000):
    for i, data in enumerate(dataloader):
        x1 = (data[0].to(device)*2)-1
        x0 = torch.randn_like(x1)
        bs = x0.shape[0]

        alpha = torch.rand(bs, device=device)
        x_alpha = alpha.view(-1,1,1,1) * x1 + (1-alpha).view(-1,1,1,1) * x0
        
        d = model(x_alpha, alpha)['sample']
        d2 = model2(x_alpha, alpha)['sample']


        loss = loss_function(torch.sum((d - (x1-x0))**2))
        loss2 = loss_function(torch.sum((d2-(x1-x0))**2))

        """print(loss.item(), loss2.item())
        if i==5:
          import time
          time.sleep(30)"""

        #loss_pairs.append((loss, loss2))

        use_this=None
        if(loss<loss2):
            chosen_model="1"
            use_this=loss
            chosen_models.append(chosen_model)
            losses.append(use_this)


            from_dict=model.state_dict()
            to_dict=model2.state_dict()
            for i in to_dict.keys():
               to_dict[i]=copy.deepcopy(from_dict[i])
            model2.load_state_dict(to_dict)

            if i!=0:
              from_dict=optimizer.state_dict()
              to_dict=optimizer2.state_dict()
              for i in to_dict['state'].keys():
                to_dict['state'][i]['step']=copy.deepcopy(from_dict['state'][i]['step']) 
                to_dict['state'][i]['exp_avg']=copy.deepcopy(from_dict['state'][i]['exp_avg'])
                to_dict['state'][i]['exp_avg_sq']=copy.deepcopy(from_dict['state'][i]['exp_avg_sq'])
                #to_dict['state'][i]['max_exp_avg_sq']=copy.deepcopy(from_dict['state'][i]['max_exp_avg_sq'])
              optimizer2.load_state_dict(to_dict)

            optimizer.zero_grad()
            use_this.backward()
            optimizer.step()

            optimizer2.zero_grad() 
            count=0
            f2=list(model.parameters())
            for p in model2.parameters():
                 p.grad=f2[count].grad.clone()
                 f2[count].grad=None  #because keeping both models' gradients makes cuda run out of memory
                 count+=1
            optimizer2.step()
            optimizer2.zero_grad() 

        else:
           chosen_model="2"
           use_this=loss2
           chosen_models.append(chosen_model)
           losses.append(use_this)


           from_dict=model2.state_dict()
           to_dict=model.state_dict()
           for i in to_dict.keys():
              to_dict[i]=copy.deepcopy(from_dict[i])
           model.load_state_dict(to_dict)

           if i!=0:
             from_dict=optimizer2.state_dict()
             to_dict=optimizer.state_dict()
             for i in to_dict['state'].keys():
               to_dict['state'][i]['step']=copy.deepcopy(from_dict['state'][i]['step']) 
               to_dict['state'][i]['exp_avg']=copy.deepcopy(from_dict['state'][i]['exp_avg'])
               to_dict['state'][i]['exp_avg_sq']=copy.deepcopy(from_dict['state'][i]['exp_avg_sq'])
               #to_dict['state'][i]['max_exp_avg_sq']=copy.deepcopy(from_dict['state'][i]['max_exp_avg_sq'])
             optimizer.load_state_dict(to_dict)


           optimizer2.zero_grad()
           use_this.backward()
           optimizer2.step()
             
             
             
            
             
             
           optimizer.zero_grad()
             #model1 doesn't have grad so instead of backward we copy the weights' grad
           count=0
           f2=list(model2.parameters())
           for p in model.parameters():
                 p.grad=f2[count].grad.clone()
                 f2[count].grad=None  #because keeping both models' gradients makes cuda run out of memory
                 count+=1
           optimizer.step()
           optimizer.zero_grad() #to clear up more memory
             
             #to allow optimizer1 to update we must copy the gradients also
             #for g1,g2 in zip(model.parameters(),model2.parameters()):
             #    g1.grad=g2.grad
             

             
           


          
        
        
        #if nb_iter==600:
        #    assert(0==1)
        
        nb_iter += 1

        if ((nb_iter % 5000 == 0) or (nb_iter==99800)):
            with torch.no_grad():
                print(f'Save export {nb_iter}')
                sample = (sample_iadb(model, x0, nb_step=128) * 0.5) + 0.5
                torchvision.utils.save_image(sample, root_folder+"image"+str(nb_iter)+".png")
                torch.save(sample, root_folder+"gen_tensor"+str(nb_iter))
                torch.save(losses, root_folder+"ls"+str(nb_iter))

                #torch.save(loss_pairs, root_folder+"lp"+str(nb_iter))
                torch.save(chosen_models, root_folder+"chosen"+str(nb_iter))
                torch.save(model.state_dict(), root_folder+"model"+str(nb_iter))
                torch.save(optimizer.state_dict(), root_folder+"optimizer"+str(nb_iter))

                torch.save(model2.state_dict(), root_folder+"second_model"+str(nb_iter))
                torch.save(optimizer2.state_dict(), root_folder+"second_optimizer"+str(nb_iter))

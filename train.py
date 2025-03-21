from io_utils import *
from simulation_utils import *
from learning_utils import *
torch.manual_seed(123)
import sys
import os
from functorch import jacrev, vmap
from torch.autograd import grad
#1.0
w_FP = 1.0
w_VP = 1.0
w_vorticity = 1.0
w_size = 1.0
w_len = 0
#w_len_value = 1.0
w_len_0D = 1.0
w_len_1D = 0
num_pretrain = 10000
eps_vor = 1e-3
eps_dist = 1e-6
coeff_adaptive_weights = 1.1

rhat1 = (torch.linspace(0, np.sqrt(1.12), 101)**2).to(device)
rhat2 = (torch.linspace(np.sqrt(1.13), np.sqrt(100), 101)**2).to(device)

device = torch.device("cuda")
real = torch.float32

# command line parse
parser = config_parser()
args = parser.parse_args()

nu = args.nu
omega = args.omega

# some switches
run_pretrain = args.run_pretrain # if this is set to true, then only pretrain() will be run
test_only = args.test_only # if this is set to true, then only test() will be run
start_over = args.start_over # if this is set to true, then the logs/[exp_name] dir. will be emptied

# some hyperparameters
print("[Train] Number of training iters: ", args.num_train)
num_iters = args.num_train # total number of training iterations
decimate_point = 20000 # LR decimates at this point
decay_gamma = 0.99
#decay_step = max(1, int(decimate_point/math.log(0.1, decay_gamma))) # decay once every (# >= 1) learning steps
decay_step = 100
save_ckpt_every = 1000
test_every = 1000
print_every = 20
num_sims = 1 # the "m" param in paper
batch_size = 1 # has to be 1

# load data
datadir = os.path.join('data', args.data_name)
print("[Data] Load from path: ", datadir)
imgs = torch.from_numpy(np.load(os.path.join(datadir, 'imgs.npy'))).to(device).type(real)
try:
    sdf = torch.from_numpy(np.load(os.path.join(datadir, 'sdf.npy'))).to(device).type(real)
except:
    print("[Boundary] SDF file doesn't exist, no boundary")
    boundary = None
else:
    print("[Boundary] SDF file exists, has boundary")
    sdf = torch.flip(sdf, [0])
    sdf = torch.permute(sdf, (1, 0))
    sdf_normal = calc_sdf_normal(sdf)
    # 1. signed distance field
    # 2. unit normal of sdf
    # 3. thickness (in pixels) 
    boundary = (sdf, sdf_normal, 2)

num_total_frames = imgs.shape[0] # seen + unseen frames
print("[Data] Number of frames we have: ", num_total_frames)
imgs = imgs[:math.ceil(num_total_frames * args.seen_ratio)] # select a number of frames to be revealed for training
num_frames, width, height, num_channels = imgs.shape
print("[Data] Number of frames revealed: ", num_frames)
num_unseen_frames = num_total_frames - num_frames
print("[Data] Number of frames concealed: ", num_unseen_frames)
#timestamps = torch.arange(num_frames).type(real)[..., None].to(device) * dt
timestamps = torch.arange(num_total_frames).type(real)[..., None].to(device) * dt_
timestamps_training = torch.arange(num_frames).type(real)[..., None].to(device) * dt_
num_available_frames = num_frames - num_sims
probs = torch.ones((num_available_frames), device = device, dtype = real)

rhat1 = rhat1.unsqueeze(-1).to(device)
rhat2 = rhat2.unsqueeze(-1).to(device)
# setup initial vort (as a vorts_num_x X vorts_num_y grid)
# vorts_num_x = 4
# vorts_num_y = 4
# num_vorts = vorts_num_x * vorts_num_y

FP_num_x = 1 # do not use this
FP_num_y = 1 # do not use this
num_FP = FP_num_x * FP_num_y # do not use this

VP_num_x = int(args.VP_num_x) # vortex particles
VP_num_y = int(args.VP_num_y) # vortex particles
num_VP = VP_num_x * VP_num_y # vortex particles

# create some directories
# logs dir
exp_name = args.exp_name
logsdir = os.path.join('logs', exp_name)
print("[Output] Results saving to: ", logsdir)
os.makedirs(logsdir, exist_ok=True)
if start_over:
    remove_everything_in(logsdir)
# folder for tests
testdir = 'tests'
testdir = os.path.join(logsdir, testdir)
os.makedirs(testdir, exist_ok=True)
# folder for ckpts
ckptdir = 'ckpts'
ckptdir = os.path.join(logsdir, ckptdir)
os.makedirs(ckptdir, exist_ok=True)
# folder for pre_trained
pretraindir = 'pretrained'
pretraindir = os.path.join(pretraindir, exp_name)
os.makedirs(pretraindir, exist_ok=True)
if run_pretrain: # if calling pretrain, then remove previous pretrain records
    remove_everything_in(pretraindir)
pre_ckptdir = os.path.join(pretraindir, 'ckpts') # ckpt for pretrain
os.makedirs(pre_ckptdir, exist_ok=True)
pre_testdir = os.path.join(pretraindir, 'tests') # test for pretrain
os.makedirs(pre_testdir, exist_ok=True)

# init or load networks
net_dict, start, grad_vars, optimizer, lr_scheduler = create_bundle(ckptdir, num_FP, num_VP, decay_step, decay_gamma, pretrain_dir = pre_ckptdir)
img_x = gen_grid(width, height, device) # grid coordinates

def eval_vel(vorts_size, vorts_w, vorts_pos, query_pos):
    
    return vort_to_vel(net_dict['model_len'], vorts_size, vorts_w, vorts_pos, query_pos, length_scale = args.vort_scale)

def dist_2_len_(dist):
    return net_dict['model_len'](dist)

def size_pred():
    size_square = net_dict['size_square_pred']
    size = torch.sqrt(size_square)
    return size

def w_pred():    
    w = net_dict['w_pred']
    return w

def comp_FP_velocity(timestamps):
    jac = vmap(jacrev((net_dict['model_FP_pos'])))(timestamps)
    post = jac[:, :, 0:1].view((timestamps.shape[0],-1,2,1))
    xt = post[:, :, 0, :]
    yt = post[:, :, 1, :]
    uv = torch.cat((xt, yt), dim = 2)
    return uv

def comp_VP_velocity(timestamps):
    jac = vmap(jacrev((net_dict['model_VP_pos'])))(timestamps)
    post = jac[:, :, 0:1].view((timestamps.shape[0],-1,2,1))
    xt = post[:, :, 0, :]
    yt = post[:, :, 1, :]
    uv = torch.cat((xt, yt), dim = 2)
    return uv

def comp_vorticity_derivative(timestamps):

    dw_dt = vmap(jacrev(net_dict['w_pred']))(timestamps)

    return dw_dt

def comp_size_square_derivative(timestamps):

    ddelta2_dt = vmap(jacrev(net_dict['size_square_pred']))(timestamps)

    return ddelta2_dt

def comp_model_len_derivative(r):
    
    dlen_dr = vmap(jacrev(net_dict['model_len']))(r)

    return dlen_dr

# pretrain (of the trajectory module)
# the scale parameter influences to the initial positions of the vortices
def pretrain(scale = 1.):
    if start > 0:
        print("[Pretrain] Pretraining needs to be the start of the training pipeline. Please re-run with --start_over set to True.")
        sys.exit()

    with torch.no_grad():
        init_poss = gen_grid(FP_num_x, FP_num_y, device).view([-1, 2])
        init_poss = scale * init_poss + 0.5 * (1.-scale) # scale the initial grid
        FP_pos_GT = init_poss[None, ...].expand(num_frames, -1, -1)
        FP_vel_GT = torch.zeros_like(FP_pos_GT)
        init_poss = gen_grid(VP_num_x, VP_num_y, device).view([-1, 2])
        init_poss = scale * init_poss + 0.5 * (1.-scale) # scale the initial grid
        VP_pos_GT = init_poss[None, ...].expand(num_frames, -1, -1)
        VP_vel_GT = torch.zeros_like(VP_pos_GT)
        
        VP_size_GT = torch.empty(VP_vel_GT.shape[0],VP_vel_GT.shape[1],1).to(device)
        VP_size_GT[:,:,:] = args.vort_scale
        
        VP_w_GT = torch.empty(VP_vel_GT.shape[0],VP_vel_GT.shape[1],1).to(device)
        VP_w_GT[:,:,:] = args.vort_w
    
    for it in range(num_pretrain):
        FP_pos_pred = net_dict['model_FP_pos'](timestamps_training).view(-1, num_FP, 2)
        FP_pos_loss = L2_Loss(FP_pos_pred, FP_pos_GT)         
        VP_pos_pred = net_dict['model_VP_pos'](timestamps_training).view(-1, num_VP, 2)
        VP_pos_loss = L2_Loss(VP_pos_pred, VP_pos_GT) 
        VP_size_square_pred = net_dict['size_square_pred'](timestamps_training).view(-1, num_VP, 1)
        VP_size_pred = torch.sqrt(VP_size_square_pred)
        VP_size_loss = L2_Loss(VP_size_pred, VP_size_GT)
        VP_w_pred = net_dict['w_pred'](timestamps_training).view(-1, num_VP, 1)
        VP_w_loss = L2_Loss(VP_w_pred, VP_w_GT)
        
        model_len_loss_0D = torch.sum(torch.relu(-(net_dict['model_len'](rhat1).view(-1, 1))))**2 + torch.sum(torch.relu(-(net_dict['model_len'](rhat2).view(-1, 1))))**2 + L2_Loss(net_dict['model_len'](torch.tensor([[0.0]]).to(device)).view(-1, 1), torch.tensor([[0.0]]).to(device)) 
        
        model_len_loss_1D = torch.sum(torch.relu(-(comp_model_len_derivative(rhat1))))**2 + torch.sum(torch.relu(comp_model_len_derivative(rhat2)))**2

        FP_vel_pred = comp_FP_velocity(timestamps_training)
        FP_vel_loss = w_FP * L2_Loss(FP_vel_pred, FP_vel_GT)
        VP_vel_pred = comp_VP_velocity(timestamps_training)
        VP_vel_loss = w_VP * L2_Loss(VP_vel_pred, VP_vel_GT)

        loss = FP_pos_loss + VP_pos_loss + FP_vel_loss + VP_vel_loss + VP_size_loss + VP_w_loss + w_len_0D * model_len_loss_0D + w_len_1D * model_len_loss_1D

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if it % 200 == 0:
            print("[Pretrain] Iter: ", it, ", loss: ", loss.detach().cpu().numpy(), "/ FP pos loss: ", FP_pos_loss.detach().cpu().numpy(), "/ FP vel loss: ", FP_vel_loss.detach().cpu().numpy(), "/ VP pos loss: ", VP_pos_loss.detach().cpu().numpy(), "/ VP vel loss: ", VP_vel_loss.detach().cpu().numpy(), "/ VP size loss: ", VP_size_loss.detach().cpu().numpy(), "/ VP w loss: ", VP_w_loss.detach().cpu().numpy(), "/ model len loss 0D: ", model_len_loss_0D.detach().cpu().numpy(), "/ model len loss 1D: ", model_len_loss_1D.detach().cpu().numpy())
    
    # save pretrained results (trajectory module only)
    path = os.path.join(pre_ckptdir, 'pretrained_FP.tar')
    torch.save({
        'model_FP_pos_state_dict': net_dict['model_FP_pos'].state_dict(),
    }, path)
    print('[Pretrain] Saved checkpoint to: ', path)
    
    path = os.path.join(pre_ckptdir, 'pretrained_VP.tar')
    torch.save({
        'model_VP_pos_state_dict': net_dict['model_VP_pos'].state_dict(),
    }, path)
    print('[Pretrain] Saved checkpoint to: ', path)
    
    path = os.path.join(pre_ckptdir, 'pretrained_size_square_pred.tar')
    torch.save({
        'size_square_pred_state_dict': net_dict['size_square_pred'].state_dict(),
    }, path)
    print('[Pretrain] Saved checkpoint to: ', path)
    
    path = os.path.join(pre_ckptdir, 'pretrained_w.tar')
    torch.save({
        'w_pred_state_dict': net_dict['w_pred'].state_dict(),
    }, path)
    print('[Pretrain] Saved checkpoint to: ', path)
    
    path = os.path.join(pre_ckptdir, 'pretrained_model_len.tar')
    torch.save({
        'model_len_state_dict': net_dict['model_len'].state_dict(),
    }, path)
    print('[Pretrain] Saved checkpoint to: ', path)
    
    with torch.no_grad():
        # output all vort positions with velocity
        values = net_dict["model_FP_pos"](timestamps)
        values = values.view([values.shape[0], -1, 2])
        uvs = comp_FP_velocity(timestamps)
        for i in range(values.shape[0]):
            print("[Pretrain] Writing test frame: ", i)
            vorts_pos_numpy = values[i].detach().cpu().numpy()
            vel_numpy = uvs[i].detach().cpu().numpy()
            write_vorts(vorts_pos_numpy, vel_numpy, pre_testdir, i)
            
        values = net_dict["model_VP_pos"](timestamps)
        values = values.view([values.shape[0], -1, 2])
        uvs = comp_VP_velocity(timestamps)
        for i in range(values.shape[0]):
            print("[Pretrain] Writing test frame: ", i)
            vorts_pos_numpy = values[i].detach().cpu().numpy()
            vel_numpy = uvs[i].detach().cpu().numpy()
            write_vorts_1(vorts_pos_numpy, vel_numpy, pre_testdir, i)

    print('[Pretrain] Complete.')


# test learned simulation
def test(curr_it):
    print ("[Test] Testing at iter: " + str(curr_it))
    currdir = os.path.join(testdir, str(curr_it))
    os.makedirs(currdir, exist_ok=True)

    with torch.no_grad():
        total_imgs = [imgs[[0]]]
        total_vels = [None]
        total_vorts = [None]
        total_vorti = [None]
        for i in range(num_frames):
            num_to_sim = 1
            if i == num_available_frames-1:
                num_to_sim += num_sims + max(num_unseen_frames, int(1.5 * num_frames)) -1 # if at the last reveal image, simulate to the end of the video
            VP_pos_pred = net_dict['model_VP_pos'](timestamps[[i]]).view((1,num_VP,2))
            VP_w_pred = net_dict['w_pred'](timestamps[[i]]).view((1,num_VP,1))
            VP_size_square_pred = net_dict['size_square_pred'](timestamps[[i]]).view((1,num_VP,1))
            VP_size_pred = torch.sqrt(VP_size_square_pred)
            sim_imgs, sim_vorts_poss, sim_vels, sim_vorts_vels, sim_vorts_vorti = simulate(total_imgs[-1].clone(), img_x, VP_pos_pred.clone(), \
                                            VP_w_pred.clone(), VP_size_pred.clone(),\
                                            num_to_sim, nu, omega, vel_func = eval_vel, \
                                            boundary = boundary)
            
            total_imgs = total_imgs + sim_imgs
            total_vels = total_vels + sim_vels
            total_vorts = total_vorts + sim_vorts_poss
            total_vorti = total_vorti + sim_vorts_vorti
            
        total_vels_npy = np.empty((len(total_imgs), width, height, 2))
        total_vorti_npy = np.empty((len(total_imgs), width, height, 1))
        total_vels_div_npy = np.empty((len(total_imgs), width, height))
        total_imgs_npy = np.empty((len(total_imgs), width, height, 3))
    
        visdir = os.path.join(currdir, 'particles')
        os.makedirs(visdir, exist_ok=True)        
        imgdir = os.path.join(currdir, 'imgs')
        os.makedirs(imgdir, exist_ok=True)
        vortdir = os.path.join(currdir, 'vorts')
        os.makedirs(vortdir, exist_ok=True)
        write_image(total_imgs[0][0].cpu().numpy(), imgdir, 0) # write init image
        for i in range(1, len(total_imgs)):
            print("[Test] Writing test frame: ", i)
            img = total_imgs[i].squeeze()
            vorti = total_vorti[i].squeeze()
            vorts_pos = total_vorts[i]
            vorts_size = torch.empty(vorts_pos.shape[0],vorts_pos.shape[1],1)
            img_vel = total_vels[i]
            
            img_vel_div_Batch = calc_div(img_vel)
            img_vel_div = torch.zeros(img_vel_div_Batch.shape[1], img_vel_div_Batch.shape[2], 1, device=device, dtype=real)
            img_vel_div = img_vel_div_Batch[0]
            img_vel_div_numpy = img_vel_div.detach().cpu().numpy()
            
            vort_img_Batch = calc_vort(img_vel, boundary)
            vort_img = torch.zeros(vort_img_Batch.shape[1], vort_img_Batch.shape[2], 1, device=device, dtype=real)
            vort_img[:,:,:] = vort_img_Batch[0,:,:,:]
            vort_img_numpy = vort_img.detach().cpu().numpy()
            
            img_numpy = img.detach().cpu().numpy()
            vorts_pos_numpy = vorts_pos.detach().cpu().numpy()
            vorts_w_numpy = vorti.detach().cpu().numpy()
            write_visualization(img_numpy, vorts_pos_numpy, vorts_w_numpy, visdir, i, boundary = boundary)
            write_image(img_numpy, imgdir, i)
            write_vorticity(vort_img_numpy, vortdir, i)
            
            total_vels_npy[i,:,:,:] = img_vel.cpu().numpy()
            total_vorti_npy[i,:,:,:] = vort_img_numpy
            total_vels_div_npy[i,:,:] = img_vel_div_numpy
            total_imgs_npy[i,:,:,:] = img_numpy
            
    # 保存文件
        np.save(os.path.join(imgdir, 'total_vels_npy.npy'), total_vels_npy)
        np.save(os.path.join(imgdir, 'total_vorti_npy.npy'), total_vorti_npy)
        np.save(os.path.join(imgdir, 'total_vels_div_npy.npy'), total_vels_div_npy)
        np.save(os.path.join(imgdir, 'total_imgs_npy.npy'), total_imgs_npy)
        

# # # # #

# if pretrain is True then run pretrain() and quit
if run_pretrain:
    pretrain(args.init_vort_dist)
    sys.exit()

# if test_only is True then run test() and quit
if test_only:
    test(start)
    sys.exit()

# below is training code
prev_time = time.time()
for it in range(start, num_iters):
    # each iter select some different starting frames
    init_frames = probs.multinomial(num_samples = batch_size, replacement = False)

    # compute velocity prescribed by dynamics module
    with torch.no_grad():
        FP_pos_pred_gradless = net_dict['model_FP_pos'](timestamps[init_frames]).view((-1,num_FP,2))
        VP_pos_pred_gradless = net_dict['model_VP_pos'](timestamps[init_frames]).view((-1,num_VP,2))
        w_pred_gradless = net_dict['w_pred'](timestamps[init_frames]).view((-1,num_VP,1))
        size_square_pred_gradless = net_dict['size_square_pred'](timestamps[init_frames]).view((-1,num_VP,1))
        size_pred_gradless = torch.sqrt(size_square_pred_gradless)
        
        vel_func = eval_vel
        img_vel_flattened = vel_func(size_pred_gradless, w_pred_gradless, VP_pos_pred_gradless, img_x.view(-1, 2))
        img_vel = img_vel_flattened.view((batch_size, img_x.shape[0], img_x.shape[1], -1))
        img_vor = calc_vort(img_vel, boundary = boundary)
        img_vor_grad = calc_grad(img_vor)
        vor_grad = bilinear_interpolate(img_vor_grad, VP_pos_pred_gradless)
        vor = bilinear_interpolate(img_vor, VP_pos_pred_gradless)
        
        img_vel_3d = torch.zeros((batch_size, img_x.shape[0], img_x.shape[1], 3), 
                        device=img_vel.device, 
                        dtype=img_vel.dtype)
        img_vel_3d[:,:,:,:2] = img_vel  
        img_omega = torch.zeros_like(img_vor)[:,:,:,:].repeat(1,1,1,3)
        img_omega[:,:,:,2] = omega
        img_b = -2* torch.cross(img_omega, img_vel_3d*args.vort_scale, dim=3)
        img_curl_force = calc_vort(img_b, boundary = boundary)
        curl_force = bilinear_interpolate(img_curl_force, VP_pos_pred_gradless)
        
        D_VP_vorticity = nu*calc_laplacian(vor) + curl_force
        D_VP_size_square = torch.full_like(D_VP_vorticity, 4*nu)
        
        
        D_VP_vel = eval_vel(size_pred_gradless, w_pred_gradless, VP_pos_pred_gradless, VP_pos_pred_gradless)
        
        if boundary is not None:
            D_VP_vel = boundary_treatment(VP_pos_pred_gradless, D_VP_vel, boundary, mode = 1)

    
    T_VP_vel = comp_VP_velocity(timestamps[init_frames]) # velocity prescribed by trajectory module
    VP_vel_loss = L2_Loss(T_VP_vel, D_VP_vel)
    
    vorticity_derivative = comp_vorticity_derivative(timestamps[init_frames]) # velocity prescribed by trajectory module
    VP_vorticity_derivative_loss = L2_Loss(vorticity_derivative, D_VP_vorticity)
    
    size_square_derivative = comp_size_square_derivative(timestamps[init_frames]) # velocity prescribed by trajectory module
    VP_size_square_derivative_loss = L2_Loss(size_square_derivative, D_VP_size_square)
    
    model_len_loss_0D = torch.sum(torch.relu(-(net_dict['model_len'](rhat1).view(-1, 1))))**2 + torch.sum(torch.relu(-(net_dict['model_len'](rhat2).view(-1, 1))))**2 + L2_Loss(net_dict['model_len'](torch.tensor([[0.0]]).to(device)).view(-1, 1), torch.tensor([[0.0]]).to(device)) 
    model_len_loss_1D = torch.sum(torch.relu(-(comp_model_len_derivative(rhat1))))**2 + torch.sum(torch.relu(comp_model_len_derivative(rhat2)))**2
    
    VP_pos_pred = net_dict['model_VP_pos'](timestamps[init_frames]).view((batch_size,num_VP,2))
    VP_w_pred = net_dict['w_pred'](timestamps[init_frames]).view((batch_size,num_VP,1))
    VP_size_square_pred = net_dict['size_square_pred'](timestamps[init_frames]).view((batch_size,num_VP,1))
    VP_size_pred = torch.sqrt(VP_size_square_pred)
    sim_imgs, sim_vorts_poss, sim_img_vels, sim_vorts_vels, sim_vorts_vorti = simulate(imgs[init_frames].clone(), img_x, VP_pos_pred, VP_w_pred, \
                            VP_size_pred, num_sims, nu, omega, vel_func = eval_vel, boundary = boundary)
    sim_imgs = torch.stack(sim_imgs)

    # comp img loss
    img_losses = []
    if boundary is None: # if no boundary then compute loss on entire images
        for i in range(batch_size):
            pred = rgb_to_yuv(sim_imgs[:, i])
            GT = rgb_to_yuv(imgs[init_frames[i]+1: init_frames[i]+1+num_sims])
            
            if it == coeff_adaptive_weights*num_iters:
                img_L2_Loss = nn.MSELoss(reduction='none')
                pixel_losses = img_L2_Loss(pred, GT)
                weights = torch.ones_like(GT)

                max_abs_vor = torch.abs(img_vor[i, :, :, 0]).max()
                for l in range(weights.shape[0]):
                    for m in range(weights.shape[1]):
                        
                        weights[l][m][0] = (torch.abs(img_vor[i][l][m][0])/(max_abs_vor+eps_vor))*10.0
                        weights[l][m][1] = weights[l][m][0]
                        weights[l][m][2] = weights[l][m][0]
            
                weighted_loss = (pixel_losses * weights).mean()

                img_losses.append(weighted_loss)

            elif it > coeff_adaptive_weights*num_iters:
                img_L2_Loss = nn.MSELoss(reduction='none')
                pixel_losses = img_L2_Loss(pred, GT)
                
                weighted_loss = (pixel_losses * weights).mean()

                img_losses.append(weighted_loss)
            else:
                img_losses.append(L2_Loss(pred, GT))
    else: # if has boundary then compute loss only on the valid regions
        OUT = (boundary[0] >= -boundary[2])
        IN = ~OUT
        for i in range(batch_size):

            pred = rgb_to_yuv(sim_imgs[:, i])
            GT = rgb_to_yuv(imgs[init_frames[i]+1: init_frames[i]+1+num_sims])
            
            if it>= coeff_adaptive_weights*num_iters:
                img_L2_Loss = nn.MSELoss(reduction='none')
                pixel_losses = img_L2_Loss(pred, GT)
                weights = torch.ones_like(GT)
                max_abs_vor = torch.abs(img_vor[i, :, :, 0]).max()
                for l in range(weights.shape[0]):
                    for m in range(weights.shape[1]):
                        
                        weights[l][m][0] = torch.abs(img_vor[i][l][m][0])/(max_abs_vor+eps_vor)
                        weights[l][m][1] = weights[l][m][0]
                        weights[l][m][2] = weights[l][m][0]
            
                weighted_loss = (pixel_losses * weights).mean()
                img_losses.append(weighted_loss)
                print('weights',weights)
            else:
                img_losses.append(L2_Loss(pred, GT))
    
    img_loss = torch.stack(img_losses).sum()

    # loss is the sum of the two losses
    loss = img_loss + w_vorticity * VP_vorticity_derivative_loss + w_VP *  VP_vel_loss + w_size * VP_size_square_derivative_loss + w_len_0D * model_len_loss_0D + w_len_1D * model_len_loss_1D

    # optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    lr_scheduler.step(loss)

    if it % print_every == 0:
        print("[Train] Iter: ", it, ", loss per batch size and sim: ", (loss/(batch_size*num_sims)).detach().cpu().numpy(), "/ img loss per batch size and sim: ", (img_loss/(batch_size*num_sims)).detach().cpu().numpy(), "/ VP vorticity derivative loss per batch size and sim: ", (VP_vorticity_derivative_loss/(batch_size*num_sims)).detach().cpu().numpy(), "/ VP vel loss per batch size and sim: ", (VP_vel_loss/(batch_size*num_sims)).detach().cpu().numpy(), "/ VP size loss per batch size and sim: ", (VP_size_square_derivative_loss/(batch_size*num_sims)).detach().cpu().numpy(), "/ model len derivative loss per batch size and sim: ", "/ model len loss 0D per batch size and sim: ", (model_len_loss_0D/(batch_size*num_sims)).detach().cpu().numpy(), "/ model len loss 1D per batch size and sim: ", (model_len_loss_1D/(batch_size*num_sims)).detach().cpu().numpy())
        curr_time = time.time()
        print("[Train] Time Cost: ", curr_time-prev_time)
        prev_time = curr_time
        
        for i, param_group in enumerate(optimizer.param_groups):
            print(f"Parameter group {i}: lr = {param_group['lr']}")

    next_it = it + 1
    # save ckpt
    if (next_it % save_ckpt_every == 0 and next_it > 0) or (next_it == num_iters):
        path = os.path.join(ckptdir, '{:06d}.tar'.format(next_it))
        torch.save({
            'global_step': next_it,
            'w_pred': net_dict['w_pred'].state_dict(),
            'size_square_pred': net_dict['size_square_pred'].state_dict(),
            'model_FP_pos_state_dict': net_dict['model_FP_pos'].state_dict(),
            'model_VP_pos_state_dict': net_dict['model_VP_pos'].state_dict(),
            'model_len_state_dict': net_dict['model_len'].state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, path)
        print('[Train] Saved checkpoints at', path)

    if (next_it % test_every == 0 and next_it > 0) or (next_it == num_iters):
        test(next_it)

print('[Train] Complete.')

import torch
from torchvision.utils import save_image
import argparse
import pandas as pd
import sys

from nerf.provider import NeRFDataset
from nerf.utils_ip_adapter_v1 import *


def obtain_tempo_input(thetas, phis, radius, H, W, opt):
    device='cuda'
    # circle pose
    thetas = torch.FloatTensor([thetas]).to(device)
    phis = torch.FloatTensor([phis]).to(device)
    radius = torch.FloatTensor([radius]).to(device)
    poses, dirs = circle_poses(device, radius=radius, theta=thetas, phi=phis, return_dirs=True, angle_overhead=opt.angle_overhead, angle_front=opt.angle_front)

    near = opt.min_near
    far = 1000 # infinite
    cx, cy = H / 2, W / 2

    # fixed focal
    fov = opt.default_fovy

    focal = H / (2 * np.tan(np.deg2rad(fov) / 2))
    intrinsics = np.array([focal, focal, cx, cy])

    projection = torch.tensor([
        [2*focal/W, 0, 0, 0],
        [0, -2*focal/H, 0, 0],
        [0, 0, -(far+near)/(far-near), -(2*far*near)/(far-near)],
        [0, 0, -1, 0]
    ], dtype=torch.float32, device=device).unsqueeze(0)

    mvp = projection @ torch.inverse(poses) # [1, 4, 4]

    # sample a low-resolution but full image
    rays = get_rays(poses, intrinsics, H, W, -1)
    return rays, mvp

# ip-adapter obtain default view rendering result
def obtain_default_render_res(opt, model):
    # for stage 1, dreamfusion
    H, W = opt.h, opt.w
    rays, mvp = obtain_tempo_input(thetas=opt.default_polar, phis=opt.default_azimuth, radius=opt.default_radius, H=H, W=W, opt=opt)

    rays_o = rays['rays_o']
    rays_d = rays['rays_d']
    B, N = rays_o.shape[:2]

    shading = 'albedo'
    ambient_ratio = 1.0
    if opt.ip_textureless:
        shading = 'textureless'
        ambient_ratio = 0.0
    # bg_color = None
    bg_color = 1.0

    outputs = model.render(rays_o, rays_d, mvp, H, W, staged=False, perturb=True, bg_color=bg_color, ambient_ratio=ambient_ratio, shading=shading)
    # pred_depth = outputs['depth'].reshape(B, 1, H, W)
    pred_rgb = outputs['image'].reshape(B, H, W, 3).permute(0, 3, 1, 2).contiguous() # [B, 3, H, W]
    pred_normal = outputs['normal_image'].reshape(B, H, W, 3).permute(0, 3, 1, 2).contiguous() # [B, 3, H, W]

    return pred_rgb, pred_normal


def obtain_rendered_video(opt, model):
    # visualize 100 frames
    H, W = opt.h, opt.w
    cur_azimuth = 0
    all_preds = []
    for _ in range(100):
        rays, mvp = obtain_tempo_input(thetas=opt.default_polar, phis=cur_azimuth, radius=opt.default_radius, H=H, W=W, opt=opt)
        rays_o = rays['rays_o']
        rays_d = rays['rays_d']
        B, N = rays_o.shape[:2]

        shading = 'albedo'
        ambient_ratio = 1.0
        if opt.ip_textureless:
            shading = 'textureless'
            ambient_ratio = 0.0
        # bg_color = None
        bg_color = 1.0

        outputs = model.render(rays_o, rays_d, mvp, H, W, staged=False, perturb=True, bg_color=bg_color, ambient_ratio=ambient_ratio, shading=shading)
        pred_rgb = outputs['image'].reshape(B, H, W, 3)

        pred = pred_rgb[0].detach().cpu().numpy()
        pred = (pred * 255).astype(np.uint8)
        all_preds.append(pred)

        cur_azimuth = cur_azimuth + 3.6
        if cur_azimuth == 180:
            cur_azimuth = -cur_azimuth

    all_preds = np.stack(all_preds, axis=0)
    imageio.mimwrite(f'test.mp4', all_preds, fps=25, quality=8, macro_block_size=1)


if __name__ == '__main__':
    # See https://stackoverflow.com/questions/27433316/how-to-get-argparse-to-read-arguments-from-a-file-with-an-option-rather-than-pre
    class LoadFromFile (argparse.Action):
        def __call__ (self, parser, namespace, values, option_string = None):
            with values as f:
                # parse arguments in the file and store them in the target namespace
                parser.parse_args(f.read().split(), namespace)

    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=open, action=LoadFromFile, help="specify a file filled with more arguments")
    parser.add_argument('--text', default=None, help="text prompt")
    parser.add_argument('--negative', default='', type=str, help="negative text prompt")
    parser.add_argument('-O', action='store_true', help="equals --fp16 --cuda_ray")
    parser.add_argument('-O2', action='store_true', help="equals --backbone vanilla")
    parser.add_argument('--test', action='store_true', help="test mode")
    parser.add_argument('--six_views', action='store_true', help="six_views mode: save the images of the six views")
    parser.add_argument('--eval_interval', type=int, default=1, help="evaluate on the valid set every interval epochs")
    parser.add_argument('--test_interval', type=int, default=100, help="test on the test set every interval epochs")
    parser.add_argument('--workspace', type=str, default='workspace')
    parser.add_argument('--seed', default=None)

    parser.add_argument('--image', default=None, help="image prompt")
    parser.add_argument('--image_config', default=None, help="image config csv")

    parser.add_argument('--known_view_interval', type=int, default=4, help="train default view with RGB loss every & iters, only valid if --image is not None.")

    parser.add_argument('--IF', action='store_true', help="experimental: use DeepFloyd IF as the guidance model for nerf stage")

    ### IP adapter training
    parser.add_argument('--ip_adapter', action='store_true', help="experimental: use IP-Adapter to generate reference gt for nerf stage")
    parser.add_argument('--ip_adapter_geometry', action='store_true', help="experimental: ip-adapter sds loss stage 2")
    parser.add_argument('--ip_adapter_prompt_delta', action='store_true', help="experimental: whether to deal with the image prompt")
    parser.add_argument('--ip_adapter_ref_img', type=str, default='data/hamburger.png', help="reference image path for ip adapter")
    parser.add_argument('--ip_adapter_sds_epoch', type=int, default=0, help="in which epoch start reference sds supervise.")
    parser.add_argument('--ip_adapter_cfg', type=float, default=100, help="the guidance scale for ip-adapter.")
    parser.add_argument('--ip_adapter_ang_scal', type=int, default=75, help="in which angel range utilize reference sds supervise.")
    parser.add_argument('--ip_adapter_loss_w', type=float, default=1.0, help="the loss weight of IPSD.")
    parser.add_argument('--ip_adapter_final_MaxT', type=float, default=0.5, help="the diffusion inference max timestep for ip-adapter.")
    parser.add_argument('--ip_textureless', action='store_true', help="experimental: whether to visual textureless result.")
    parser.add_argument('--ip_video', action='store_true', help="experimental: whether to visual video.")

    parser.add_argument('--guidance', type=str, nargs='*', default=['SD'], help='guidance model')
    parser.add_argument('--guidance_scale', type=float, default=100, help="diffusion model classifier-free guidance scale")

    parser.add_argument('--save_mesh', action='store_true', help="export an obj mesh with texture")
    parser.add_argument('--mcubes_resolution', type=int, default=256, help="mcubes resolution for extracting mesh")
    parser.add_argument('--decimate_target', type=int, default=5e4, help="target face number for mesh decimation")

    parser.add_argument('--dmtet', action='store_true', help="use dmtet finetuning")
    parser.add_argument('--tet_grid_size', type=int, default=128, help="tet grid size")
    parser.add_argument('--init_with', type=str, default='', help="ckpt to init dmtet")
    parser.add_argument('--lock_geo', action='store_true', help="disable dmtet to learn geometry")

    ### training options
    parser.add_argument('--iters', type=int, default=10000, help="training iters")
    parser.add_argument('--lr', type=float, default=1e-3, help="max learning rate")
    parser.add_argument('--ckpt', type=str, default='latest', help="possible options are ['latest', 'scratch', 'best', 'latest_model']")
    parser.add_argument('--cuda_ray', action='store_true', help="use CUDA raymarching instead of pytorch")
    parser.add_argument('--taichi_ray', action='store_true', help="use taichi raymarching")
    parser.add_argument('--max_steps', type=int, default=1024, help="max num steps sampled per ray (only valid when using --cuda_ray)")
    parser.add_argument('--num_steps', type=int, default=64, help="num steps sampled per ray (only valid when not using --cuda_ray)")
    parser.add_argument('--upsample_steps', type=int, default=32, help="num steps up-sampled per ray (only valid when not using --cuda_ray)")
    parser.add_argument('--update_extra_interval', type=int, default=16, help="iter interval to update extra status (only valid when using --cuda_ray)")
    parser.add_argument('--max_ray_batch', type=int, default=4096, help="batch size of rays at inference to avoid OOM (only valid when not using --cuda_ray)")
    parser.add_argument('--latent_iter_ratio', type=float, default=0.2, help="training iters that only use albedo shading")
    parser.add_argument('--albedo_iter_ratio', type=float, default=0, help="training iters that only use albedo shading")
    parser.add_argument('--min_ambient_ratio', type=float, default=0.1, help="minimum ambient ratio to use in lambertian shading")
    parser.add_argument('--textureless_ratio', type=float, default=0.2, help="ratio of textureless shading")
    parser.add_argument('--jitter_pose', action='store_true', help="add jitters to the randomly sampled camera poses")
    parser.add_argument('--jitter_center', type=float, default=0.2, help="amount of jitter to add to sampled camera pose's center (camera location)")
    parser.add_argument('--jitter_target', type=float, default=0.2, help="amount of jitter to add to sampled camera pose's target (i.e. 'look-at')")
    parser.add_argument('--jitter_up', type=float, default=0.02, help="amount of jitter to add to sampled camera pose's up-axis (i.e. 'camera roll')")
    parser.add_argument('--uniform_sphere_rate', type=float, default=0, help="likelihood of sampling camera location uniformly on the sphere surface area")
    parser.add_argument('--grad_clip', type=float, default=-1, help="clip grad of all grad to this limit, negative value disables it")
    parser.add_argument('--grad_clip_rgb', type=float, default=-1, help="clip grad of rgb space grad to this limit, negative value disables it")
    # model options
    parser.add_argument('--bg_radius', type=float, default=1.4, help="if positive, use a background model at sphere(bg_radius)")
    parser.add_argument('--density_activation', type=str, default='exp', choices=['softplus', 'exp'], help="density activation function")
    parser.add_argument('--density_thresh', type=float, default=10, help="threshold for density grid to be occupied")
    parser.add_argument('--blob_density', type=float, default=5, help="max (center) density for the density blob")
    parser.add_argument('--blob_radius', type=float, default=0.2, help="control the radius for the density blob")
    # network backbone
    parser.add_argument('--backbone', type=str, default='grid', choices=['grid_tcnn', 'grid', 'vanilla', 'grid_taichi'], help="nerf backbone")
    parser.add_argument('--optim', type=str, default='adan', choices=['adan', 'adam'], help="optimizer")
    parser.add_argument('--sd_version', type=str, default='2.1', choices=['1.5', '2.0', '2.1'], help="stable diffusion version")
    parser.add_argument('--hf_key', type=str, default=None, help="hugging face Stable diffusion model key")
    # try this if CUDA OOM
    parser.add_argument('--fp16', action='store_true', help="use float16 for training")
    parser.add_argument('--vram_O', action='store_true', help="optimization for low VRAM usage")
    # rendering resolution in training, increase these for better quality / decrease these if CUDA OOM even if --vram_O enabled.
    parser.add_argument('--w', type=int, default=64, help="render width for NeRF in training")
    parser.add_argument('--h', type=int, default=64, help="render height for NeRF in training")
    parser.add_argument('--known_view_scale', type=float, default=1.5, help="multiply --h/w by this for known view rendering")
    parser.add_argument('--known_view_noise_scale', type=float, default=2e-3, help="random camera noise added to rays_o and rays_d")
    parser.add_argument('--dmtet_reso_scale', type=float, default=8, help="multiply --h/w by this for dmtet finetuning")
    parser.add_argument('--batch_size', type=int, default=1, help="images to render per batch using NeRF")

    ### dataset options
    parser.add_argument('--bound', type=float, default=1, help="assume the scene is bounded in box(-bound, bound)")
    parser.add_argument('--dt_gamma', type=float, default=0, help="dt_gamma (>=0) for adaptive ray marching. set to 0 to disable, >0 to accelerate rendering (but usually with worse quality)")
    parser.add_argument('--min_near', type=float, default=0.01, help="minimum near distance for camera")

    parser.add_argument('--radius_range', type=float, nargs='*', default=[3.0, 3.5], help="training camera radius range")
    parser.add_argument('--theta_range', type=float, nargs='*', default=[45, 105], help="training camera range along the polar angles (i.e. up and down). See advanced.md for details.")
    parser.add_argument('--phi_range', type=float, nargs='*', default=[-180, 180], help="training camera range along the azimuth angles (i.e. left and right). See advanced.md for details.")
    parser.add_argument('--fovy_range', type=float, nargs='*', default=[10, 30], help="training camera fovy range")

    parser.add_argument('--default_radius', type=float, default=3.2, help="radius for the default view")
    parser.add_argument('--default_polar', type=float, default=90, help="polar for the default view")
    parser.add_argument('--default_azimuth', type=float, default=0, help="azimuth for the default view")
    parser.add_argument('--default_fovy', type=float, default=20, help="fovy for the default view")

    parser.add_argument('--progressive_view', action='store_true', help="progressively expand view sampling range from default to full")
    parser.add_argument('--progressive_view_init_ratio', type=float, default=0.2, help="initial ratio of final range, used for progressive_view")
    
    parser.add_argument('--progressive_level', action='store_true', help="progressively increase gridencoder's max_level")

    parser.add_argument('--angle_overhead', type=float, default=30, help="[0, angle_overhead] is the overhead region")
    parser.add_argument('--angle_front', type=float, default=60, help="[0, angle_front] is the front region, [180, 180+angle_front] the back region, otherwise the side region.")
    parser.add_argument('--t_range', type=float, nargs='*', default=[0.02, 0.98], help="stable diffusion time steps range")
    parser.add_argument('--dont_override_stuff',action='store_true', help="Don't override t_range, etc.")


    ### regularizations
    parser.add_argument('--lambda_entropy', type=float, default=1e-3, help="loss scale for alpha entropy")
    parser.add_argument('--lambda_opacity', type=float, default=0, help="loss scale for alpha value")
    parser.add_argument('--lambda_orient', type=float, default=1e-2, help="loss scale for orientation")
    parser.add_argument('--lambda_tv', type=float, default=0, help="loss scale for total variation")
    parser.add_argument('--lambda_wd', type=float, default=0, help="loss scale")

    parser.add_argument('--lambda_mesh_normal', type=float, default=0.5, help="loss scale for mesh normal smoothness")
    parser.add_argument('--lambda_mesh_laplacian', type=float, default=0.5, help="loss scale for mesh laplacian")

    parser.add_argument('--lambda_guidance', type=float, default=1, help="loss scale for SDS")
    parser.add_argument('--lambda_rgb', type=float, default=1000, help="loss scale for RGB")
    parser.add_argument('--lambda_mask', type=float, default=500, help="loss scale for mask (alpha)")
    parser.add_argument('--lambda_normal', type=float, default=0, help="loss scale for normal map")
    parser.add_argument('--lambda_depth', type=float, default=10, help="loss scale for relative depth")
    parser.add_argument('--lambda_2d_normal_smooth', type=float, default=0, help="loss scale for 2D normal image smoothness")
    parser.add_argument('--lambda_3d_normal_smooth', type=float, default=0, help="loss scale for 3D normal image smoothness")

    ### debugging options
    parser.add_argument('--save_guidance', action='store_true', help="save images of the per-iteration NeRF renders, added noise, denoised (i.e. guidance), fully-denoised. Useful for debugging, but VERY SLOW and takes lots of memory!")
    parser.add_argument('--save_guidance_interval', type=int, default=10, help="save guidance every X step")

    ### GUI options
    parser.add_argument('--gui', action='store_true', help="start a GUI")
    parser.add_argument('--W', type=int, default=800, help="GUI width")
    parser.add_argument('--H', type=int, default=800, help="GUI height")
    parser.add_argument('--radius', type=float, default=5, help="default GUI camera radius from center")
    parser.add_argument('--fovy', type=float, default=20, help="default GUI camera fovy")
    parser.add_argument('--light_theta', type=float, default=60, help="default GUI light direction in [0, 180], corresponding to elevation [90, -90]")
    parser.add_argument('--light_phi', type=float, default=0, help="default GUI light direction in [0, 360), azimuth")
    parser.add_argument('--max_spp', type=int, default=1, help="GUI rendering max sample per pixel")

    parser.add_argument('--zero123_config', type=str, default='./pretrained/zero123/sd-objaverse-finetune-c_concat-256.yaml', help="config file for zero123")
    parser.add_argument('--zero123_ckpt', type=str, default='./pretrained/zero123/105000.ckpt', help="ckpt for zero123")
    parser.add_argument('--zero123_grad_scale', type=str, default='angle', help="whether to scale the gradients based on 'angle' or 'None'")
    parser.add_argument('--zero123_grads_num', type=float, default=1.0, help="whether to scale the gradients based on 'angle' or 'None'")

    parser.add_argument('--dataset_size_train', type=int, default=100, help="Length of train dataset i.e. # of iterations per epoch")
    parser.add_argument('--dataset_size_valid', type=int, default=8, help="# of frames to render in the turntable video in validation")
    parser.add_argument('--dataset_size_test', type=int, default=100, help="# of frames to render in the turntable video at test time")

    parser.add_argument('--exp_start_iter', type=int, default=None, help="start iter # for experiment, to calculate progressive_view and progressive_level")
    parser.add_argument('--exp_end_iter', type=int, default=None, help="end iter # for experiment, to calculate progressive_view and progressive_level")

    opt = parser.parse_args()

    opt.h = int(opt.h * opt.dmtet_reso_scale)
    opt.w = int(opt.w * opt.dmtet_reso_scale)
    opt.known_view_scale = 1

    if opt.backbone == 'vanilla':
        from nerf.network import NeRFNetwork
    elif opt.backbone == 'grid':
        from nerf.network_grid import NeRFNetwork
    elif opt.backbone == 'grid_tcnn':
        from nerf.network_grid_tcnn import NeRFNetwork
    elif opt.backbone == 'grid_taichi':
        opt.cuda_ray = False
        opt.taichi_ray = True
        import taichi as ti
        from nerf.network_grid_taichi import NeRFNetwork
        taichi_half2_opt = True
        taichi_init_args = {"arch": ti.cuda, "device_memory_GB": 4.0}
        if taichi_half2_opt:
            taichi_init_args["half2_vectorization"] = True
        ti.init(**taichi_init_args)
    else:
        raise NotImplementedError(f'--backbone {opt.backbone} is not implemented!')
    
    if opt.seed is not None:
        seed_everything(int(opt.seed))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = NeRFNetwork(opt).to(device)

    # if opt.dmtet and opt.init_with != '':
    #     if opt.init_with.endswith('.pth'):
    #         # load pretrained weights to init dmtet
    #         state_dict = torch.load(opt.init_with, map_location=device)
    #         model.load_state_dict(state_dict['model'], strict=False)
    #         if opt.cuda_ray:
    #             model.mean_density = state_dict['mean_density']
    #         model.init_tet()
    #     else:
    #         # assume a mesh to init dmtet (experimental, not working well now!)
    #         import trimesh
    #         mesh = trimesh.load(opt.init_with, force='mesh', skip_material=True, process=False)
    #         model.init_tet(mesh=mesh)
    
    state_dict = torch.load(opt.init_with, map_location=device)
    model.load_state_dict(state_dict['model'], strict=False)
    if model.cuda_ray:
        if 'mean_density' in state_dict:
            model.mean_density = state_dict['mean_density']

    if opt.dmtet:
        if 'tet_scale' in state_dict:
            new_scale = torch.from_numpy(state_dict['tet_scale']).to(device)
            model.verts *= new_scale / model.tet_scale
            model.tet_scale = new_scale

    if opt.ip_video:
        obtain_rendered_video(opt, model)
    else:
        pred_rgb, pred_normal = obtain_default_render_res(opt, model)
        save_image(pred_rgb, "vis.png")

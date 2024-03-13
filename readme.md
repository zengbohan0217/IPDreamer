# IPDreamer

The following content includes the environment installing of IPDreamer, training and testing commands, as well as a demonstration of the results.



## Install


### Install with pip

```bash
pip install -r requirements.txt
```

### Download pre-trained models

To use image-conditioned 3D generation, you need to download some pretrained checkpoints manually:
* [Zero-1-to-3](https://github.com/cvlab-columbia/zero123) for diffusion backend.
    It is hard-coded in `guidance/zero123_utils.py`.
    ```bash
    cd pretrained/zero123
    wget https://zero123.cs.columbia.edu/assets/zero123-xl.ckpt
    ```
* [Omnidata](https://github.com/EPFL-VILAB/omnidata/tree/main/omnidata_tools/torch) for depth and normal prediction.
    These ckpts are hardcoded in `preprocess_image.py`.
    ```bash
    mkdir pretrained/omnidata
    cd pretrained/omnidata
    # assume gdown is installed
    gdown '1Jrh-bRnJEjyMCS7f-WsaFlccfPjJPPHI&confirm=t' # omnidata_dpt_depth_v2.ckpt
    gdown '1wNxVO4vVbDEMEpnAi_jwQObf2MFodcBR&confirm=t' # omnidata_dpt_normal_v2.ckpt
    ```
* [IP-Adapter](https://github.com/tencent-ailab/IP-Adapter) for image prompts.
    ```bash
    mkdir IP-Adapter
    cd IP-Adapter
    git lfs install
    git clone https://huggingface.co/h94/IP-Adapter
    GIT_LFS_SKIP_SMUDGE=1
    ```

For DMTet, we port the pre-generated `32/64/128` resolution tetrahedron grids under `tets`.
The 256 resolution one can be found [here](https://drive.google.com/file/d/1lgvEKNdsbW5RS4gVxJbgBS4Ac92moGSa/view?usp=sharing).

### Build extension (optional)
By default, we use [`load`](https://pytorch.org/docs/stable/cpp_extension.html#torch.utils.cpp_extension.load) to build the extension at runtime.
We also provide the `setup.py` to build each extension:
```bash
cd stable-dreamfusion

# install all extension modules
bash scripts/install_ext.sh

# if you want to install manually, here is an example:
pip install ./raymarching # install to python path (you still need the raymarching/ folder, since this only installs the built extension.)
```

### Taichi backend (optional)
Use [Taichi](https://github.com/taichi-dev/taichi) backend for Instant-NGP. It achieves comparable performance to CUDA implementation while **No CUDA** build is required. Install Taichi with pip:
```bash
pip install -i https://pypi.taichi.graphics/simple/ taichi-nightly
```

### Trouble Shooting:
* we assume working with the latest version of all dependencies, if you meet any problems from a specific dependency, please try to upgrade it first (e.g., `pip install -U diffusers`). If the problem still holds, [reporting a bug issue](https://github.com/ashawkey/stable-dreamfusion/issues/new?assignees=&labels=bug&template=bug_report.yaml&title=%3Ctitle%3E) will be appreciated!
* `[F glutil.cpp:338] eglInitialize() failed Aborted (core dumped)`: this usually indicates problems in OpenGL installation. Try to re-install Nvidia driver, or use nvidia-docker as suggested in https://github.com/ashawkey/stable-dreamfusion/issues/131 if you are using a headless server.
* `TypeError: xxx_forward(): incompatible function arguments`： this happens when we update the CUDA source and you used `setup.py` to install the extensions earlier. Try to re-install the corresponding extension (e.g., `pip install ./gridencoder`).


## Training

Training command for scribble-guided 3D object generation.

```bash
# obtain inference image
python obtain_IPadapter_image.py --text 'A car made out of sushi' --out_path data/IPDreamer/sushi_car_scribble/v1/ --scribble_img_path data/scribble_sample/sushi_car/v1.png

# obtain normal and depth image
python preprocess_image.py --path 'data/IPDreamer/sushi_car_scribble/v1/reference.png'

# NeRF stage
## zero123
python main_v1.py -O --image data/IPDreamer/sushi_car_scribble/v1/reference_rgba.png --workspace output/trial_zero123 --iters 5000 --zero123_ckpt pretrained/zero123/zero123-xl.ckpt --sd_version 2.1

# mesh stage
## geometry
python main_v1.py -O --text "A car made out of sushi" --workspace output/trial_dmtet_geo --dmtet --iters 10000 --init_with output/trial_zero123/checkpoints/df.pth --sd_version 2.1 --ip_adapter --ip_adapter_ref_img data/IPDreamer/sushi_car_scribble/v1/reference_normal.png --ip_adapter_ang_scal 180 --ip_adapter_sds_epoch 0 --lambda_normal 1 --ip_adapter_cfg 100 --guidance_scale 100 --ip_adapter_geometry --guidance KKK
## texture
python main_v1.py -O --text "A car made out of sushi" --workspace output/trial_dmtet_tex --dmtet --iters 15000 --init_with output/trial_dmtet_geo/checkpoints/df.pth --sd_version 2.1 --ip_adapter --ip_adapter_ref_img data/IPDreamer/sushi_car_scribble/v1/reference.png --ip_adapter_ang_scal 180 --ip_adapter_sds_epoch 0 --lambda_normal 1 --ip_adapter_cfg 100 --guidance_scale 100 --ip_adapter_prompt_delta --guidance KKK
```

Training command for 3D object texture editing.

```bash
# obtain inference image
python obtain_IPadapter_image.py --text 'A high-quality shoe' --out_path data/IPDreamer/shoe_scribble/v1/ --scribble_img_path data/scribble_sample/shoe/v1.png

# obtain normal and depth image
python preprocess_image.py --path 'data/IPDreamer/shoe_scribble/v1/reference.png'

# NeRF stage
## zero123
python main_v1.py -O --image data/IPDreamer/shoe_scribble/v1/reference_rgba.png --workspace output/trial_zero123 --iters 5000 --zero123_ckpt pretrained/zero123/zero123-xl.ckpt --sd_version 2.1

# 3D Edit
## obtain normal and depth image
python preprocess_image.py --path 'data/scribble_sample/shoe/shoe1/reference.png'
## geometry
python main_v1.py -O --text "A high-quality shoe" --workspace output/trial_dmtet_geo --dmtet --iters 10000 --init_with output/trial_zero123/checkpoints/df.pth --sd_version 2.1 --ip_adapter --ip_adapter_ref_img data/scribble_sample/shoe/shoe1/reference_normal.png --ip_adapter_ang_scal 180 --ip_adapter_sds_epoch 0 --lambda_normal 1 --ip_adapter_cfg 100 --guidance_scale 100 --ip_adapter_geometry --guidance KKK
## texture
python main_v1.py -O --text "A high-quality shoe" --workspace output/trial_dmtet_tex --dmtet --iters 15000 --init_with output/trial_dmtet_geo/checkpoints/df.pth --sd_version 2.1 --ip_adapter --ip_adapter_ref_img data/scribble_sample/shoe/shoe1/reference.png --ip_adapter_ang_scal 180 --ip_adapter_sds_epoch 0 --lambda_normal 1 --ip_adapter_cfg 100 --guidance_scale 100 --ip_adapter_prompt_delta --guidance KKK

```


## Evalutation

Visualize the 3D object generated by IPDreamer.

```bash
python CUDA_VISIBLE_DEVICES=7 python test_v1.py --init_with output/trial_dmtet_tex/checkpoints/df.pth --dmtet --lambda_normal 1 --default_radius 2.8 --default_azimuth -45 --default_polar 80 --ip_video 
```

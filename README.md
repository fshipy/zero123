# Zero-1-to-3: Zero-shot One Image to 3D Object
### [Project Page](https://zero123.cs.columbia.edu/)  | [Paper](https://arxiv.org/abs/2003.08934) | [Weights](https://drive.google.com/drive/folders/1geG1IO15nWffJXsmQ_6VLih7ryNivzVs?usp=sharing) | [Demo (precomputed)](https://huggingface.co/spaces/cvlab/zero123) | [Demo (live)](placeholder)

[Zero-1-to-3: Zero-shot One Image to 3D Object](https://zero123.cs.columbia.edu/)  
 [Ruoshi Liu](https://people.eecs.berkeley.edu/~bmild/)<sup>1</sup>, [Rundi Wu](https://www.cs.columbia.edu/~rundi/)<sup>1</sup>,[Basile Van Hoorick](https://basile.be/about-me/)<sup>1</sup>,[Pavel Tokmakov](https://pvtokmakov.github.io/home/)<sup>2</sup>,[Sergey Zakharov](https://zakharos.github.io/)<sup>2</sup>,[Carl Vondrick](https://www.cs.columbia.edu/~vondrick/)<sup>1</sup> <br>
 <sup>1</sup>Columbia University, <sup>2</sup>Toyota Research Institute
 
##  Usage
###  Novel View Synthesis
```
conda create -n zero123 python=3.9
conda activate zero123
cd zero123
pip install -r requirements.txt
git clone https://github.com/CompVis/taming-transformers.git
pip install -e taming-transformers/
git clone https://github.com/openai/CLIP.git
pip install -e CLIP/
```

Download checkpoint under `zero123`:

```
https://drive.google.com/drive/folders/1geG1IO15nWffJXsmQ_6VLih7ryNivzVs?usp=sharing
```

Run our gradio demo for novel view synthesis:

```
python gradio_new.py
```

Note that this app uses around 29 GB of VRAM, so it may not be possible to run it on any GPU.

### 3D Reconstruction

```
cd 3drec
pip install -r requirements.txt
python run_zero123.py \
    --scene 'pikachu' \
    --index 0 \
    --n_steps 10000 \
    --lr 0.05 \
    --sd.scale 100.0 \
    --emptiness_weight 0 \
    --depth_smooth_weight 10000. \
    --near_view_weight 10000. \
    --train_view True \
    --prefix 'experiments/exp_wild' \
    --vox.blend_bg_texture False \
    --nerf_path 'data/nerf_wild'
```
We tested the installation processes on a system with Ubuntu 20.04 with an NVIDIA GPU with Ampere architecture.

##  Acknowledgement
This repocitory is based on [Stable Diffusion](https://github.com/CompVis/stable-diffusion), [Objaverse](https://objaverse.allenai.org/), and [SJC](https://github.com/pals-ttic/sjc/). We would like to thank the authors of these work for publicly releasing their code. We would like to thank the authors of [NeRDi](https://arxiv.org/abs/2212.03267) and [SJC](https://github.com/pals-ttic/sjc/) for their helpful feedback.

We would like to thank Changxi Zheng for many helpful discussions. This research is based on work partially supported by the Toyota Research Institute, the DARPA MCS program under Federal Agreement No. N660011924032, and the NSF NRI Award #1925157.

##  Citation
Placeholder
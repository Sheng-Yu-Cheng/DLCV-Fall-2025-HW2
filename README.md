# DLCV-Fall-2025-HW2
Please click [this link](https://docs.google.com/presentation/d/1hvJTxwLJPOLekcOxziF-u4kuIUimPXTbSjdEWVeLHX4/edit?usp=sharing)

## Deadline

**2025/10/22 (Wed.) 23:59 (GMT+8)**

## Packages

This homework should be completed using Python 3.8.5. For a list of packages you are allowed to import in this assignment, please refer to the `stable-diffusion/environment.yaml` for more details.

### Installation

You can run the following commands to install all the packages listed in the `environment.yaml`:

```sh
cd stable-diffusion
conda env create -f environment.yaml
conda activate ldm
```

To install the stable-diffusion v1.4. You can download the sd-v1-4.ckpt file from [hugging face link](https://huggingface.co/CompVis/stable-diffusion-v-1-4-original/tree/main) or use wget

```sh
wget https://huggingface.co/CompVis/stable-diffusion-v-1-4-original/resolve/main/sd-v1-4.ckpt
```

place the model checkpoint in the following path:

```sh
stable-diffusion/models/ldm/stable-diffusion-v1/model.ckpt
```

You can run the following command to test if you successfully build the environment
```sh
python scripts/txt2img.py --prompt "a photograph of an astronaut riding a horse"
```

(Optional)
You can also install the environment with pip and requirements.txt, note that the version of pytorch should match your CUDA version. [pytorch CUDA version](https://pytorch.org/get-started/previous-versions/)
```sh
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
pip install -r requirements.txt
cd stable-diffusion
pip install -e .
```

> :warning: **Note**: Using packages with different versions will very likely lead to compatibility issues, so make sure that you install the correct version if one is specified above. E-mail or ask the TAs first if you want to import other packages.

> :warning: **Important**: You can **NOT** use **diffuser** for easy implementation!

## Hint

Your model is a DDPM, located in stable-diffusion/ldm/models/diffusion/ddpm.py. You may need to use model.cond_stage_model for the text encoder. Since Diffusion v1.4 is a latent diffusion model that operates in the latent space, functions such as model.encode_first_stage and model.get_first_stage_encoding may also be necessary. Additionally, model.q_sample could be useful for your process.

## Grading

We provide the code for evaluation of hw2_3. For details on how to use it, please refer to the explanation in the slides.

## Q&A

If you have any problems related to HW2, you may:

- **Use TA Hours**: Tue. 16:30:~17:29 in MK514, Fri. 16:30 ~ 17:20​.
- **Contact TAs by Email**: [ntu-dlcv-2025-fall-ta@googlegroups.com](mailto:ntu-dlcv-2025-fall-ta@googlegroups.com) with the title `[HW2] Problems about ...`
- **Post Your Question**: Under the HW2 discussion section in NTU COOL.

Feel free to reach out if you have any questions or need further assistance!
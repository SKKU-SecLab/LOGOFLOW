

## LogoFlow: Attention-Guided Adversarial Logo Generation against Siamese Phishing Detectors

LogoFlow is a framework designed to generate adversarial logo samples capable of evading state-of-the-art visual phishing detectors, such as **PhishIntention**.  
This repository includes the full training pipeline, sampling script, and all necessary components to reproduce the adversarial logo generation process.

By leveraging attention-guided perturbations and rectified flow–based generative modeling, LogoFlow enables controlled manipulation of logo appearance while preserving human-recognizable semantics.

<img src="logoflow_framework.pdf" width="500" />

<img src="target_modification.pdf" width="500" />

<br>
This project is based on the following repositories:
<br>
- Rectified Model implementation by Phil Wang (lucidrains) https://github.com/lucidrains/rectified-flow-pytorch<br>
- Siamese Model implementation by gyNancy https://github.com/gyNancy/Visualphish_public/image_attack/guided_diffusion<br>
- PhishIntention by Ruofan Liu (lindsey98) https://github.com/lindsey98/PhishIntention.git

---

## Environment

Set up the LogoFlow environment using **Python 3.8**:

```bash
conda create -n LogoFlow python=3.8
conda activate LogoFlow
```

Install required dependencies:

```bash
pip install -r requirements.txt
```

## Pretrained Models

The pretrained OCR Siamese model weights used in LOGOFLOW are publicly available at this repository:

- **demo_downgrade.pth.tar**
- **bit.pth.tar**

🔗 https://github.com/gyNancy/Visualphish_public/blob/main/image_attack/guided_diffusion/OCR_siamese_utils/output/targetlist_lr0.01/
<br>
🔗 https://github.com/gyNancy/Visualphish_public/blob/main/image_attack/guided_diffusion/OCR_siamese_utils/

Please download the weights and place them under:

- "LOGOFLOW/rectified_flow_pytorch/guided_diffusion/OCR_siamese_utils/output/targetlist_lr0.01/bit.pth.tar"
- "LOGOFLOW/rectified_flow_pytorch/guided_diffusion/OCR_siamese_utils/demo_downgrade.pth.tar"
  

## Training

Train the adversarial logo generator using:

```python
python train_logo.py
```

## Sampling

To generate adversarial logo samples from a trained model:

```python
python sampling.py
```


## Citations

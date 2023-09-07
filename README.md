# DeepDINO-DepthEstimation 🦖

DeepDINO is a supervised depth estimation model that leverages the DINOv2 backbone for rich visual representations. DINO, which stands for Self-Supervised Vision Transformers, is introduced in the paper [Learning Robust Visual Features without Supervision](https://arxiv.org/abs/2304.07193).

![DeepDINO Architecture](images/architecture.png)

## 🌐 Table of Contents

- [DeepDINO-DepthEstimation 🦖](#deepdino-depthestimation-)
  - [🌐 Table of Contents](#-table-of-contents)
  - [💻 Installation](#-installation)
  - [🚀 Examples](#-examples)
  - [🤝 Contributing](#-contributing)
    - [Docstring Guidelines](#docstring-guidelines)
  - [📄 License](#-license)

## 💻 Installation

Clone the repository and install the necessary requirements:
```bash
git clone https://github.com/Tottowich/DeepDINO-DepthEstimation.git
cd DeepDINO-DepthEstimation
pip install -r requirements.txt
pip install git+https://github.com/facebookresearch/dinov2
```

## 🚀 Examples

<!-- Display GIFs found in videos/-->

Live videos of the model in action can be found in the [videos](videos/) folder. These are streams of the model running on a NVIDIA 3060 ti GPU on 720x1280 resolution.
| ![Living room](videos/living_room.gif) | ![Office](videos/office.gif) |


## 🤝 Contributing 

We welcome contributions from the community! If you're interested in enhancing DeepDINO, please follow our [Contributing Guidelines](CONTRIBUTING.md) and [Code of Conduct](CODE_OF_CONDUCT.md).

### Docstring Guidelines

For documenting Python code, please adhere to the Totto-style docstring guidelines. Detailed guidelines can be found [here](docs/docstring_guidelines.md).

## 📄 License

[// License Information]


<html>
<head>
    <title>Playable Videos</title>
</head>
<body>

<h1>Model in Action</h1>

<h2>Living Room</h2>
<video controls width="300">
    <source src="videos/living_room.mp4" type="video/mp4">
    Your browser does not support the video tag.
</video>

<h2>Office</h2>
<video controls width="300">
    <source src="videos/office.mp4" type="video/mp4">
    Your browser does not support the video tag.
</video>

</body>
</html>
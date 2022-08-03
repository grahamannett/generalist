# For ubuntu install


## Install mujoco-py

commands ran:
conda install -c conda-forge glew -y
conda install -c conda-forge mesalib -y
conda install -c menpo glfw3 -y

export CPATH=/home/graham/anaconda3/envs/py310/include

pip install patchelf

pip install mujoco-py

then run: python -c 'import mujoco_py'

pip install git+https://github.com/rail-berkeley/d4rl@master#egg=d4rl


pip install torch torchvision transformers datasets sentencepiece  einops



# for setup.py
make the project importable better or whatever

pip install -e .


# Data
- https://github.com/rom1504/img2dataset
- https://github.com/allenai/aokvqa
    - the way this model/repo works: converts train/test/val images/questions to embeddings (model.encode_image)


# Image Patches Related
Found various implementations, still not exactly clear to me what the ideal way is
- https://sachinruk.github.io/blog/pytorch/data/2021/07/03/Image-Patches.html
- https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial15/Vision_Transformer.html



# Design choices

## Embeddings

One of the factors





# Datasets
- https://sites.google.com/view/d4rl/home



# Other repos

https://www.youtube.com/watch?v=P_xeshTnPZg
https://github.com/lucidrains/perceiver-pytorch/blob/main/perceiver_pytorch/perceiver_io.py
https://github.com/deepmind/deepmind-research/blob/master/perceiver/io_processors.py
https://github.com/deepmind/deepmind-research/tree/master/perceiver/train
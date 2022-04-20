# PC2-PU

This repository contains a Pytorch implementation of the paper:

[PC2-PU: Patch Correlation and Position Correction for Effective Point Cloud Upsampling](https://arxiv.org/abs/2109.09337).

## Getting Started

1. Clone the repository:

   ```shell
   git clone https://github.com/chenlongwhu/PC2-PU.git
   cd PC2-PU
   ```
   Installation instructions for Ubuntu 18.04:
   * Make sure <a href="https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html">CUDA</a>  and <a href="https://docs.nvidia.com/deeplearning/sdk/cudnn-install/index.html">cuDNN</a> are installed. Only this configurations has been tested:

     - Python 3.7.11, Pytorch 1.6.0
    * Follow <a href="https://pytorch.org/">Pytorch installation procedure</a>. Note that the version of cudatoolkit must be strictly consistent with the version of CUDA

2. Install KNN_cuda.
    ```
    pip install --upgrade https://github.com/unlimblue/KNN_CUDA/releases/download/0.2/KNN_CUDA-0.2-py3-none-any.whl
    ```
3. Install Pointnet2_ops
    ```
    pip install "git+git://github.com/erikwijmans/Pointnet2_PyTorch.git#egg=pointnet2_ops&subdirectory=pointnet2_ops_lib"
    ```

4. Install emd
    ```
    cd emd_module
    python setup.py install
    ```

5. Install h5py tensorboard
    ```
    conda install h5py
    conda install tensorboard
    ```

6. Train the model:
    First, you need to download the training patches in HDF5 format from [GoogleDrive](https://drive.google.com/drive/folders/1Mam85gXD9DTamltacgv8ZznSyDBbBovv?usp=sharing) and put it in folder `data`.
    Then run:
   ```shell
   cd code
   python main.py --log_dir log/PC2-PU
   ```

7. Evaluate the model:
    First, you need to download the pretrained model from [GoogleDrive](https://drive.google.com/file/d/1CebnBUtX2OsoPnBNtquUVfZmgqRQPfhm/view?usp=sharing), extract it and put it in folder `log/PC2-PU`.
    Then run:
   ```shell
   cd code
   python main.py --phase test --log_dir log/PC2-PU --checkpoint_path model-best.pth.tar
   ```
   You will see the input and output results in the folder `log/PC2-PU`.

8. The training and testing mesh files can be downloaded from [GoogleDrive](https://drive.google.com/open?id=1BNqjidBVWP0_MUdMTeGy1wZiR6fqyGmC).

### Evaluation code
We provide the evaluation code. In order to use it, you need to install the CGAL library. Please refer [this link](https://www.cgal.org/download/linux.html) and  [PU-Net](https://github.com/yulequan/PU-Net) to install this library.
Then:
   ```shell
   cd evaluation_code
   cmake .
   make
   ./evaluation Icosahedron.off Icosahedron.xyz
   ```
The second argument is the mesh, and the third one is the predicted points.

## Citation

If PC2-PU is useful for your research, please consider citing:

    @misc{long2021pc2pu,
      title={PC2-PU: Patch Correlation and Position Correction for Effective Point Cloud Upsampling},
      author={Chen Long and Wenxiao Zhang and Ruihui Li and Hao Wang and Zhen Dong and Bisheng Yang},
      year={2021},
      eprint={2109.09337},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
    }

## Related Repositories
The original code framework is rendered from ["PUGAN_pytorch"](https://github.com/UncleMEDM/PUGAN-pytorch). It is developed by [Haolin Liu](https://github.com/UncleMEDM) at The Chinese University of HongKong.

The original code of emd is rendered from ["MSN"](https://github.com/Colin97/MSN-Point-Cloud-Completion). It is developed by [Liu Minghua](http://cseweb.ucsd.edu/~mil070/) at The University of California, San Diego.

The original code of chamfer3D is rendered from ["chamferDistancePytorch"](https://github.com/ThibaultGROUEIX/ChamferDistancePytorch/tree/master/chamfer3D). It is developed by [ThibaultGROUEIX](http://imagine.enpc.fr/~groueixt).

The original code of helper is rendered from ["Self-supervised Sparse-to-Dense:  Self-supervised Depth Completion from LiDAR and Monocular Camera"](https://github.com/fangchangma/self-supervised-depth-completion). It is developed by [Fangchang Ma](http://www.mit.edu/~fcma/), Guilherme Venturelli Cavalheiro, and [Sertac Karaman](http://karaman.mit.edu/) at MIT.

### Questions

Please contact 'chenlong107@whu.edu.cn'

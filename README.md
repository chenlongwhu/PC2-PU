# PC2-PU

This repository contains a Pytorch implementation of the paper:

[PC2-PU: Patch Correlation and Position Correction for Effective Point Cloud Upsampling](https://arxiv.org/abs/2109.09337).

## News
We checked our code and found a problem, which is fixed now. The results have also changed. Our main experiment results are as follows.

1. Benchmark(x4)  : CD: 0.2317 -> 0.2327 , HD: 2.5942 -> 2.5962
2. Benchmark(x16) : CD: 0.0998 -> 0.1004 , HD: 2.8692 -> 2.9130
3. Add 0.5% noise : CD: 0.2604 -> 0.2598
4. Add 1.0% noise : CD: 0.3586 -> 0.3589
5. Add 2.0% noise : CD: 0.7727 -> 0.7701
6. Generalization Test : CD: 5099 -> 0.5102, HD: 5.9618 -> 5.9555


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
    pip install "git+https://github.com/erikwijmans/Pointnet2_PyTorch.git/#egg=pointnet2_ops&subdirectory=pointnet2_ops_lib"
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
   python main.py --phase test --log_dir log/PC2-PU --checkpoint_path model_best.pth.tar
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

    @inproceedings{long2022pc2pu,
      title={PC2-PU: Patch Correlation and Position Correction for Effective Point Cloud Upsampling},
      author={Chen Long and Wenxiao Zhang and Ruihui Li and Hao Wang and Zhen Dong and Bisheng Yang},
      year={2022},
      booktitle={Proceedings of the 30th ACM International Conference on Multimedia},
      url= {https://doi.org/10.1145/3503161.3547777},
      doi= {10.1145/3503161.3547777},
    }

## Related Repositories
The original code framework is rendered from ["PUGAN_pytorch"](https://github.com/UncleMEDM/PUGAN-pytorch). It is developed by [Haolin Liu](https://github.com/UncleMEDM) at The Chinese University of HongKong.

The original code of emd is rendered from ["MSN"](https://github.com/Colin97/MSN-Point-Cloud-Completion). It is developed by [Liu Minghua](http://cseweb.ucsd.edu/~mil070/) at The University of California, San Diego.

The original code of chamfer3D is rendered from ["chamferDistancePytorch"](https://github.com/ThibaultGROUEIX/ChamferDistancePytorch/tree/master/chamfer3D). It is developed by [ThibaultGROUEIX](http://imagine.enpc.fr/~groueixt).

The original code of helper is rendered from ["Self-supervised Sparse-to-Dense:  Self-supervised Depth Completion from LiDAR and Monocular Camera"](https://github.com/fangchangma/self-supervised-depth-completion). It is developed by [Fangchang Ma](http://www.mit.edu/~fcma/), Guilherme Venturelli Cavalheiro, and [Sertac Karaman](http://karaman.mit.edu/) at MIT.

### Questions

Please contact 'chenlong107@whu.edu.cn'

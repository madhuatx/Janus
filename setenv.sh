# MADHU: These are rocm-specific post-install steps to get pytorch
# working on rocm


# install special index packages from rocm
# see https://rocm.docs.amd.com/projects/radeon/en/latest/index.html
wget https://repo.radeon.com/rocm/manylinux/rocm-rel-6.2.3/torch-2.3.0%2Brocm6.2.3-cp310-cp310-linux_x86_64.whl
wget https://repo.radeon.com/rocm/manylinux/rocm-rel-6.2.3/torchvision-0.18.0%2Brocm6.2.3-cp310-cp310-linux_x86_64.whl
wget https://repo.radeon.com/rocm/manylinux/rocm-rel-6.2.3/pytorch_triton_rocm-2.3.0%2Brocm6.2.3.5a02332983-cp310-cp310-linux_x86_64.whl
pip3 uninstall torch torchvision pytorch-triton-rocm
pip3 install torch-2.3.0+rocm6.2.3-cp310-cp310-linux_x86_64.whl torchvision-0.18.0+rocm6.2.3-cp310-cp310-linux_x86_64.whl pytorch_triton_rocm-2.3.0+rocm6.2.3.5a02332983-cp310-cp310-linux_x86_64.whl

# Taken from https://rocm.docs.amd.com/projects/radeon/en/latest/docs/install/native_linux/install-onnx.html
pip3 install https://repo.radeon.com/rocm/manylinux/rocm-rel-6.2.3/onnxruntime_rocm-1.18.0-cp310-cp310-linux_x86_64.whl

# take from https://rocm.docs.amd.com/projects/radeon/en/latest/docs/install/native_linux/install-migraphx.html

# Setup pythonpath for migraphx module
export PYTHONPATH=/opt/rocm/lib:$PYTHONPATH

#setup instructions for torch_migraphx module
git clone https://github.com/ROCmSoftwarePlatform/torch_migraphx.git ./env/torch_migraphx
cd ./env/torch_migraphx/py
export TORCH_CMAKE_PATH=$(python -c "import torch; print(torch.utils.cmake_prefix_path)")
pip install .
cd ../../../


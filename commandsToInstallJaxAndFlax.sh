conda create -n dl
conda activate dl
conda install jax
conda install jaxlib
pip install flax
conda install ipykernel

conda create -n dl_311 python=3.11
conda activate dl_311
# pip install clu # this does not seem to work
conda install -y jaxlib; conda install -y jax; conda install -y ipykernel; pip install flax
conda install tensorflow
pip install tensorflow_datasets
conda deactivate
# conda env remove --name dl_311

# conda create -n dl_38 python=3.8
# conda activate dl_38
# # pip install clu
# pip install tensorflow
# pip install tensorflow_datasets
# conda install -y jaxlib; conda install -y jax; conda install -y ipykernel; pip install flax; 
# conda env remove --name dl_38

# conda create -n dl_39 python=3.9
# conda activate dl_39
# # pip install clu
# pip install tensorflow
# pip install tensorflow_datasets
# conda deactivate
# conda env remove --name dl_39

# conda create -n dl_312 python=3.12
# conda activate dl_312
# pip install clu
# pip install tensorflow
# pip install tensorflow_datasets
# conda deactivate
# conda env remove --name dl_312


# conda create -n dl_310 python=3.10
# conda activate dl_310
# # pip install clu
# pip install tensorflow
# pip install tensorflow_datasets
# # conda install -y jaxlib; conda install -y jax; conda install -y ipykernel; pip install flax; 
# conda deactivate
# conda env remove --name dl_310


conda create -n exp3 
conda activate exp3
# pip install clu # this does not seem to work
conda install -y matplotlib
conda install ipykernel

conda install -y jaxlib; conda install -y jax; conda install -y ipykernel; pip install flax
conda install -y tensorflow
pip install -y tensorflow_datasets 
conda deactivate
# conda env remove --name exp

conda create -n exp_311 python=3.11
conda activate exp_311
conda install -y matplotlib
conda install -y ipykernel


conda create -n dl_alt
conda activate dl_alt
conda install -y Plotly
conda install -y ipykernel
pip install pandas
conda deactivate
conda env remove --name dl_alt

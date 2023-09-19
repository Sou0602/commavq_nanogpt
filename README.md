# commavq_nanogpt

## Contents
* config
    * train_commavq.py - config file for training with configuration 1
    * train_commavq_1.py - config file for training with configuration 2
    * train_commavq_2.py - config file for training with configuration 3
    * eval_commavq.py - config file for evaluation
* data
    * commavq
        * prepare.py - Prepares the dataset to train.bin and val.bin
        * create_meta_pkl.py - Update the meta vocab size 
* out-commavq   - Folder with checkpoints for config 1 (not present in the repo, add from google drive[https://drive.google.com/drive/folders/1ir8xxywEh3fWsWhd66ARkIDFHPQzCoBU?usp=sharing])
* out-commavq_1 - Folder with checkpoints for config 2 (not present in the repo, add from google drive[https://drive.google.com/drive/folders/1ir8xxywEh3fWsWhd66ARkIDFHPQzCoBU?usp=sharing])
* out-commavq_2 - Folder with checkpoints for config 3 (not present in the repo, add from google drive[https://drive.google.com/drive/folders/1ir8xxywEh3fWsWhd66ARkIDFHPQzCoBU?usp=sharing])
* train.py - Training Script
* model.py - Model Information
* configurator.py - Helper to update the Training Arguments from files in config directory
* eval.py - Evaluation Script
* export_to_onnx.py - To convert from checkpoint to Onnx
* eval_commavq_ort.py - Evalaute Onnx models with onnx runtimes
* run_tests.sh  - Bash script to run tests


## Creating the Environment
```
$ conda create --name <env_name> --file env_requirements.txt
$ conda activate <env_name>
```

## Load Data in the data/commavq directory

We need to create the data in the right format of train.bin and val.bin.  
  
The file below is a modified version of prepare.py from the comma-vq dataset:  
https://github.com/commaai/commavq/blob/master/nanogpt/prepare.py  
  
```
$ python data/commavq prepare.py
```

The configuration files have 3 model configurations.
* train_commavq.py (8 heads, vocab_size = 1048)
* train_commavq_1.py (8 heads, vocab_size = 1048)
* train_commavq_2.py (6 heads, vocab_size = 1026)

In my models, vocab_size/output dimension is different for the model with 6 heads and with 8 heads. 
The model with 6 heads has a vocab_size of 1026, and the model with 8 heads. 

To update the vocab size, change the n_heads parameter. The default file has 1048 for 8 heads.
```
$ python data/commavq create_meta_pkl.py --n_heads=8
```

## Run Tests to check the data location, dataloader, and the model.
```
$ chmod +x run_tests.sh
```

```
$ ./run_tests.sh
```

## Training the configurations
```
$ python train.py config/train_commavq.py
```

To run different configurations, change the config file. Alternatively, create new configurations or update the parameters for different models in the config files.
For the model sizes, the training iterations can be stopped earlier, change the parameter max_iters for shorter training times. 

## Evaluating the Models
This file is updated from the eval.ipynb notebook from the commavq repo, https://github.com/commaai/commavq/blob/master/notebooks/eval.ipynb   

```
$ python eval.py
```

In the config/eval_commavq.py, change the out_dir parameter to evaluate different models. 
Alternatively, the eval file also takes an argument --ckpt_path for evaluating the model at the input checkpoint path. 

```
$ python eval.py --ckpt_path MODEL_PATH
```
## Exporting the Models to ONNX
The scaled_dot_product_attention operator is not included in the torch package. So, I updated the symbolic_opset14.py file in the onnx directory.(More details about this in the Training Log)

```
$ cp -f onnx-utils/symbolic_opset14.py $(pip show torch | grep "Location:" | awk '{print $2}')/torch/onnx
```
Make changes to the symbolic_opset.py to support scaled dot product attention.  
In the directory, onnx_utils has an updated version for symbolic_opset14.py, which can be replaced with the file in the onnx directory.  

Change the models by changing the Out_dir parameter in config/eval_commavq.py file. The file converts a ckpt.pt checkpoint to ckpt.onnx Onnx file in the same directory and terms with a dummy input with OnnxRuntime.
```
$ python export_to_onnx.py 
```

## Evaluating the Models with Onnx Files
Change the models by changing the Out_dir parameter in config/eval_commavq.py file.
```
$ python eval_commavq_ort.py
```

## Evaluating from checkpoints
Link to the checkpoints:
https://drive.google.com/drive/folders/1ir8xxywEh3fWsWhd66ARkIDFHPQzCoBU?usp=sharing

The drive link has three folders names, out-commavq, out-commavq_1,out-commavq_2. Each folder has a ckpt.pt and ckpt.onnx. To evaluate with these checkpoints, download the folder and copy it to the root directory of the repo. 
In config/eval_commavq.py, change the out_dir accordingly. 

## Running on Multiple Nodes
The code can also run on the karpathy train.py and model.py, if the config and data are moved to the karpathy's nanogpt repo. Follow more instructions in that repo to run on multiple nodes.




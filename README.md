# CSRNet-rs
This repository provides a port of the [pytorch CSRNet model](https://github.com/leeyeehoo/CSRNet-pytorch)
to rust using [burn-rs](https://github.com/burn-rs/burn) with that in hand, you
will be able to deploy a crowdcounting model without needing python on your
target environment.

This repo comprises three subprojects: `csrnet-import`, `csrnet-labeling` and, 
`csrnet`. However chances are, the only two that might be of interest to you are 
`csrnet-labeling` and `csrnet`. Indeed, `csrnet-import` only serves the point of 
generating the `burn` code for the model (that is `csrnet/src/model/csrnet.rs`) 
+ the weights (`csrnet/src/model/csrnet.mpk.gz`) from the onnx version of the model.

## Training on your own private set of images

### 1. Annotate your own dataset
To create your own dataset, you should proceed in several steps (this repo has
you covered for all of them). First, you should gather all your images and place
them in a directory `<mydataset>/images` and create a folder next to it called
`<mydataset>/ground_truth`. Once that is done, you can run the command:
`csrnet-labeling <path_to_mydataset>`. This will spawn a web interface that allows
to to click on each person on the image. This tool will generate a `.npy` file
for each of the images you annotated in the `<mydataset>/ground_truth` directory.

### 2. Prepare data for training
Once your data is annotated, you shoud run the `prepara_dataset.py` script that
is going to *resize your images* and generate the ground truth files in a `.h5` 
format. Once the script has run, you are ready to split your data (to create 
a training and a validation set) and then to proceed to the actual training. 

Note that the resizing step is **not mandatory**. However, it does help in 
speeding up the training process **a lot**.

### 3. Train
The actual training of a model is quite simple too. In essence, all what should
be done is to run `csrnet train -t <training_dataset> -v <valisation_dataset>`.
In addition to those, the csrnet tool lets you provide some extra arguments, 
eg a pretrained model you want to fine-tune. The following provides you
with a detailed view of the available options:
``` 
./target/release/csrnet train -h
csrnet-train 0.1.0
Train the model

USAGE:
    csrnet train [OPTIONS] --train <train> --validation <validation>

FLAGS:
    -h, --help       Prints help information
    -V, --version    Prints version information

OPTIONS:
    -b, --batch-size <batch-size>    Minibatch size to use during training [default: 1]
    -e, --epochs <epochs>            Number of epoch during with the training should be performed [default: 10]
    -m, --model <model>              The model to use
    -s, --seed <seed>                A seed for the prng [default: 42]
    -t, --train <train>              Path to the training dataset
    -v, --validation <validation>    Path to the validation dataset
```

### You can now perform inference
Once the training is complete, you will be able to use the `csrnet` tool to 
carry inference on your own live data. To do so, you will want to reuse either
the final trained model, or a given specific checkpoint. 

#### Using your trained model.
The training phase creates the model and checkpoints in two different locations. 
The final model which is obtained after completing the whole training is stored
as `outputs/model.bin`. 

The checkpoints, on the other hand, are created under `artifacts/checkpoints/`
each individual checkpoint is then available under 
`artifacts/checkpoints/checkpoint/checkpoint-xx.bin`. 

#### Actually performing the inference
Performing the inference is usually done using
`csrnet infer -m outputs/model.bin -i <path_to_image>.jpg`.

The following snipped details all the possible options that can be used when
carrying inference:
```
./target/release/csrnet infer -h
csrnet-infer 0.1.0
Uses the trained model to perform inference (that is: actually count people in an image)

USAGE:
    csrnet infer [OPTIONS] --image <image>

FLAGS:
    -h, --help       Prints help information
    -V, --version    Prints version information

OPTIONS:
    -i, --image <image>        The image whose number of people should be counted
    -j, --justify <justify>    Generate a justification image showing the estimation of the ground truth for the tested
                               image
```

If you want to compare the quality of a given prediction with its corresponding
ground truth, you can use the `csrnet check <ground_truth_h5>` command to do 
just that. More information about that command can be obtained using 
`csrnet check -h`.

## Compiling for your own machaine
To compile this project for your own machine, you should just run the command
`cargo build --release` along with the right feature flags. Those feature 
flags are meant to let you choose if you intend to use a CPU only implementation
and if not, what GPU backend you intend to use.

For that purspose, I have defined several features to configure which backend 
you most likely want to use. 

### CPU
If you don't have any gpu available, you should perform the compilation using
the `--features "cpu"` flag. In that case, the compiled tool is going to use 
an ndarray backend (with statically linked openblas on windows and linux and
a mac optimized version on apple hardware).

If you go the cpu road, chances are that your binary will take quite a bit of 
time to process even one single image (so I advise you to shrink them down
before passing them to `csrnet`). This is why I recommend using one of 
the gpu capable versions whenever possible.

### WGPU
Ideally, this would be the default way to go when there is a gpu available
and you don't know how to make use of it. Unfortunately, the default limits 
are IMHO a bit too low at the time, which makes it impractical to use when
working on large images. (But I think this is going to be resolved in a near
future #fingerscrossed -- if you are one of the burn-wgpu authors, let me 
know how I can help).

To use this backend, just use the `--features "wgpu"` flag when compiling.

### TCH
Using this feature, you will use the backend based on tch-rs (that is the
backend that uses binding to pytorch c++ api). Just because it uses the 
pytorch bindings doesn't mean you need to fiddle with your python setup to get
it working. It is one of the possibilities, but I found it to be surprisingly
simple to set it up by hand. Also, this handmade setup works better for me
because I makes it easy to have MPS device enabled on my mac.

To perform my handmade setup, I followed the (brief) explanations given in
[this post](https://github.com/LaurentMazare/tch-rs/issues/488#issuecomment-1664261286
In practice, it means that I downloaded the prebuilt binary from here:
https://github.com/mlverse/libtorch-mac-m1/releases/download/LibTorchOpenMP/libtorch-v2.0.0.zip
Then i unpacked it and added the following lines to my `.profile`
```
export LIBTORCH=$HOME/libtorch
export LIBTORCH_LIB=$HOME/libtorch
export DYLD_LIBRARY_PATH=$LIBTORCH/lib:$DYLD_LIBRARY_PATH
```
That was it. All that was left after that was to allow the usage of some dylib
that were not signed my apple. It works like a charm for me.

To use this backend, just use the `--features "tch"` flag when compiling. 

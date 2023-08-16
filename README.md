# CSRNet-rs
This repository provides a port of the [pytorch CSRNet model](https://github.com/leeyeehoo/CSRNet-pytorch)
to rust using [burn-rs](https://github.com/burn-rs/burn) with that in hand, you
will be able to deploy a crowdcounting model without needing python on your
target environment.

This repo comprises two subprojects: `csrnet-import` and `csrnet-infer`. However
chances are, the only that will be of interest to you is `csrnet-infer`. Indeed,
`csrnet-import` only serves the point of generating the `burn` code for the model
(that is `csrnet-infer/src/model/csrnet.rs`) + the weights 
(`csrnet-infer/src/model/csrnet.mpk.gz`) from the onnx version of the model.

## Features
I have defined several features to configure which backend you most likely 
want to use. 

### CPU
If you don't have any gpu available, you should perform the compilation using
the `--features "cpu"` flag. In that case, the compiled tool is going to use 
an ndarray backend (with statically linked openblas on windows and linux and
a mac optimized version on apple hardware).

If you go the cpu road, chances are that your binary will take quite a bit of 
time to process even one single image (so I advise you to shrink them down
before passing them to `csrnet-infer`. This is why I recommend using one of 
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

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

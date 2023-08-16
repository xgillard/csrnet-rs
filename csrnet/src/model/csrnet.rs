// Generated from ONNX "./resources/csrnet.onnx" by burn-import
use burn::nn::conv::Conv2d;
use burn::nn::conv::Conv2dConfig;
use burn::nn::pool::MaxPool2d;
use burn::nn::pool::MaxPool2dConfig;
use burn::nn::PaddingConfig2d;
use burn::record::Recorder;
use burn::{
    module::Module,
    tensor::{backend::Backend, Tensor},
};


#[derive(Module, Debug)]
pub struct Model<B: Backend> {
    conv2d1: Conv2d<B>,
    conv2d2: Conv2d<B>,
    maxpool2d1: MaxPool2d,
    conv2d3: Conv2d<B>,
    conv2d4: Conv2d<B>,
    maxpool2d2: MaxPool2d,
    conv2d5: Conv2d<B>,
    conv2d6: Conv2d<B>,
    conv2d7: Conv2d<B>,
    maxpool2d3: MaxPool2d,
    conv2d8: Conv2d<B>,
    conv2d9: Conv2d<B>,
    conv2d10: Conv2d<B>,
    conv2d11: Conv2d<B>,
    conv2d12: Conv2d<B>,
    conv2d13: Conv2d<B>,
    conv2d14: Conv2d<B>,
    conv2d15: Conv2d<B>,
    conv2d16: Conv2d<B>,
    conv2d17: Conv2d<B>,
    phantom: core::marker::PhantomData<B>,
}

impl<B: Backend> Default for Model<B> {
    fn default() -> Self {
        let record =
            burn::record::DefaultFileRecorder::<burn::record::FullPrecisionSettings>::new()
                .load("./csrnet/src/model/csrnet".into())
                .expect("Record file to exist.");
        Self::new_with(record)
    }
}

impl<B: Backend> Model<B> {
    #[allow(unused_variables)]
    pub fn new_with(record: ModelRecord<B>) -> Self {
        let conv2d1 = Conv2dConfig::new([3, 64], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init_with(record.conv2d1);
        let conv2d2 = Conv2dConfig::new([64, 64], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init_with(record.conv2d2);
        let maxpool2d1 = MaxPool2dConfig::new([2, 2])
            .with_strides([2, 2])
            .with_padding(PaddingConfig2d::Valid)
            .init();
        let conv2d3 = Conv2dConfig::new([64, 128], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init_with(record.conv2d3);
        let conv2d4 = Conv2dConfig::new([128, 128], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init_with(record.conv2d4);
        let maxpool2d2 = MaxPool2dConfig::new([2, 2])
            .with_strides([2, 2])
            .with_padding(PaddingConfig2d::Valid)
            .init();
        let conv2d5 = Conv2dConfig::new([128, 256], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init_with(record.conv2d5);
        let conv2d6 = Conv2dConfig::new([256, 256], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init_with(record.conv2d6);
        let conv2d7 = Conv2dConfig::new([256, 256], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init_with(record.conv2d7);
        let maxpool2d3 = MaxPool2dConfig::new([2, 2])
            .with_strides([2, 2])
            .with_padding(PaddingConfig2d::Valid)
            .init();
        let conv2d8 = Conv2dConfig::new([256, 512], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init_with(record.conv2d8);
        let conv2d9 = Conv2dConfig::new([512, 512], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init_with(record.conv2d9);
        let conv2d10 = Conv2dConfig::new([512, 512], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init_with(record.conv2d10);
        let conv2d11 = Conv2dConfig::new([512, 512], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(2, 2))
            .with_dilation([2, 2])
            .with_groups(1)
            .with_bias(true)
            .init_with(record.conv2d11);
        let conv2d12 = Conv2dConfig::new([512, 512], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(2, 2))
            .with_dilation([2, 2])
            .with_groups(1)
            .with_bias(true)
            .init_with(record.conv2d12);
        let conv2d13 = Conv2dConfig::new([512, 512], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(2, 2))
            .with_dilation([2, 2])
            .with_groups(1)
            .with_bias(true)
            .init_with(record.conv2d13);
        let conv2d14 = Conv2dConfig::new([512, 256], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(2, 2))
            .with_dilation([2, 2])
            .with_groups(1)
            .with_bias(true)
            .init_with(record.conv2d14);
        let conv2d15 = Conv2dConfig::new([256, 128], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(2, 2))
            .with_dilation([2, 2])
            .with_groups(1)
            .with_bias(true)
            .init_with(record.conv2d15);
        let conv2d16 = Conv2dConfig::new([128, 64], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(2, 2))
            .with_dilation([2, 2])
            .with_groups(1)
            .with_bias(true)
            .init_with(record.conv2d16);
        let conv2d17 = Conv2dConfig::new([64, 1], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init_with(record.conv2d17);
        Self {
            conv2d1,
            conv2d2,
            maxpool2d1,
            conv2d3,
            conv2d4,
            maxpool2d2,
            conv2d5,
            conv2d6,
            conv2d7,
            maxpool2d3,
            conv2d8,
            conv2d9,
            conv2d10,
            conv2d11,
            conv2d12,
            conv2d13,
            conv2d14,
            conv2d15,
            conv2d16,
            conv2d17,
            phantom: core::marker::PhantomData,
        }
    }

    #[allow(dead_code)]
    pub fn new() -> Self {
        let conv2d1 = Conv2dConfig::new([3, 64], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init();
        let conv2d2 = Conv2dConfig::new([64, 64], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init();
        let maxpool2d1 = MaxPool2dConfig::new([2, 2])
            .with_strides([2, 2])
            .with_padding(PaddingConfig2d::Valid)
            .init();
        let conv2d3 = Conv2dConfig::new([64, 128], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init();
        let conv2d4 = Conv2dConfig::new([128, 128], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init();
        let maxpool2d2 = MaxPool2dConfig::new([2, 2])
            .with_strides([2, 2])
            .with_padding(PaddingConfig2d::Valid)
            .init();
        let conv2d5 = Conv2dConfig::new([128, 256], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init();
        let conv2d6 = Conv2dConfig::new([256, 256], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init();
        let conv2d7 = Conv2dConfig::new([256, 256], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init();
        let maxpool2d3 = MaxPool2dConfig::new([2, 2])
            .with_strides([2, 2])
            .with_padding(PaddingConfig2d::Valid)
            .init();
        let conv2d8 = Conv2dConfig::new([256, 512], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init();
        let conv2d9 = Conv2dConfig::new([512, 512], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init();
        let conv2d10 = Conv2dConfig::new([512, 512], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init();
        let conv2d11 = Conv2dConfig::new([512, 512], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(2, 2))
            .with_dilation([2, 2])
            .with_groups(1)
            .with_bias(true)
            .init();
        let conv2d12 = Conv2dConfig::new([512, 512], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(2, 2))
            .with_dilation([2, 2])
            .with_groups(1)
            .with_bias(true)
            .init();
        let conv2d13 = Conv2dConfig::new([512, 512], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(2, 2))
            .with_dilation([2, 2])
            .with_groups(1)
            .with_bias(true)
            .init();
        let conv2d14 = Conv2dConfig::new([512, 256], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(2, 2))
            .with_dilation([2, 2])
            .with_groups(1)
            .with_bias(true)
            .init();
        let conv2d15 = Conv2dConfig::new([256, 128], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(2, 2))
            .with_dilation([2, 2])
            .with_groups(1)
            .with_bias(true)
            .init();
        let conv2d16 = Conv2dConfig::new([128, 64], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(2, 2))
            .with_dilation([2, 2])
            .with_groups(1)
            .with_bias(true)
            .init();
        let conv2d17 = Conv2dConfig::new([64, 1], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init();
        Self {
            conv2d1,
            conv2d2,
            maxpool2d1,
            conv2d3,
            conv2d4,
            maxpool2d2,
            conv2d5,
            conv2d6,
            conv2d7,
            maxpool2d3,
            conv2d8,
            conv2d9,
            conv2d10,
            conv2d11,
            conv2d12,
            conv2d13,
            conv2d14,
            conv2d15,
            conv2d16,
            conv2d17,
            phantom: core::marker::PhantomData,
        }
    }

    #[allow(clippy::let_and_return)]
    pub fn forward(&self, input1: Tensor<B, 4>) -> Tensor<B, 4> {
        let conv2d1_out1 = self.conv2d1.forward(input1);
        let relu1_out1 = burn::tensor::activation::relu(conv2d1_out1);
        let conv2d2_out1 = self.conv2d2.forward(relu1_out1);
        let relu2_out1 = burn::tensor::activation::relu(conv2d2_out1);
        let maxpool2d1_out1 = self.maxpool2d1.forward(relu2_out1);
        let conv2d3_out1 = self.conv2d3.forward(maxpool2d1_out1);
        let relu3_out1 = burn::tensor::activation::relu(conv2d3_out1);
        let conv2d4_out1 = self.conv2d4.forward(relu3_out1);
        let relu4_out1 = burn::tensor::activation::relu(conv2d4_out1);
        let maxpool2d2_out1 = self.maxpool2d2.forward(relu4_out1);
        let conv2d5_out1 = self.conv2d5.forward(maxpool2d2_out1);
        let relu5_out1 = burn::tensor::activation::relu(conv2d5_out1);
        let conv2d6_out1 = self.conv2d6.forward(relu5_out1);
        let relu6_out1 = burn::tensor::activation::relu(conv2d6_out1);
        let conv2d7_out1 = self.conv2d7.forward(relu6_out1);
        let relu7_out1 = burn::tensor::activation::relu(conv2d7_out1);
        let maxpool2d3_out1 = self.maxpool2d3.forward(relu7_out1);
        let conv2d8_out1 = self.conv2d8.forward(maxpool2d3_out1);
        let relu8_out1 = burn::tensor::activation::relu(conv2d8_out1);
        let conv2d9_out1 = self.conv2d9.forward(relu8_out1);
        let relu9_out1 = burn::tensor::activation::relu(conv2d9_out1);
        let conv2d10_out1 = self.conv2d10.forward(relu9_out1);
        let relu10_out1 = burn::tensor::activation::relu(conv2d10_out1);
        let conv2d11_out1 = self.conv2d11.forward(relu10_out1);
        let relu11_out1 = burn::tensor::activation::relu(conv2d11_out1);
        let conv2d12_out1 = self.conv2d12.forward(relu11_out1);
        let relu12_out1 = burn::tensor::activation::relu(conv2d12_out1);
        let conv2d13_out1 = self.conv2d13.forward(relu12_out1);
        let relu13_out1 = burn::tensor::activation::relu(conv2d13_out1);
        let conv2d14_out1 = self.conv2d14.forward(relu13_out1);
        let relu14_out1 = burn::tensor::activation::relu(conv2d14_out1);
        let conv2d15_out1 = self.conv2d15.forward(relu14_out1);
        let relu15_out1 = burn::tensor::activation::relu(conv2d15_out1);
        let conv2d16_out1 = self.conv2d16.forward(relu15_out1);
        let relu16_out1 = burn::tensor::activation::relu(conv2d16_out1);
        let conv2d17_out1 = self.conv2d17.forward(relu16_out1);
        conv2d17_out1
    }
}

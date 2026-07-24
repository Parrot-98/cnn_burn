use burn::{
    nn::{
        conv::{Conv2d, Conv2dConfig},
        pool::{
            AdaptiveAvgPool2d,
            AdaptiveAvgPool2dConfig,
            MaxPool2d,
            MaxPool2dConfig,
        },
        BatchNorm,
        BatchNormConfig,
        Linear,
        LinearConfig,
        Relu,
    },
    module::Module,
    tensor::{backend::Backend, Tensor},
};

#[derive(Module, Debug)]
pub struct ConvNet<B: Backend> {
    conv1: Conv2d<B>,
    bn1: BatchNorm<B, 2>,

    conv2: Conv2d<B>,
    bn2: BatchNorm<B, 2>,

    conv3: Conv2d<B>,
    bn3: BatchNorm<B, 2>,

    conv4: Conv2d<B>,
    bn4: BatchNorm<B, 2>,

    pool: MaxPool2d,
    adaptive_pool: AdaptiveAvgPool2d,
    relu: Relu,

    fc1: Linear<B>,
    fc2: Linear<B>,
}

impl<B: Backend> ConvNet<B> {
    pub fn new(num_classes: usize, device: &B::Device) -> Self {
        let conv1 =
            Conv2dConfig::new([3, 32], [3, 3])
                .with_padding(
                    burn::nn::PaddingConfig2d::Explicit(1, 1)
                )
                .init(device);

        let bn1 =
            BatchNormConfig::new(32)
                .init(device);

        let conv2 =
            Conv2dConfig::new([32, 64], [3, 3])
                .with_padding(
                    burn::nn::PaddingConfig2d::Explicit(1, 1)
                )
                .init(device);

        let bn2 =
            BatchNormConfig::new(64)
                .init(device);

        let conv3 =
            Conv2dConfig::new([64, 128], [3, 3])
                .with_padding(
                    burn::nn::PaddingConfig2d::Explicit(1, 1)
                )
                .init(device);

        let bn3 =
            BatchNormConfig::new(128)
                .init(device);

        let conv4 =
            Conv2dConfig::new([128, 256], [3, 3])
                .with_padding(
                    burn::nn::PaddingConfig2d::Explicit(1, 1)
                )
                .init(device);

        let bn4 =
            BatchNormConfig::new(256)
                .init(device);

        let pool = MaxPool2dConfig::new([2, 2])
            .with_strides([2, 2])
            .init();

        let adaptive_pool =
            AdaptiveAvgPool2dConfig::new([1, 1]).init();

        let relu = Relu::new();

        let fc1 = LinearConfig::new(256, 512)
            .init(device);

        let fc2 = LinearConfig::new(512, num_classes)
            .init(device);

        Self {
            conv1,
            bn1,
            conv2,
            bn2,
            conv3,
            bn3,
            conv4,
            bn4,
            pool,
            adaptive_pool,
            relu,
            fc1,
            fc2,
        }
    }

    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 2> {
        let x = self.conv1.forward(x);
        let x = self.bn1.forward(x);
        let x = self.relu.forward(x);
        let x = self.pool.forward(x);

        let x = self.conv2.forward(x);
        let x = self.bn2.forward(x);
        let x = self.relu.forward(x);
        let x = self.pool.forward(x);

        let x = self.conv3.forward(x);
        let x = self.bn3.forward(x);
        let x = self.relu.forward(x);
        let x = self.pool.forward(x);

        let x = self.conv4.forward(x);
        let x = self.bn4.forward(x);
        let x = self.relu.forward(x);
        let x = self.pool.forward(x);

        let x = self.adaptive_pool.forward(x);

        let x = x.flatten(1, 3);

        let x = self.fc1.forward(x);
        let x = self.relu.forward(x);

        self.fc2.forward(x)
    }
}
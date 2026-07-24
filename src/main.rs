mod data;
mod model;
mod train;

use crate::data::{ImageBatcher, PlainTarDataset};
use crate::model::ConvNet;

use burn::{
    backend::{
        autodiff::Autodiff,
        wgpu::{Wgpu, WgpuDevice},
    },
    data::dataloader::DataLoaderBuilder,
    optim::AdamConfig,
    train::{
        metric::{AccuracyMetric, LossMetric},
        ClassificationOutput, LearnerBuilder,
    },
};

fn main() {
    type MyBackend = Wgpu<f32, i32>;
    type MyAutodiffBackend = Autodiff<MyBackend>;

    let device = WgpuDevice::default();

    println!("Initializing datasets from tar files...");
    let dataset = PlainTarDataset::new_from_huggingface("validation", 64);

    // Utilize your i9 with 12 workers
    let batcher_train = ImageBatcher::<MyAutodiffBackend>::new(device.clone());
    let dataloader_train = DataLoaderBuilder::new(batcher_train)
        .batch_size(8)
        .shuffle(42)
        .num_workers(12)
        .build(dataset.clone());

    let batcher_val = ImageBatcher::<MyBackend>::new(device.clone());
    let dataloader_val = DataLoaderBuilder::new(batcher_val)
        .batch_size(8)
        .num_workers(12)
        .build(dataset);

    let model = ConvNet::<MyAutodiffBackend>::new(1000, &device);
    let optimizer = AdamConfig::new().init();

    // TextRenderer outputs standard line-by-line text to fix black-screen issues
    let learner = LearnerBuilder::<
        MyAutodiffBackend,
        ClassificationOutput<MyAutodiffBackend>,
        ClassificationOutput<MyBackend>,
        _,
        _,
        _,
    >::new("./artifacts")
        .metric_train_numeric(LossMetric::new())
        .metric_train_numeric(AccuracyMetric::new())
        .metric_valid_numeric(LossMetric::new())
        .metric_valid_numeric(AccuracyMetric::new())
        .with_file_checkpointer(burn::record::CompactRecorder::new())
        .devices(vec![device.clone()])
        .num_epochs(5)
        .build(model, optimizer, 3e-4);

    println!("\nStarting training loop...");
    let _trained_model = learner.fit(dataloader_train, dataloader_val);
}
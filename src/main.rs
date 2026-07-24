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

    println!("===== NEW MAIN.RS =====");
    println!("Initializing datasets from tar files...");

    let mut dataset =
        PlainTarDataset::new_from_huggingface("validation", 1);

    dataset.items.truncate(16);

    println!(
        "Dataset contains {} indexed images",
        dataset.items.len()
    );

    if dataset.items.is_empty() {
        eprintln!("Dataset is empty. Training cannot start.");
        return;
    }

    let batcher_train =
        ImageBatcher::<MyAutodiffBackend>::new(device.clone());

    let dataloader_train = DataLoaderBuilder::new(batcher_train)
        .batch_size(1)
        .shuffle(42)
        .num_workers(0)
        .build(dataset.clone());

    let batcher_val =
        ImageBatcher::<MyBackend>::new(device.clone());

    let dataloader_val = DataLoaderBuilder::new(batcher_val)
        .batch_size(1)
        .num_workers(0)
        .build(dataset);

    println!("Creating model...");

    let model =
        ConvNet::<MyAutodiffBackend>::new(1000, &device);

    println!("Creating optimizer...");

    let optimizer = AdamConfig::new().init();

    println!("Creating learner...");

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
        .with_file_checkpointer(
            burn::record::CompactRecorder::new(),
        )
        .devices(vec![device.clone()])
        .num_epochs(1)
        .build(model, optimizer, 3e-4);

    println!("Starting training loop...");

    let _trained_model =
        learner.fit(dataloader_train, dataloader_val);

    println!("Training finished.");
}
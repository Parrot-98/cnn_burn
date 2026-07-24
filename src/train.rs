use crate::data::ImageBatch;
use crate::model::ConvNet;
use burn::{
    nn::loss::CrossEntropyLossConfig,
    tensor::backend::{AutodiffBackend, Backend},
    train::{
        ClassificationOutput,
        TrainOutput,
        TrainStep,
        ValidStep,
    },
};

impl<B: AutodiffBackend> TrainStep<ImageBatch<B>, ClassificationOutput<B>> for ConvNet<B> {
    fn step(&self, batch: ImageBatch<B>) -> TrainOutput<ClassificationOutput<B>> {
        let output = self.forward(batch.images);
        let loss = CrossEntropyLossConfig::new()
            .init(&output.device())
            .forward(output.clone(), batch.targets.clone());

        let grads = loss.backward();

        TrainOutput::new(
            self,
            grads,
            ClassificationOutput::new(loss, output, batch.targets),
        )
    }
}

impl<B: Backend> ValidStep<ImageBatch<B>, ClassificationOutput<B>> for ConvNet<B> {
    fn step(&self, batch: ImageBatch<B>) -> ClassificationOutput<B> {
        let output = self.forward(batch.images);
        let loss = CrossEntropyLossConfig::new()
            .init(&output.device())
            .forward(output.clone(), batch.targets.clone());

        ClassificationOutput::new(loss, output, batch.targets)
    }
}
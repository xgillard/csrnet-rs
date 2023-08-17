use burn::{config::Config, tensor::{backend::{ADBackend, Backend}, Tensor}, optim::{AdamConfig, decay::WeightDecayConfig}, data::dataloader::DataLoaderBuilder, record::{Recorder, NoStdTrainingRecorder}, module::Module, train::{LearnerBuilder, metric::{LossMetric, Adaptor, LossInput}, TrainStep, TrainOutput, ValidStep}, nn::loss::{MSELoss, Reduction}};

use crate::{data::{CsrnetBatcher, CsrnetDataset, CsrnetBatch}, model::csrnet::Model, utils::model};


#[derive(Config, Default)]
pub struct CsrnetTrainingConfig {
    pub model_file: Option<String>,
    pub checkpoints: String, 
    pub output: String, 

    pub train: String,
    pub validation: String,

    #[config(default = 10)]
    pub num_epochs: usize,

    #[config(default = 5e-5)]
    pub decay_rate: f64,

    #[config(default = 4)]
    pub batch_size: usize,

    #[config(default = 4)]
    pub num_workers: usize,

    #[config(default = 42)]
    pub seed: u64,
}

pub fn run<B: ADBackend<FloatElem = f32>>(config: CsrnetTrainingConfig, device: B::Device) {
    // Config
    let optimizer = AdamConfig::new()
        .with_epsilon(1e-8)
        .with_weight_decay(Some(WeightDecayConfig::new(config.decay_rate)));
    B::seed(config.seed);

    // Data
    let batcher_train = CsrnetBatcher;
    // innerbackend => equivalent a no_grad()
    let batcher_valid = CsrnetBatcher;

    let dataloader_train = DataLoaderBuilder::new(batcher_train)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(CsrnetDataset::new(device.clone(), &config.train));

    let dataloader_test = DataLoaderBuilder::new(batcher_valid)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(CsrnetDataset::new(device.clone(), &config.validation));

    // Model
    let model = model::<B>(&config.model_file);

    let learner = LearnerBuilder::new(&config.checkpoints)
        .metric_train_plot(LossMetric::new())
        .metric_valid_plot(LossMetric::new())
        .with_file_checkpointer(3, NoStdTrainingRecorder::new())
        .devices(vec![device])
        .num_epochs(config.num_epochs)
        .build(model, optimizer.init(), 1e-4);

    let model_trained = learner.fit(dataloader_train, dataloader_test);

    config
        .save(format!("{}/config.json", config.output).as_str())
        .unwrap();

    NoStdTrainingRecorder::new()
        .record(
            model_trained.into_record(),
            format!("{}/model", config.output).into(),
        )
        .expect("Failed to save trained model");
}


/// Simple regression output adapted for multiple metrics.
pub struct CsrnetTrainingOutput<B: Backend> {
    /// The loss.
    pub loss: Tensor<B, 1>,

    /// The output.
    pub output: Tensor<B, 4>,

    /// The targets.
    pub targets: Tensor<B, 4>,
}

impl<B: Backend> Adaptor<LossInput<B>> for CsrnetTrainingOutput<B> {
    fn adapt(&self) -> LossInput<B> {
        LossInput::new(self.loss.clone().sum())
    }
}

impl <B: Backend> Model<B> {
    pub fn forward_pass(&self, item: CsrnetBatch<B>) -> CsrnetTrainingOutput<B> {
        let targets = item.labels;
        let output = self.forward(item.images);
        let loss = MSELoss::new();
        let loss = loss.forward(output.clone(), targets.clone(), Reduction::Sum);

        CsrnetTrainingOutput {
            loss,
            output,
            targets,
        }
    }
}

impl<B: ADBackend> TrainStep<CsrnetBatch<B>, CsrnetTrainingOutput<B>> for Model<B> {
    fn step(&self, item: CsrnetBatch<B>) -> TrainOutput<CsrnetTrainingOutput<B>> {
        let item = self.forward_pass(item);

        TrainOutput::new(self, item.loss.backward(), item)
    }
}

impl<B: Backend> ValidStep<CsrnetBatch<B>, CsrnetTrainingOutput<B>> for Model<B> {
    fn step(&self, item: CsrnetBatch<B>) -> CsrnetTrainingOutput<B> {
        self.forward_pass(item)
    }
}

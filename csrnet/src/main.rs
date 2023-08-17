use burn::tensor::backend::Backend;
use burn_autodiff::ADBackendDecorator;
#[cfg(feature = "cpu")]
use burn_ndarray::{NdArrayBackend, NdArrayDevice};
#[cfg(feature = "tch")]
use burn_tch::{TchBackend, TchDevice};
#[cfg(feature = "wgpu")]
use burn_wgpu::{WgpuBackend, WgpuDevice, AutoGraphicsApi};

pub mod model;
pub mod data;
pub mod train;
pub mod utils;

use structopt::StructOpt;
use train::CsrnetTrainingConfig;

#[cfg(feature = "cpu")]
type CCBackend = NdArrayBackend<f32>;
#[cfg(feature = "wgpu")]
type CCBackend = WgpuBackend<AutoGraphicsApi, f32, i32>;
#[cfg(feature = "tch")]
type CCBackend = TchBackend<f32>;

fn device() -> <CCBackend as Backend>::Device {
    #[cfg(feature = "cpu")]
    let dev = NdArrayDevice::default();
    #[cfg(feature = "wgpu")]
    let dev = WgpuDevice::default();
    #[cfg(all(feature = "tch", target_os = "macos"))]
    let dev = TchDevice::Mps;
    #[cfg(all(feature = "tch", not(target_os = "macos")))]
    let dev = TchDevice::Vulkan;

    dev
}

/// The model to perform crowdcounting -- count the number of persons in the
/// picture of a crowd.
#[derive(Debug, structopt::StructOpt)]
enum Args {
    /// Uses the trained model to perform inference (that is: actually count people in an image)
    Infer {
        /// The model to use
        #[structopt(short, long)]
        model: Option<String>,
        /// The image whose number of people should be counted.
        #[structopt(short, long)]
        image: String,
    },
    /// Train the model
    Train {
        /// The model to use
        #[structopt(short, long)]
        model: Option<String>,
        /// Path to the training dataset
        #[structopt(short, long)]
        train: String,
        /// Path to the validation dataset
        #[structopt(short, long)]
        validation: String,
        /// Minibatch size to use during training
        #[structopt(short, long, default_value="1")]
        batch_size: usize,
        /// Number of epoch during with the training should be performed
        #[structopt(short, long, default_value="10")]
        epochs: usize,
        /// A seed for the prng
        #[structopt(short, long, default_value="42")]
        seed: u64,
    },
    /// Check the value contained in a ground truth file
    Check {
        /// path to a ground truth h5 file
        ground_truth: String,
    }
}

fn main() {
    let args = Args::from_args();
    match &args {
        Args::Infer { model, image } => {
            let model = utils::model::<CCBackend>(model);
            let tensor = utils::prepare_image(image, true);

            let output = model.forward(tensor.unsqueeze());
            let output = output.sum().into_scalar();
            println!("{output:?}");
        },
        Args::Train { model, train, validation, batch_size, epochs, seed } => {
            let config = CsrnetTrainingConfig {
                model_file : model.clone(), 
                checkpoints: "./artifacts/checkpoints/".to_string(), 
                output:  "./outputs/".to_string(), 
                
                train: train.clone(),
                validation: validation.clone(),

                batch_size: *batch_size,
                num_epochs: *epochs,
                num_workers: 4,
                seed: *seed,
                ..Default::default()
            };
            train::run::<ADBackendDecorator<CCBackend>>(config, device());
        },
        Args::Check { ground_truth } => {
            let value: f32 = utils::read_density_map::<CCBackend, &String>(ground_truth).sum().into_scalar();
            println!("{value:?}");
        }
    }
}

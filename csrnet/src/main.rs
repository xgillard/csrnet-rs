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

use model::csrnet;
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
        /// The image whose number of people should be counted.
        #[structopt(short, long)]
        image: String
    },
    /// Work in progress
    Train {
        path: String
    },
}

fn main() {
    let args = Args::from_args();
    match &args {
        Args::Infer { image } => {
            let model = csrnet::Model::<CCBackend>::default();
            let tensor = utils::prepare_image(image);

            let output = model.forward(tensor.unsqueeze());
            let output = output.sum().into_scalar();
            println!("{output:?}");
        },
        Args::Train { path } => {
            let config = CsrnetTrainingConfig {
                model_file : None, 
                checkpoints: "./artifacts/checkpoints/".to_string(), 
                output:  "./outputs/".to_string(), 
                
                train: path.clone(),
                validation: path.clone(),

                batch_size: 1,
                num_epochs: 10,
                num_workers: 4,
                seed: 42,
                ..Default::default()
            };
            train::run::<ADBackendDecorator<CCBackend>>(config, device());
        }
    }
}

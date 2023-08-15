use burn::tensor::{Tensor, Shape, backend::Backend, Data};
use burn_wgpu::{WgpuBackend, AutoGraphicsApi};

pub mod model;
use image::{GenericImageView, imageops::FilterType};
use model::csrnet;
use structopt::StructOpt;

type CCBackend = WgpuBackend<AutoGraphicsApi, f32, i32>;

/// Returns the number of people that have been counted in an image
#[derive(Debug, structopt::StructOpt)]
struct Args {
    /// The image whose number of people should be counted.
    #[structopt(short, long)]
    image: String
}

fn prepare_image<B: Backend>(path: &str) -> Tensor<B, 4> {
    let im = image::open(path).expect("open image");
    let (w, h) = im.dimensions();
    let im = im.resize(9 * w / 10, 9 * h / 10, FilterType::Nearest);
    let (w, h) = im.dimensions();

    let im = im.to_rgb8();
    let w = w as usize;
    let h = h as usize;
    //
    let mut data = vec![];

    // la normalisation n'est pas non plus utilisee dans le papier original
    // meme si elle est encore presente dans le code (mais pas utilisee).
    //
    const MEAN_MAGIC: [f32;3] = [0.485, 0.456, 0.406];
    const STD_MAGIC:  [f32;3] = [0.229, 0.224, 0.225];
    const SUB_MAGIC:  [f32;3] = [92.8207477031, 95.2757037428, 104.877445883];
    for c in 0..3 {
        for y in 0..h  {
            for x in 0..w {
                let pixel = im.get_pixel(x as u32, y as u32);
                let pixel = f32::from(pixel[c]);
                // la normalisation n'est pas non plus utilisee dans le papier 
                // original meme si elle est encore presente dans le code 
                // (mais pas utilisee).
                //
                //let pixel = pixel / 255.0;
                //let pixel = (pixel - MEAN_MAGIC[c]) / STD_MAGIC[c];
                //let pixel = pixel * 255.0;
                let pixel = pixel - SUB_MAGIC[c];
                data.push(pixel);
            }
        }
    }

    // todo subtract some value from each dim
    let tensor = Tensor::from_floats(Data::from(data.as_slice()))
        .reshape(Shape::from([3, h, w]));

    tensor.unsqueeze()
}

fn main() {
    let args = Args::from_args();
    let model = csrnet::Model::<CCBackend>::default();
    let tensor = prepare_image::<CCBackend>(&args.image);

    let output: Tensor<WgpuBackend<AutoGraphicsApi, f32, i32>, 4> = model.forward(tensor);
    let output = output.sum().into_scalar();
    println!("{output:?}");
}

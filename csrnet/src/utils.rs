use std::path::Path;

use burn::tensor::{Tensor, Shape, backend::Backend, Data};
use hdf5::File;
use image::GenericImageView;

pub fn prepare_image<B: Backend, P: AsRef<Path>>(path: P) -> Tensor<B, 3> {
    let im = image::open(path).expect("open image");
    let (w, h) = im.dimensions();

    let im = im.to_rgb8();
    let w = w as usize;
    let h = h as usize;
    //
    let mut data = vec![];

    // la normalisation n'est pas non plus utilisee dans le papier original
    // meme si elle est encore presente dans le code (mais pas utilisee).
    //
    //const MEAN_MAGIC: [f32;3] = [0.485, 0.456, 0.406];
    //const STD_MAGIC:  [f32;3] = [0.229, 0.224, 0.225];
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

    let tensor = Tensor::from_floats(Data::from(data.as_slice()))
        .reshape(Shape::from([3, h, w]));

    tensor
}

pub fn read_density_map<B, P: AsRef<Path>>(path: P) -> Tensor<B, 3> 
where B: Backend<FloatElem = f32>
{
    let dsm = File::open(path).expect("cannot open hdf5 file");
    let dsm = dsm.dataset("density").expect("cannot list attrs");
    let dsm = dsm.read_2d::<f32>().expect("cannot read density map");
    let shape = dsm.shape();

    let data = Data::from(dsm.as_slice().unwrap());
    let tensor = Tensor::from_data(data);
    let tensor = tensor.reshape([1, shape[0], shape[1]]);
    tensor
}
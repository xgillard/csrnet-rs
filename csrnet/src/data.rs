use std::{path::{Path, PathBuf}, fs};

use burn::{tensor::{backend::Backend, Tensor}, data::{dataloader::batcher::Batcher, dataset::{transform::{Mapper, MapperDataset}, Dataset}}};

use crate::utils;

#[derive(Debug, Clone)]
pub struct CsrnetItem<B: Backend> {
    pub image:       Tensor<B, 3>,
    pub density_map: Tensor<B, 3>
}

#[derive(Debug, Clone)]
pub struct FileCsrnetDataset {
    items: Vec<(PathBuf, PathBuf)>
}
impl FileCsrnetDataset {
    pub fn new<P: AsRef<Path>>(path_to_ds: P) -> Self {
        let mut items = vec![];

        let imdir = PathBuf::from(path_to_ds.as_ref()).join("images");
        let gtdir = PathBuf::from(path_to_ds.as_ref()).join("ground_truth");

        let imdir = fs::read_dir(imdir).expect("could not read directory");
        for image in imdir {
            if let Ok(image) = image {
                let impath = image.path();
                let stem = impath.file_stem().unwrap();
                let h5name = format!("GT_{}.h5", stem.to_str().unwrap());

                let h5path = gtdir.join(h5name);
                items.push((impath, h5path));
            }
        }

        Self { items }
    }
}

impl Dataset<(PathBuf, PathBuf)> for FileCsrnetDataset {
    fn get(&self, index: usize) -> Option<(PathBuf, PathBuf)> {
        self.items.get(index).cloned()
    }

    fn len(&self) -> usize {
        self.items.len()
    }
}

#[derive(Debug, Clone)]
pub struct ToCsrnetItem<B: Backend> {
    pub device: B::Device
}

impl <B> Mapper<(PathBuf, PathBuf), CsrnetItem<B>> for ToCsrnetItem<B>
where B: Backend<FloatElem = f32>
{
    fn map(&self, item: &(PathBuf, PathBuf)) -> CsrnetItem<B> {
        let im = utils::prepare_image(&item.0, true).to_device(&self.device);
        let h5 = utils::read_density_map(&item.1).to_device(&self.device);
        CsrnetItem { image: im, density_map: h5 }
    }
}

type MappedDataset<B> = MapperDataset<FileCsrnetDataset, ToCsrnetItem<B>, (PathBuf, PathBuf)>;
pub struct CsrnetDataset<B: Backend> {
    dataset: MappedDataset<B>
}
impl <B: Backend> CsrnetDataset<B> {
    pub fn new<P: AsRef<Path>>(device: B::Device, path: P) -> Self {
        let dataset = MapperDataset::new(FileCsrnetDataset::new(path), ToCsrnetItem{device});
        Self {dataset}
    }
}
impl <B: Backend<FloatElem = f32>> Dataset<CsrnetItem<B>> for CsrnetDataset<B> {
    fn get(&self, index: usize) -> Option<CsrnetItem<B>> {
        self.dataset.get(index)
    }

    fn len(&self) -> usize {
        <MapperDataset<FileCsrnetDataset, ToCsrnetItem<B>, (PathBuf, PathBuf)> as burn::data::dataloader::Dataset<CsrnetItem<B>>>::len(&self.dataset)
    }
}

#[derive(Debug, Clone)]
pub struct CsrnetBatch<B: Backend> {
    pub images: Tensor<B, 4>,
    pub labels: Tensor<B, 4>
}

#[derive(Debug, Clone, Copy)]
pub struct CsrnetBatcher;
impl <B: Backend> Batcher<CsrnetItem<B>, CsrnetBatch<B>> for CsrnetBatcher {
    fn batch(&self, items: Vec<CsrnetItem<B>>) -> CsrnetBatch<B> {
        let mut images = vec![];
        let mut labels = vec![];
        for CsrnetItem { image, density_map } in items {
            images.push(image.unsqueeze());
            labels.push(density_map.unsqueeze());
        }
        let images = Tensor::cat(images, 0);
        let labels = Tensor::cat(labels, 0);

        CsrnetBatch{images, labels}
    }
}
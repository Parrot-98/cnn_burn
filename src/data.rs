use burn::{
    data::dataloader::batcher::Batcher,
    data::dataset::Dataset,
    tensor::{backend::Backend, Tensor, TensorData},
};

use image::ImageReader;
use serde::{Deserialize, Serialize};

use std::{
    collections::{HashMap, HashSet},
    fs::File,
    io::{BufReader, BufWriter, Cursor, Read},
    path::Path,
};

use tar::Archive;

#[derive(Clone, Debug)]
pub struct ImageItem {
    pub image_data: Vec<f32>,
    pub label: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RemoteTarEntry {
    pub shard_url: String,
    pub filename: String,
    pub label: usize,
}

#[derive(Clone)]
pub struct PlainTarDataset {
    pub items: Vec<RemoteTarEntry>,
}

impl PlainTarDataset {
    /// Index remote WebDataset shards from Hugging Face over HTTP without saving to disk.
    ///
    /// - `prefix`: "validation" (64 shards) or "train" (1024 shards)
    /// - `num_shards`: Number of `.tar` shards to index
    pub fn new_from_huggingface(prefix: &str, num_shards: usize) -> Self {
        let cache_path = Path::new("hf_dataset_index.json");

        // Load cached index if present locally to save network calls on restart
        if cache_path.exists() {
            println!("Loading cached dataset index from {:?}...", cache_path);
            if let Ok(file) = File::open(cache_path) {
                let reader = BufReader::new(file);
                if let Ok(items) = serde_json::from_reader::<_, Vec<RemoteTarEntry>>(reader) {
                    println!("Successfully loaded cached index with {} images!", items.len());
                    return Self { items };
                }
            }
        }

        println!("Indexing remote Hugging Face WebDataset shards ({}) ...", num_shards);

        let mut raw_items: Vec<(String, String, String)> = Vec::new();
        let mut class_names: HashSet<String> = HashSet::new();

        let token = std::env::var("HF_TOKEN").unwrap_or_default();
        let client = reqwest::blocking::Client::new();

        for shard_idx in 0..num_shards {
            let shard_url = format!(
                "https://huggingface.co/datasets/timm/imagenet-1k-wds/resolve/main/imagenet1k-{}-{:04}.tar",
                prefix, shard_idx
            );

            println!("Indexing remote shard: {}...", shard_url);

            let mut request = client.get(&shard_url);
            if !token.is_empty() {
                request = request.bearer_auth(&token);
            }

            let Ok(response) = request.send() else {
                println!("Failed to connect to {}", shard_url);
                continue;
            };

            let mut archive = Archive::new(response);

            let Ok(entries) = archive.entries() else {
                println!("Could not read remote archive {}", shard_url);
                continue;
            };

            for tar_entry in entries.flatten() {
                let Ok(file_path) = tar_entry.path() else {
                    continue;
                };

                let filename = file_path.to_string_lossy().into_owned();

                // WebDataset stores images as .jpg/.jpeg/.png
                if !filename.ends_with(".jpg")
                    && !filename.ends_with(".jpeg")
                    && !filename.ends_with(".png")
                    && !filename.ends_with(".JPEG")
                {
                    continue;
                }

                let Some(file_name) = Path::new(&filename).file_name() else {
                    continue;
                };

                let clean_filename = file_name.to_string_lossy();
                let class_key = clean_filename
                    .split('_')
                    .next()
                    .unwrap_or("");

                if !class_key.starts_with('n') {
                    continue;
                }

                let class_key = class_key.to_string();
                class_names.insert(class_key.clone());

                raw_items.push((shard_url.clone(), filename, class_key));
            }
        }

        println!("Found {} unique classes across remote shards!", class_names.len());

        let mut sorted_classes: Vec<String> = class_names.into_iter().collect();
        sorted_classes.sort();

        let class_map: HashMap<String, usize> = sorted_classes
            .into_iter()
            .enumerate()
            .map(|(label, class_name)| (class_name, label))
            .collect();

        let items: Vec<RemoteTarEntry> = raw_items
            .into_iter()
            .filter_map(|(shard_url, filename, class_key)| {
                let label = class_map.get(&class_key)?;
                Some(RemoteTarEntry {
                    shard_url,
                    filename,
                    label: *label,
                })
            })
            .collect();

        // Save local JSON index so indexing only happens once
        if let Ok(file) = File::create(cache_path) {
            let writer = BufWriter::new(file);
            let _ = serde_json::to_writer(writer, &items);
        }

        Self { items }
    }

    pub fn split(&self, train_ratio: f32) -> (Self, Self) {
        let split_index = (self.items.len() as f32 * train_ratio) as usize;
        let train_items = self.items[..split_index].to_vec();
        let valid_items = self.items[split_index..].to_vec();

        (
            Self { items: train_items },
            Self { items: valid_items },
        )
    }

    fn read_image_from_stream(shard_url: &str, filename: &str) -> Option<Vec<f32>> {
        let token = std::env::var("HF_TOKEN").unwrap_or_default();
        let client = reqwest::blocking::Client::new();
        let mut request = client.get(shard_url);

        if !token.is_empty() {
            request = request.bearer_auth(&token);
        }

        let response = request.send().ok()?;
        let mut archive = Archive::new(response);

        for entry in archive.entries().ok()? {
            let mut entry = entry.ok()?;

            if entry.path().ok()?.to_string_lossy() == filename {
                let mut buffer = Vec::new();
                entry.read_to_end(&mut buffer).ok()?;

                let img = ImageReader::new(Cursor::new(buffer))
                    .with_guessed_format()
                    .ok()?
                    .decode()
                    .ok()?;

                let resized = img.resize_exact(
                    224,
                    224,
                    image::imageops::FilterType::Triangle,
                );

                let rgb = resized.to_rgb8();

                let mut r_chan = Vec::with_capacity(224 * 224);
                let mut g_chan = Vec::with_capacity(224 * 224);
                let mut b_chan = Vec::with_capacity(224 * 224);

                for pixel in rgb.pixels() {
                    let r = pixel[0] as f32 / 255.0;
                    let g = pixel[1] as f32 / 255.0;
                    let b = pixel[2] as f32 / 255.0;

                    // Standard ImageNet mean/std normalization
                    let r = (r - 0.485) / 0.229;
                    let g = (g - 0.456) / 0.224;
                    let b = (b - 0.406) / 0.225;

                    r_chan.push(r);
                    g_chan.push(g);
                    b_chan.push(b);
                }

                let mut data = Vec::with_capacity(3 * 224 * 224);
                data.extend(r_chan);
                data.extend(g_chan);
                data.extend(b_chan);

                return Some(data);
            }
        }

        None
    }
}

impl Dataset<ImageItem> for PlainTarDataset {
    fn get(&self, index: usize) -> Option<ImageItem> {
        let entry = self.items.get(index)?;
        let image_data = Self::read_image_from_stream(&entry.shard_url, &entry.filename)?;

        Some(ImageItem {
            image_data,
            label: entry.label,
        })
    }

    fn len(&self) -> usize {
        self.items.len()
    }
}

#[derive(Clone)]
pub struct ImageBatcher<B: Backend> {
    pub device: B::Device,
}

impl<B: Backend> ImageBatcher<B> {
    pub fn new(device: B::Device) -> Self {
        Self { device }
    }
}

#[derive(Clone, Debug)]
pub struct ImageBatch<B: Backend> {
    pub images: Tensor<B, 4>,
    pub targets: Tensor<B, 1, burn::tensor::Int>,
}

impl<B: Backend> Batcher<ImageItem, ImageBatch<B>> for ImageBatcher<B> {
    fn batch(&self, items: Vec<ImageItem>) -> ImageBatch<B> {
        let batch_size = items.len();

        let mut images_vec = Vec::with_capacity(batch_size * 3 * 224 * 224);
        let mut labels_vec = Vec::with_capacity(batch_size);

        for item in items {
            images_vec.extend(item.image_data);
            labels_vec.push(item.label as i64);
        }

        let images = Tensor::<B, 1>::from_floats(images_vec.as_slice(), &self.device)
            .reshape([batch_size, 3, 224, 224]);

        let label_data = TensorData::new(labels_vec, vec![batch_size]);
        let targets = Tensor::<B, 1, burn::tensor::Int>::from_data(label_data, &self.device);

        ImageBatch { images, targets }
    }
}
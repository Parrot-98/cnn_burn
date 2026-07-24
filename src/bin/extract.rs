use std::fs::File;
use std::path::Path;
use tar::Archive;
use walkdir::WalkDir;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let tars_dir = "./path/to/tar_files"; // Folder containing your .tar files
    let output_dir = "./data/imagenet_extracted"; // Output folder

    println!("Extracting .tar files from {}...", tars_dir);

    for entry in WalkDir::new(tars_dir).into_iter().filter_map(|e| e.ok()) {
        let path = entry.path();

        if path.extension().and_then(|s| s.to_str()) == Some("tar") {
            println!("Extracting: {:?}", path.file_name().unwrap());

            // Open the file directly without any decompression decoder
            let file = File::open(path)?;
            let mut archive = Archive::new(file);

            archive.unpack(output_dir)?;
        }
    }

    println!("Extraction complete! Saved to {}", output_dir);
    Ok(())
}
use base64::{engine::general_purpose::STANDARD, Engine};
use std::io::Error;
use std::path::Path;
use tokio::fs;

pub async fn load_image(path: impl AsRef<Path>) -> Result<Vec<u8>, Error> {
    fs::read(path).await
}

pub fn base64_encode(data: &[u8]) -> String {
    STANDARD.encode(&data)
}

pub async fn load_image_as_base64(path: impl AsRef<Path>) -> Result<String, Error> {
    let data = load_image(path).await?;
    Ok(base64_encode(&data))
}

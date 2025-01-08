use std::error::Error;
use std::path::Path;

pub trait LlmClientChat {
    type Error: Error + Send + Sync + 'static;

    fn new(base_url: Option<&str>, api_key: Option<&str>) -> Self
    where
        Self: Sized;

    async fn send_message(
        &self,
        model: impl Into<String>,
        text: impl AsRef<str>,
        image_path: Option<impl AsRef<Path>>,
        temperature: Option<f32>,
    ) -> Result<String, Self::Error>;
}

pub trait LlmClientEmbedding {
    type Error: Error + Send + Sync + 'static;

    fn new(base_url: Option<&str>, api_key: Option<&str>) -> Self
    where
        Self: Sized;

    async fn embed(
        &self,
        model: impl Into<String>,
        text: impl AsRef<str>,
    ) -> Result<Vec<f32>, Self::Error>;
}

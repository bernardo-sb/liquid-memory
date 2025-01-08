use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json;

#[derive(Debug, Serialize, Deserialize)]
pub struct TextEmbeddingRequest {
    pub inputs: Vec<String>,
}

pub struct TextEmbeddingInference {
    pub client: Client,
    pub base_url: String,
}

impl TextEmbeddingInference {
    pub fn new(base_url: Option<&str>) -> Self {
        Self {
            client: Client::new(),
            base_url: base_url.unwrap_or("http://localhost:8888").to_string(),
        }
    }

    pub async fn embed(
        &self,
        text: Vec<String>,
    ) -> Result<Vec<Vec<f32>>, Box<dyn std::error::Error>> {
        let request = TextEmbeddingRequest { inputs: text };
        let response = self
            .client
            .post(format!("{}/embed", self.base_url))
            .json(&request)
            .send()
            .await?;

        //  Example response:
        // [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]

        let text = response.text().await?;
        let data: Vec<Vec<f32>> = serde_json::from_str(&text)?;
        Ok(data)
    }
}

// TODO: /rerank, /predict (classification)

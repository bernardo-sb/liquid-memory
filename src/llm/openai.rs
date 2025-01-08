use super::llm_client::{LlmClientChat, LlmClientEmbedding};
use crate::utils::load_image;
use base64::{engine::general_purpose::STANDARD, Engine};
use reqwest::{Client, StatusCode};
use serde::{Deserialize, Serialize};
use std::env;
use std::path::Path;
use thiserror::Error;

#[derive(Debug, Serialize, Deserialize)]
pub struct Message {
    role: String,
    content: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct EmbeddingRequest {
    model: String,
    prompt: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct EmbeddingResponse {
    embedding: Vec<f32>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct CompletionTokensDetails {
    reasoning_tokens: i32,
    accepted_prediction_tokens: i32,
    rejected_prediction_tokens: i32,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Usage {
    prompt_tokens: i32,
    completion_tokens: i32,
    total_tokens: i32,
    completion_tokens_details: Option<CompletionTokensDetails>, // Optional for Ollama compatibility
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Choice {
    message: Message,
    logprobs: Option<serde_json::Value>,
    finish_reason: String,
    index: i32,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct OpenAIResponse {
    id: String,
    object: String,
    created: i64,
    model: String,
    usage: Usage,
    choices: Vec<Choice>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ErrorResponse {
    error: MessageError,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct MessageError {
    message: String,
    #[serde(rename = "type")]
    error_type: String,
    code: Option<String>,
}

#[derive(Debug, Error)]
pub enum OpenAIError {
    #[error("API Error: {status}, {message}")]
    ApiError {
        status: reqwest::StatusCode,
        message: String,
    },
    #[error("Environment variable not found: {0}")]
    EnvError(#[from] env::VarError),
    #[error("IO Error: {0}")]
    IoError(#[from] std::io::Error),
    #[error("Request Error: {0}")]
    RequestError(#[from] reqwest::Error),
    #[error("Image Error: {0}")]
    ImageError(String),
}

pub struct OpenAIClient {
    client: Client,
    base_url: String,
    api_key: String,
}

impl OpenAIClient {
    fn get_or_load_key(key: Option<&str>) -> String {
        match key {
            Some(val) => val.to_string(),
            None => env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY must be set"),
        }
    }

    fn get_or_load_url(url: Option<&str>) -> String {
        match url {
            Some(val) => val.to_string(),
            None => env::var("OPENAI_BASE_URL").unwrap_or("https://api.openai.com".to_string()),
        }
    }

    fn create_payload(
        model: &str,
        text: &str,
        image_buffer: Option<Vec<u8>>,
        temperature: Option<f32>,
    ) -> serde_json::Value {
        let mut messages = Vec::new();
        let mut content = Vec::new();

        content.push(serde_json::json!({
            "type": "text",
            "text": text
        }));

        // Example format:
        // messages": [
        //   {
        //     "role": "user",
        //     "content": [
        //         {"type": "text", "text": "What's in this image?"},
        //         {
        //             "type": "image_url",
        //             "image_url": {"url": "data:image/jpeg;base64,{base64_image}"},
        //         }
        //     ]
        //   }
        // ]
        if let Some(buffer) = image_buffer {
            let image_base64 = STANDARD.encode(&buffer);
            let content_img = serde_json::json!({
                    "type": "image_url",
                    "image_url": {
                        "url": format!("data:image/jpeg;base64,{image_base64}")
                    }
                }
            );

            content.push(content_img);
        }

        messages.push(serde_json::json!({
            "role": "user",
            "content": content
        }));

        serde_json::json!({
            "model": model,
            "messages": messages,
            "temperature": temperature.unwrap_or(0.7),
            "max_tokens": 1024 // TODO!
        })
    }

    pub fn new(base_url: Option<&str>, api_key: Option<&str>) -> Self {
        OpenAIClient {
            client: Client::new(),
            base_url: Self::get_or_load_url(base_url),
            api_key: Self::get_or_load_key(api_key),
        }
    }

    async fn create_chat_completion(
        &self,
        model: &str,
        text: &str,
        image_path: Option<&str>,
        temperature: Option<f32>,
    ) -> Result<OpenAIResponse, Box<dyn std::error::Error>> {
        let image_buffer = match image_path {
            Some(path) => Some(load_image(path).await?),
            None => None,
        };

        let payload = Self::create_payload(model, text, image_buffer, temperature);
        let url = format!("{}/v1/chat/completions", self.base_url);

        let response = self
            .client
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&payload)
            .send()
            .await?;

        if !response.status().is_success() {
            let error_response = response.json::<ErrorResponse>().await?;
            return Err(Box::new(std::io::Error::new(
                std::io::ErrorKind::Other,
                error_response.error.message,
            )));
        }

        let result = response.json::<OpenAIResponse>().await?;
        Ok(result)
    }
}

impl LlmClientChat for OpenAIClient {
    type Error = OpenAIError;

    fn new(base_url: Option<&str>, api_key: Option<&str>) -> Self {
        Self {
            client: Client::new(),
            base_url: Self::get_or_load_url(base_url),
            api_key: Self::get_or_load_key(api_key),
        }
    }

    async fn send_message(
        &self,
        model: impl Into<String>,
        text: impl AsRef<str>,
        image_path: Option<impl AsRef<Path>>,
        temperature: Option<f32>,
    ) -> Result<String, OpenAIError> {
        let response = self
            .create_chat_completion(
                &model.into(),
                text.as_ref(),
                image_path.as_ref().map(|p| p.as_ref().to_str().unwrap()),
                temperature,
            )
            .await
            .unwrap();

        Ok(response.choices[0].message.content.clone())
    }
}

impl LlmClientEmbedding for OpenAIClient {
    type Error = OpenAIError;

    fn new(base_url: Option<&str>, api_key: Option<&str>) -> Self {
        Self {
            client: Client::new(),
            base_url: Self::get_or_load_url(base_url),
            api_key: Self::get_or_load_key(api_key),
        }
    }

    async fn embed(
        &self,
        model: impl Into<String>,
        text: impl AsRef<str>,
    ) -> Result<Vec<f32>, OpenAIError> {
        let payload = serde_json::json!({
            "model": model.into(),
            "prompt": text.as_ref()
        });
        let url = format!("{}/api/embeddings", self.base_url);

        let response = self
            .client
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&payload)
            .send()
            .await?;

        if !response.status().is_success() {
            let error_response = response.json::<ErrorResponse>().await?;
            // build status code from String
            let status_code = error_response
                .error
                .code
                .as_ref()
                .and_then(|code| code.parse().ok())
                .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
            return Err(OpenAIError::ApiError {
                status: status_code,
                message: error_response.error.message,
            });
        }

        let result: EmbeddingResponse = response.json().await?;
        let embeddings = result.embedding;
        Ok(embeddings)
    }
}

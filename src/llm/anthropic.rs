use super::llm_client::LlmClientChat;
use crate::utils::load_image;
use base64::{engine::general_purpose::STANDARD, Engine};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::{env, path::Path};
use thiserror::Error;

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct ContentItem {
    #[serde(default)]
    text: String,
    #[serde(rename = "type")]
    content_type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    source: Option<ImageSource>,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct ImageSource {
    #[serde(rename = "type")]
    source_type: String,
    media_type: String,
    data: String,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct Usage {
    cache_creation_input_tokens: i32,
    cache_read_input_tokens: i32,
    input_tokens: i32,
    output_tokens: i32,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct AnthropicResponse {
    id: String,
    content: Vec<ContentItem>,
    model: String,
    role: String,
    stop_reason: String,
    stop_sequence: Option<String>,
    #[serde(rename = "type")]
    message_type: String,
    usage: Usage,
}

#[derive(Debug, Error)]
pub enum AnthropicError {
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

#[derive(Debug, Serialize)]
struct Message {
    role: String,
    content: Vec<ContentItem>,
}

#[derive(Debug, Serialize)]
struct RequestPayload {
    model: String,
    max_tokens: u32,
    messages: Vec<Message>,
    temperature: Option<f32>,
}

pub struct AnthropicClient {
    client: Client,
    base_url: String,
    api_key: String,
    version: String,
}

impl AnthropicClient {
    fn get_or_load_key(key: Option<&str>) -> String {
        match key {
            Some(val) => val.to_string(),
            None => env::var("ANTHROPIC_API_KEY")
                .expect("ANTHROPIC_API_KEY environment variable not set"),
        }
    }

    fn get_or_load_url(url: Option<&str>) -> String {
        match url {
            Some(val) => val.to_string(),
            None => env::var("ANTHROPIC_BASE_URL")
                .unwrap_or_else(|_| "https://api.anthropic.com".to_string()),
        }
    }

    fn create_content(
        text: &str,
        image_data: Option<Vec<u8>>,
    ) -> Result<Vec<ContentItem>, AnthropicError> {
        let mut content = vec![ContentItem {
            text: text.to_string(),
            content_type: "text".to_string(),
            source: None,
        }];

        if let Some(image_buffer) = image_data {
            let image_base64 = STANDARD.encode(&image_buffer);
            content.push(ContentItem {
                text: String::new(),
                content_type: "image".to_string(),
                source: Some(ImageSource {
                    source_type: "base64".to_string(),
                    media_type: "image/png".to_string(),
                    data: image_base64,
                }),
            });
        }

        Ok(content)
    }

    pub fn new(base_url: Option<&str>, api_key: Option<&str>, version: Option<&str>) -> Self {
        Self {
            client: Client::new(),
            base_url: Self::get_or_load_url(base_url),
            api_key: Self::get_or_load_key(api_key),
            version: version.unwrap_or("2023-06-01").to_string(),
        }
    }

    pub async fn create_message(
        &self,
        model: impl Into<String>,
        max_tokens: u32,
        text: impl AsRef<str>,
        image_path: Option<impl AsRef<Path>>,
        temperature: Option<f32>,
    ) -> Result<AnthropicResponse, AnthropicError> {
        let image_data = match image_path {
            Some(path) => Some(load_image(path).await?),
            None => None,
        };

        let content = Self::create_content(text.as_ref(), image_data)?;
        let payload = RequestPayload {
            model: model.into(),
            max_tokens,
            messages: vec![Message {
                role: "user".to_string(),
                content,
            }],
            temperature,
        };

        let url = format!("{}/v1/messages", self.base_url);
        let response = self
            .client
            .post(&url)
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", &self.version)
            .header("content-type", "application/json")
            .json(&payload)
            .send()
            .await?;

        if !response.status().is_success() {
            let status = response.status();
            let message = response
                .text()
                .await
                .unwrap_or_else(|_| "Unable to fetch error details".to_string());
            return Err(AnthropicError::ApiError { status, message });
        }

        Ok(response.json().await?)
    }
}

impl LlmClientChat for AnthropicClient {
    type Error = AnthropicError;

    fn new(base_url: Option<&str>, api_key: Option<&str>) -> Self {
        Self {
            client: Client::new(),
            base_url: Self::get_or_load_url(base_url),
            api_key: Self::get_or_load_key(api_key),
            version: "2023-01-01".to_string(),
        }
    }

    async fn send_message(
        &self,
        model: impl Into<String>,
        text: impl AsRef<str>,
        image_path: Option<impl AsRef<Path>>,
        temperature: Option<f32>,
    ) -> Result<String, AnthropicError> {
        let response = self
            .create_message(
                model,
                4096, // default max tokens
                text,
                image_path,
                temperature,
            )
            .await?;

        // Combine all text content from the response
        Ok(response
            .content
            .into_iter()
            .filter(|item| item.content_type == "text")
            .map(|item| item.text)
            .collect::<Vec<_>>()
            .join(""))
    }
}

// #[cfg(test)]
// mod tests {
//     use super::*;
//     use mockito::mock;
//     use serde_json::json;

//     #[tokio::test]
//     async fn test_create_message_success() {
//         let mock_server = mock("POST", "/v1/messages")
//             .match_header("content-type", "application/json")
//             .with_status(200)
//             .with_body(
//                 json!({
//                     "id": "test_id",
//                     "content": [{"text": "Test response", "type": "text"}],
//                     "model": "claude-3",
//                     "role": "assistant",
//                     "stop_reason": "stop_sequence",
//                     "stop_sequence": null,
//                     "type": "message",
//                     "usage": {
//                         "cache_creation_input_tokens": 0,
//                         "cache_read_input_tokens": 0,
//                         "input_tokens": 10,
//                         "output_tokens": 5
//                     }
//                 })
//                 .to_string(),
//             )
//             .create();

//         let client = AnthropicClient::new(
//             Some(&mockito::server_url()),
//             Some("test_key"),
//             Some("2023-06-01"),
//         );
//         let response = client
//             .create_message("claude-3", 100, "Test message", None::<&str>)
//             .await
//             .unwrap();

//         assert_eq!(response.id, "test_id");
//         mock_server.assert();
//     }
// }

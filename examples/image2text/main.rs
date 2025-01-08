use liquid_memory::llm::llm_client::LlmClientChat;
use liquid_memory::llm::openai::OpenAIClient;
use liquid_memory::vectorstore::qdrant_client::QdrantClient;
use tokio;

#[tokio::main]
async fn main() {
    let openai_client = OpenAIClient::new(None, None); // Run with env vars
                                                       // let openai_client = OpenAIClient::new(Some("http://localhost:11434"), Some("sk-")); // Run with Ollama
    let response = openai_client
        .send_message(
            "gpt-4o-mini",
            // "llama3.2-vision", // Ollama client
            "What is in the image?",
            Some("images/boot.png"),
            Some(0.0),
        )
        .await
        .unwrap();

    println!("Response: {}", response);
}

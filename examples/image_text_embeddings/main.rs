use liquid_memory::embeddings::text_embedding_inference::TextEmbeddingInference;
use liquid_memory::llm::llm_client::{LlmClientChat, LlmClientEmbedding};
use liquid_memory::llm::openai::OpenAIClient;
use liquid_memory::utils::{base64_encode, load_image};
use liquid_memory::vectorstore::qdrant_client::QdrantClient;

use qdrant_client::qdrant::{Distance, VectorParamsBuilder};
use qdrant_client::Payload;
use tokio;

const IMAGE_DESCRIPTION: &str = "Response: The image shows a product page for a pair of ankle boots, with the title \"KHAITE Marfa 25mm suede ankle boots\" and a price tag of €799. 

* The main focus of the image is the boot itself, which is shown in a light brown color.
        + The boot has a low heel and an elasticated top for easy slip-on.
        + It appears to be made from high-quality suede material.
* Below the image of the boot, there are several lines of text that provide additional information about the product.
        + The first line reads \"KHAITE\", which is likely the brand name.
        + The second line reads \"Marfa 25mm suede ankle boots\", which describes the style and material of the boot.
        + The third line shows the price of the boot, which is €799.
* In the top-right corner of the image, there is a small heart icon, which may indicate that the product has been favorited by the user.

Overall, the image provides a clear view of the product and its features, as well as some basic information about the brand and price.";

#[tokio::main]
async fn main() {
    let image_path = "images/boot.png";
    // Image Embedding
    let image_embedding_client = TextEmbeddingInference::new(Some("http://localhost:8000"));
    let image = load_image(image_path).await.unwrap();
    let image_base64 = base64_encode(&image);

    println!("Computing image embedding...");
    let embeddings_img = image_embedding_client
        .embed(vec![image_base64])
        .await
        .unwrap();

    // Print truncated
    println!("Image Embeddings: {:#?}", embeddings_img);
    println!("Image Embeddings dim: {:#?}", embeddings_img[0].len());

    // Image to Text
    // uncommet to run with Ollama
    // let ollama_client = OpenAIClient::new(Some("http://localhost:11434"), Some("sk-")); // Run with env vars

    // println!("Computing image to text...");
    // let response = ollama_client
    //     .send_message(
    //         "llama3.2-vision",
    //         "What is in the image?",
    //         Some(image_path),
    //         Some(0.0),
    //     )
    //     .await
    //     .unwrap();

    // println!("Image2Text response: {}", response);

    //Text Embedding
    // let texts = vec![response];
    let texts = vec![IMAGE_DESCRIPTION.to_string()];
    println!("Computing text embedding...");
    let text_embedding_client = TextEmbeddingInference::new(Some("http://localhost:8888"));
    let embeddings_txt = text_embedding_client.embed(texts.clone()).await.unwrap();
    println!("Text Embeddings: {:#?}", embeddings_txt);
    println!("Text Embeddings dim: {:#?}", embeddings_txt[0].len());

    // Vectorstore
    println!("Creating vectorstore...");
    let vs_client = QdrantClient::new("http://localhost:6334");

    let collection_name = "test_collection";
    let collection_exists = vs_client.check_collection(collection_name).await.unwrap();
    let vec_size_img = embeddings_img[0].len() as u64;
    let vec_size_txt = embeddings_txt[0].len() as u64;
    if !collection_exists {
        println!("Creating collection...");
        vs_client
            .create_multivector_collection(collection_name, vec_size_img, vec_size_txt)
            .await
            .unwrap();
    }

    // Upsert Points
    println!("Upserting points...");
    let text = texts[0].clone();
    vs_client
        .upsert_points_multivector(
            collection_name,
            embeddings_img[0].clone(),
            embeddings_txt[0].clone(),
            Payload::try_from(serde_json::json!({"text": text, "image_path": image_path})).unwrap(),
        )
        .await
        .unwrap();

    // Search
    println!("Searching...");
    let (img_response, txt_response) = vs_client
        .query_points_multivector(
            collection_name,
            embeddings_img[0].clone(),
            embeddings_txt[0].clone(),
            10,
        )
        .await
        .unwrap();
    println!("Image Response: {:#?}", img_response);
    println!("Text Response: {:#?}", txt_response);

    // Clean up
    vs_client.delete_collection(collection_name).await.unwrap();
}

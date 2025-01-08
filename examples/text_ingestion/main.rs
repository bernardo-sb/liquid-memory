use liquid_memory::embeddings::text_embedding_inference::TextEmbeddingInference;
use liquid_memory::vectorstore::qdrant_client::{texts_to_payload, QdrantClient};
use tokio;

use qdrant_client::qdrant::{Distance, VectorParamsBuilder};

#[tokio::main]
async fn main() {
    let client = QdrantClient::new("http://localhost:6334");

    let collection_name = "test_collection";
    let collection_exists = client.check_collection(collection_name).await.unwrap();
    if !collection_exists {
        client
            .create_collection(
                collection_name,
                VectorParamsBuilder::new(1024, Distance::Cosine),
            )
            .await
            .unwrap();
    }

    let sentences = vec![
        "What is Deep Learning?".to_string(),
        "What is Machine Learning?".to_string(),
        "What is Artificial Intelligence?".to_string(),
        "What is Natural Language Processing?".to_string(),
        "What is Computer Vision?".to_string(),
        "What is Reinforcement Learning?".to_string(),
        "What is Generative AI?".to_string(),
    ];
    let payload = texts_to_payload(sentences.clone(), "text").unwrap();

    let tei_client = TextEmbeddingInference::new(Some("http://localhost:8888"));
    let embeddings = tei_client.embed(sentences.clone()).await.unwrap();

    client
        .upsert_points(collection_name, embeddings.clone(), payload)
        .await
        .unwrap();

    let response = client
        .query_points(collection_name, embeddings[0].clone(), 5)
        .await
        .unwrap();

    println!("Response: {:#?}", response);

    // Get data and scores
    let data: Vec<(String, f32)> = response
        .result
        .iter()
        .map(|r| (r.payload.get("text").unwrap().to_string(), r.score))
        .collect();

    println!("Query: {}", sentences[0].clone());
    println!("Data: {:#?}", data);

    // Clean up
    client.delete_collection(collection_name).await.unwrap();
}

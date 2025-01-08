use crate::embeddings::text_embedding_inference::TextEmbeddingInference;
use crate::llm::llm_client::LlmClientChat;
use crate::utils::load_image_as_base64;
use crate::vectorstore::qdrant_client::QdrantClient;
use anyhow::Result;
use chrono;
use qdrant_client::Payload;
use serde_json::json;

pub async fn ingest_images(
    collection_name: &str,
    image_paths: Vec<String>,
    embedding_url: &str,
    client: &QdrantClient,
) -> Result<()> {
    let mut embeddings = Vec::new();
    let mut payloads = Vec::new();

    for image_path in image_paths {
        let base64_image = vec![load_image_as_base64(&image_path).await?];

        let image_embedding_client = TextEmbeddingInference::new(Some(embedding_url));
        let response = image_embedding_client
            .embed(base64_image.clone())
            .await
            .unwrap();

        embeddings.push(response[0].clone());

        let payload = Payload::try_from(json!({
            "image_path": image_path,
            "timestamp": chrono::Utc::now().to_rfc3339()
        }))?;
        payloads.push(payload);
    }

    // Upsert points to vector store
    client
        .upsert_points(collection_name, embeddings, payloads)
        .await?;
    Ok(())
}

pub async fn ingest_texts(
    collection_name: &str,
    texts: Vec<String>,
    text_embedding_client: &TextEmbeddingInference,
    client: &QdrantClient,
) -> Result<()> {
    let embeddings = text_embedding_client.embed(texts.clone()).await.unwrap();

    let payloads: Vec<Payload> = texts
        .into_iter()
        .map(|text| {
            Payload::try_from(json!({
                "text": text,
                "timestamp": chrono::Utc::now().to_rfc3339()
            }))
            .unwrap()
        })
        .collect();

    client
        .upsert_points(collection_name, embeddings, payloads)
        .await?;
    Ok(())
}

pub async fn ingest_multivector(
    collection_name: &str,
    image_paths: Vec<String>,
    texts: Vec<String>,
    image_embedding_url: &str,
    text_embedding_url: &str,
    client: &QdrantClient,
) -> Result<()> {
    let text_embedding_client = TextEmbeddingInference::new(Some(text_embedding_url));
    let image_embedding_client = TextEmbeddingInference::new(Some(image_embedding_url));

    let mut images = Vec::new();

    let text_embeddings = text_embedding_client.embed(texts.clone()).await.unwrap();

    for image_path in image_paths {
        let base64_image = load_image_as_base64(&image_path).await?;
        images.push(base64_image.clone());
    }

    let image_embedding = image_embedding_client.embed(images.clone()).await.unwrap();

    for (idx, (image_path, text)) in images.iter().zip(texts.iter()).enumerate() {
        let payload = Payload::try_from(json!({
            "image_path": image_path,
            "text": text,
            "timestamp": chrono::Utc::now().to_rfc3339()
        }))?;

        client
            .upsert_points_multivector(
                collection_name,
                image_embedding[idx].clone(),
                text_embeddings[idx].clone(),
                payload,
            )
            .await?;
    }

    Ok(())
}

pub async fn ingest_image_to_text(
    collection_name: &str,
    model: &str,
    temperature: Option<f32>,
    image_paths: Vec<String>,
    prompt: String,
    llm_client: impl LlmClientChat,
    image_embedding_url: &str,
    text_embedding_url: &str,
    client: &QdrantClient,
) -> Result<()> {
    let mut texts = Vec::new();

    // Generate text descriptions for each image using LLM
    for image_path in image_paths.clone() {
        let text_description = llm_client
            .send_message(model, &prompt, Some(image_path), temperature)
            .await?;
        texts.push(text_description);
    }

    ingest_multivector(
        collection_name,
        image_paths,
        texts,
        image_embedding_url,
        text_embedding_url,
        client,
    )
    .await?;

    Ok(())
}

//TODO: Ingest with Image2Text

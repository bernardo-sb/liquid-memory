use qdrant_client::qdrant::{
    CreateCollectionBuilder, Distance, Filter, ListCollectionsResponse, PointStruct,
    PointsOperationResponse, QueryPointsBuilder, QueryResponse, ScalarQuantizationBuilder,
    SearchBatchPointsBuilder, SearchBatchResponse, SearchParamsBuilder, SearchPointsBuilder,
    SearchResponse, UpsertPointsBuilder, VectorParamsBuilder, Vectors, VectorsConfigBuilder,
};
use qdrant_client::{Payload, Qdrant, QdrantError};
use std::collections::HashMap;
use uuid::Uuid;

pub fn texts_to_payload(texts: Vec<String>, field_name: &str) -> Result<Vec<Payload>, QdrantError> {
    texts
        .iter()
        .map(|s| Payload::try_from(serde_json::json!({field_name: s})))
        .collect::<Result<Vec<Payload>, _>>()
}

fn to_point_struct(point: Vec<f32>, payload: Payload) -> PointStruct {
    PointStruct::new(Uuid::new_v4().to_string(), point, payload)
}

pub struct QdrantClient {
    client: Qdrant,
}

impl QdrantClient {
    pub fn new(url: &str) -> Self {
        let client = Qdrant::from_url(url).build().unwrap();
        Self { client }
    }

    pub async fn get_collections(&self) -> Result<ListCollectionsResponse, QdrantError> {
        let collections = self.client.list_collections().await?;
        Ok(collections)
    }

    pub async fn create_collection(
        &self,
        collection_name: impl Into<String>,
        vector_params: VectorParamsBuilder,
    ) -> Result<(), QdrantError> {
        self.client
            .create_collection(
                CreateCollectionBuilder::new(collection_name)
                    .vectors_config(vector_params)
                    .quantization_config(ScalarQuantizationBuilder::default()),
            )
            .await?;
        Ok(())
    }

    pub async fn create_multivector_collection(
        &self,
        collection_name: impl Into<String>,
        vector_size_img: u64,
        vector_size_txt: u64,
    ) -> Result<(), QdrantError> {
        let mut vectors_config = VectorsConfigBuilder::default();
        vectors_config.add_named_vector_params(
            "image",
            VectorParamsBuilder::new(vector_size_img, Distance::Cosine).build(),
        );
        vectors_config.add_named_vector_params(
            "text",
            VectorParamsBuilder::new(vector_size_txt, Distance::Cosine).build(),
        );

        self.client
            .create_collection(
                CreateCollectionBuilder::new(collection_name).vectors_config(vectors_config),
            )
            .await?;
        Ok(())
    }

    pub async fn delete_collection(
        &self,
        collection_name: impl Into<String>,
    ) -> Result<(), QdrantError> {
        self.client.delete_collection(collection_name).await?;
        Ok(())
    }

    pub async fn delete_all_collections(&self) -> Result<(), QdrantError> {
        let collections = self.get_collections().await.unwrap();
        for collection in collections.collections.iter() {
            self.delete_collection(collection.name.clone())
                .await
                .unwrap();
        }
        Ok(())
    }

    pub async fn check_collection(
        &self,
        collection_name: impl Into<String>,
    ) -> Result<bool, QdrantError> {
        let collection_exists = self.client.collection_exists(collection_name).await?;
        Ok(collection_exists)
    }

    pub async fn upsert_points(
        &self,
        collection_name: &str,
        embeddings: Vec<Vec<f32>>,
        payload: Vec<Payload>,
    ) -> Result<PointsOperationResponse, QdrantError> {
        let points: Vec<PointStruct> = embeddings
            .iter()
            .zip(payload.iter())
            .map(|(embedding, payload)| to_point_struct(embedding.clone(), payload.clone()))
            .collect();
        self.client
            .upsert_points(UpsertPointsBuilder::new(collection_name, points).build())
            .await
    }

    pub async fn upsert_points_multivector(
        &self,
        collection_name: &str,
        vec_img: Vec<f32>,
        vec_txt: Vec<f32>,
        payload: Payload,
    ) -> Result<PointsOperationResponse, QdrantError> {
        let response = self
            .client
            .upsert_points(
                UpsertPointsBuilder::new(
                    collection_name,
                    vec![PointStruct::new(
                        Uuid::new_v4().to_string(),
                        HashMap::from([
                            ("image".to_string(), vec_img),
                            ("text".to_string(), vec_txt),
                        ]),
                        payload,
                    )],
                )
                .wait(true),
            )
            .await?;

        Ok(response)
    }

    pub async fn query_points(
        &self,
        collection_name: impl Into<String>,
        vector: Vec<f32>,
        limit: u64,
    ) -> Result<QueryResponse, QdrantError> {
        let response = self
            .client
            .query(
                QueryPointsBuilder::new(collection_name)
                    .query(vector)
                    .limit(limit)
                    .with_payload(true)
                    .with_vectors(true),
            )
            .await?;

        Ok(response)
    }

    pub async fn query_points_named(
        &self,
        collection_name: impl Into<String>,
        vector: Vec<f32>,
        limit: u64,
        vector_name: impl Into<String>,
    ) -> Result<QueryResponse, QdrantError> {
        let response = self
            .client
            .query(
                QueryPointsBuilder::new(collection_name)
                    .query(vector)
                    .limit(limit)
                    .using(vector_name),
            )
            .await?;
        Ok(response)
    }

    pub async fn query_points_multivector(
        &self,
        collection_name: impl Into<String>,
        image_vector: Vec<f32>,
        text_vector: Vec<f32>,
        limit: u64,
    ) -> Result<(QueryResponse, QueryResponse), QdrantError> {
        let (tx_img, rx_img) = tokio::sync::oneshot::channel();
        let (tx_txt, rx_txt) = tokio::sync::oneshot::channel();

        let collection_name = collection_name.into();

        let uri = &self.client.config.uri;

        let client_img = Qdrant::from_url(uri).build().unwrap();
        let client_txt = Qdrant::from_url(uri).build().unwrap();

        let collection_name_cln = collection_name.clone();
        tokio::spawn(async move {
            let img_response = client_img
                .query(
                    QueryPointsBuilder::new(collection_name)
                        .query(image_vector)
                        .limit(limit)
                        .with_payload(true)
                        .with_vectors(true)
                        .using("image"),
                )
                .await
                .unwrap();
            let _ = tx_img.send(img_response);
        });
        tokio::spawn(async move {
            let txt_response = client_txt
                .query(
                    QueryPointsBuilder::new(collection_name_cln)
                        .query(text_vector)
                        .limit(limit)
                        .with_payload(true)
                        .with_vectors(true)
                        .using("text"),
                )
                .await
                .unwrap();
            let _ = tx_txt.send(txt_response);
        });

        let image_response = rx_img.await.unwrap();
        let text_response = rx_txt.await.unwrap();
        Ok((image_response, text_response))
    }

    pub async fn search_points(
        &self,
        collection_name: impl Into<String>,
        vector: impl Into<Vec<f32>>,
        limit: u64,
        filter: Option<Filter>,
    ) -> Result<SearchResponse, QdrantError> {
        let search_result = self
            .client
            .search_points(
                SearchPointsBuilder::new(collection_name, vector, limit)
                    .filter(filter.unwrap_or(Filter::default()))
                    .with_payload(false)
                    .params(SearchParamsBuilder::default().exact(true)),
            )
            .await?;
        Ok(search_result)
    }
    pub async fn search_batch_points(
        &self,
        collection_name: impl Into<String>,
        vectors: Vec<Vec<f32>>,
        limit: u64,
        filter: Option<Filter>,
        //todo: payload
    ) -> Result<SearchBatchResponse, QdrantError> {
        let mut searches = vec![];
        let collection_name = collection_name.into();
        for vector in vectors {
            let search = SearchPointsBuilder::new(collection_name.clone(), vector, limit)
                .filter(filter.clone().unwrap_or(Filter::default()))
                .build();
            searches.push(search);
        }
        let results = self
            .client
            .search_batch_points(SearchBatchPointsBuilder::new(collection_name, searches))
            .await?;
        Ok(results)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    async fn setup() -> String {
        let client = QdrantClient::new("http://localhost:6334");
        let uuid = Uuid::new_v4().to_string();
        let collection_name = format!("test_collection-{uuid}");
        client
            .create_collection(
                &collection_name,
                VectorParamsBuilder::new(5, Distance::Cosine),
            )
            .await
            .unwrap();

        collection_name
    }

    async fn setup_multivector() -> String {
        let client = QdrantClient::new("http://localhost:6334");
        let uuid = Uuid::new_v4().to_string();
        let collection_name = format!("test_collection-{uuid}");
        client
            .create_multivector_collection(&collection_name, 5, 5)
            .await
            .unwrap();

        collection_name
    }

    async fn insert_points(collection_name: &str) {
        let client = QdrantClient::new("http://localhost:6334");

        let payload = Payload::try_from(serde_json::json!({"text": "Hello World"})).unwrap();
        client
            .upsert_points(
                collection_name,
                vec![vec![0.1, 0.2, 0.3, 0.4, 0.5]],
                vec![payload],
            )
            .await
            .unwrap();
    }

    async fn insert_multivector_points(collection_name: &str) {
        let client = QdrantClient::new("http://localhost:6334");
        client
            .upsert_points_multivector(
                collection_name,
                vec![0.1, 0.2, 0.3, 0.4, 0.5],
                vec![-0.1, -0.2, -0.3, -0.4, -0.5],
                Payload::try_from(
                    serde_json::json!({"text": "Hello World", "image_path": "path/to/image.jpg"}),
                )
                .unwrap(),
            )
            .await
            .unwrap();
    }

    async fn clean_up() {
        let client = QdrantClient::new("http://localhost:6334");
        client.delete_all_collections().await.unwrap();
    }

    #[test]
    fn test_to_point_struct() {
        let point = vec![0.1, 0.2, 0.3, 0.4, 0.5];
        let vectors = Some(Vectors::from(point.clone()));
        let payload = Payload::try_from(serde_json::json!({"text": "Hello World"})).unwrap();
        let point_struct = to_point_struct(point, payload);
        assert_eq!(point_struct.vectors, vectors);
    }

    #[test]
    fn test_texts_to_payload() {
        let texts = vec!["Hello World".to_string(), "Ola Mundo".to_string()];
        let payload = texts_to_payload(texts, "text").unwrap();
        assert_eq!(payload.len(), 2);
    }

    #[tokio::test]
    async fn test_qdrant_client_create_collection() {
        // Run the test
        let client = QdrantClient::new("http://localhost:6334");
        client
            .create_collection(
                "test_collection",
                VectorParamsBuilder::new(5, Distance::Cosine),
            )
            .await
            .unwrap();

        // Clean up
        clean_up().await;
    }

    #[tokio::test]
    async fn test_qdrant_client() {
        // Setup
        let collection_name = setup().await;

        // Run the test
        let client = QdrantClient::new("http://localhost:6334");
        let collections = client.get_collections().await.unwrap();
        assert!(collections
            .collections
            .iter()
            .any(|c| c.name == collection_name));
        println!("{:?}", collections);

        // Clean up
        clean_up().await;
    }
    #[tokio::test]
    async fn test_qdrant_client_check_collection() {
        let collection_name = setup().await;

        // Run the test
        let client = QdrantClient::new("http://localhost:6334");
        let collection_exists = client.check_collection(&collection_name).await.unwrap();
        assert!(collection_exists);

        // Clean up
        clean_up().await;
    }

    #[tokio::test]
    async fn test_qdrant_client_delete_collection() {
        let collection_name = setup().await;

        // Run the test
        let client = QdrantClient::new("http://localhost:6334");
        client.delete_collection(&collection_name).await.unwrap();

        // Clean up
        clean_up().await;
    }

    #[tokio::test]
    async fn test_qdrant_client_delete_all_collections() {
        // Setup
        let _ = setup().await;
        let _ = setup().await;

        // Run the test
        let client = QdrantClient::new("http://localhost:6334");
        client.delete_all_collections().await.unwrap();

        // Assert
        let collections = client.get_collections().await.unwrap();
        assert!(collections.collections.is_empty());
    }

    #[tokio::test]
    async fn test_qdrant_client_upsert_points() {
        let collection_name = setup().await;

        let payload = Payload::try_from(serde_json::json!({"text": "Hello World"})).unwrap();
        // Run the test
        let client = QdrantClient::new("http://localhost:6334");
        client
            .upsert_points(
                &collection_name,
                vec![vec![0.1, 0.2, 0.3, 0.4, 0.5]],
                vec![payload],
            )
            .await
            .unwrap();

        // Clean up
        clean_up().await;
    }

    #[tokio::test]
    async fn test_qdrant_client_query_points() {
        let collection_name = setup().await;
        insert_points(&collection_name).await;

        // Run the test
        let client = QdrantClient::new("http://localhost:6334");
        client
            .query_points(&collection_name, vec![0.1, 0.2, 0.3, 0.4, 0.5], 10)
            .await
            .unwrap();

        // Clean up
        clean_up().await;
    }

    #[tokio::test]
    async fn test_qdrant_client_search_points() {
        // Setup
        let collection_name = setup().await;
        insert_points(&collection_name).await;

        // Run the test
        let client = QdrantClient::new("http://localhost:6334");
        client
            .search_points(&collection_name, vec![0.1, 0.2, 0.3, 0.4, 0.5], 10, None)
            .await
            .unwrap();

        // Clean up
        clean_up().await;
    }

    #[tokio::test]
    async fn test_qdrant_client_search_batch_points() {
        // Setup
        let collection_name = setup().await;
        insert_points(&collection_name).await;

        // Run the test
        let client = QdrantClient::new("http://localhost:6334");
        client
            .search_batch_points(&collection_name, vec![], 10, None)
            .await
            .unwrap();

        // Clean up
        clean_up().await;
    }

    #[tokio::test]
    async fn test_qdrant_client_create_multivector_collection() {
        // Run the test
        let client = QdrantClient::new("http://localhost:6334");
        client
            .create_multivector_collection("test_collection_mv", 10, 10)
            .await
            .unwrap();

        // Clean up
        clean_up().await;
    }

    #[tokio::test]
    async fn test_qdrant_client_upsert_points_multivector() {
        // Setup
        let collection_name = setup_multivector().await;

        // Run the test
        let client = QdrantClient::new("http://localhost:6334");
        client
            .upsert_points_multivector(
                &collection_name,
                vec![0.1, 0.2, 0.3, 0.4, 0.5],
                vec![-0.1, -0.2, -0.3, -0.4, -0.5],
                Payload::try_from(
                    serde_json::json!({"text": "Hello World", "image_path": "path/to/image.jpg"}),
                )
                .unwrap(),
            )
            .await
            .unwrap();

        // Clean up
        clean_up().await;
    }

    #[tokio::test]
    async fn test_qdrant_client_query_points_multivector() {
        // Setup
        let collection_name = setup_multivector().await;
        insert_multivector_points(&collection_name).await;

        // Run the test
        let client = QdrantClient::new("http://localhost:6334");
        client
            .query_points_multivector(
                &collection_name,
                vec![0.1, 0.2, 0.3, 0.4, 0.5],
                vec![-0.1, -0.2, -0.3, -0.4, -0.5],
                10,
            )
            .await
            .unwrap();

        // Clean up
        clean_up().await;
    }

    #[tokio::test]
    async fn test_qdrant_client_query_points_named() {
        // Setup
        let collection_name = setup_multivector().await;
        insert_multivector_points(&collection_name).await;

        // Run the test
        let client = QdrantClient::new("http://localhost:6334");
        client
            .query_points_named(&collection_name, vec![0.1, 0.2, 0.3, 0.4, 0.5], 10, "text")
            .await
            .unwrap();

        // Clean up
        clean_up().await;
    }
}

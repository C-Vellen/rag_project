## Projet de RAG (Retrieval Augmented Generation)


1. ### Préparation des données:
Déposer les fichiers (.pdf, .txt, .md) et les dossiers dans **rag_project/src/documents**

Variables dans config.py:
```python
    # Chunking
    chunk_size: int = 1000
    chunk_overlap: int = 200

    # Embedding
    embedding_model: str = "text-embedding-3-small"
```


2. ### Lancement:

```bash
# réinitialiser / lancer la BD vectorielle:
docker compose down -v
docker compose up -d

# lancer l'ingestion des données:
poetry run python -m src.main src/documents
```

3. ### Voir la BD vectorielle :
```bash
# se connecter à la BD:
docker compose exec postgres psql -U rag -d ragdb
```
```sql
-- table des collections:
select * from langchain_pg_collection

-- table des embeddings(tronqué pour la lisibilité):
select 
    substring(id::text, 1, 5) || '...' as id,
    substring(collection_id::text, 1, 4) || '...' as col_id,
    left(embedding::text, 40) || '...]' as "embedding dimension = 1536",
    left(document::text, 80) || '...' || right(document::text, 20) as "document chunk ~ 1000 caractères / 180 mots",
    left(cmetadata::text, 1) || '...' || right(cmetadata::text, 6) as cmetadata
    from langchain_pg_embedding;

-- dimension des vecteurs :
select 
    collection_id, vector_dims(embedding) as dimensions, 
    count(*) as nb_chunks
    from langchain_pg_embedding
group by collection_id, vector_dims(embedding);

-- taille min/max/moyenne des chunks en caratères et mots
select 
    min(length(document))  as chars_min,
    max(length(document))  as chars_max,
    avg(length(document))::int  as chars_avg,
    -- approximation mots : diviser par 5 (fr/en)
    avg(length(document) / 5)::int  as mots_avg
    from langchain_pg_embedding;

-- table des embeddings(tronqué pour la lisibilité):
select 
    substring(id::text, 1, 5) || '...' as id,
    left(embedding::text, 40) || '...]' as "embedding dimension = 512",
    left(document::text, 120) as "document chunk ~ 100 caractères / 20 mots",
    left(cmetadata::text, 1) || '...' || right(cmetadata::text, 6) as cmetadata
    from langchain_pg_embedding;



```



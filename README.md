## Projet de RAG (Retrieval Augmented Generation)


1. ### Préparation des données:
Déposer les fichiers (.pdf, .txt, .md) et les dossiers dans **rag_project/src/documents**
Les fichiers déjà ingérés peuvent être déposés dans **rag_project/src/archives_documents** pour ne pas être ingérés 2 fois.

Embeddings model: 
- option 1 : embedding sur serveur openAI (nécessite une clé API) avec text-embedding-3-small
- option 2 : embedding local avec BAAI/bge-m3 (hugging face).
Choix au niveau de config.py

Variables dans config.py:
```python
    # Chunking
    chunk_length: int = 1000 # max 1000 caractères / chunk
    chunk_overlap: int = 200 # recouvrement de 200 caractères entre 2 chunks
    k: int = 4 # nombre de chunks sélectionnés pour générer la réponses
```


2. ### Lancement:
```bash
# réinitialiser / lancer la BD vectorielle:
docker compose down -v
docker compose up -d

# lancer l'ingestion des données:
poetry run python -m src.main ingest src/documents

# poser une question (prompt)
poetry run python -m src.main query "Bonjour, comment allez-vous ?"
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
    left(embedding::text, 40) || '...]' as "embedding",
    left(document::text, 80) || '...' || right(document::text, 20) as "document chunk ~ 1000 caractères / 180 mots",
    left(cmetadata::text, 1) || '...' || right(cmetadata::text, 6) as cmetadata
    from langchain_pg_embedding;q

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
    left(embedding::text, 40) || '...]' as "embeddings",
    left(document::text, 75) || '...' || right(document::text, 15) as "document chunk",
    right(left(cmetadata::text, 42)::text,16) || '...' || right(cmetadata::text, 6) as cmetadata
    from langchain_pg_embedding;



```



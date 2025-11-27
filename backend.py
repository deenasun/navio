import os
from supabase import AsyncClient, acreate_client
import asyncio
import dotenv
import json
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
# from sentence_transformers import SentenceTransformer

from typing import Optional, List
from pydantic import BaseModel


class Match(BaseModel):
    id: int
    description: str
    workflow: dict
    similarity: float


class SupabaseResponse(BaseModel):
    success: bool
    message: str
    matches: Optional[List[Match]] = None
    error: Optional[str] = None
    inserted_id: Optional[int] = None


class SupabaseInsert(BaseModel):
    embedding: list[float]
    description: str
    workflow: dict


# class SentenceTransformerEmbedder:
#     def __init__(self):
#         self.model = SentenceTransformer("all-miniLM-L12-v2")

#     def embed_text(self, text: str):
#         return self.model.encode(text)

#     def similarity(self, text1: str, text2: str):
#         return self.model.similarity(text1, text2)


class Embedder:
    def __init__(self):
        self.model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L12-v2")
        self.tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L12-v2")

    def mean_pooling(self, model_output, attention_mask):
        """
        Args:
            model_output: The output of the model.
            attention_mask: The attention mask.

        Returns:
            Average of all the model's output embeddings for an entire sequence.
        """
        token_embeddings = model_output[0]  # Extract last hidden state outputted by model for all tokens
        attention_mask = attention_mask.unsqueeze(-1)  # (batch_size, seq_len) -> (batch_size, seq_len, 1)
        attention_mask = attention_mask.expand(token_embeddings.shape).float()  # (batch_size, seq_len, 1) -> (batch_size, seq_len, 384)
        return torch.sum(token_embeddings * attention_mask, 1) / torch.clamp(attention_mask.sum(1), min=1e-9)  # (batch_size, 384)

    def embed_text(self, text: str, batch=False, return_numpy=True):
        """
        Embeds text into a (384, ) vector.
        First, the model's tokenizer tokenizes the text and returns input_ids and attention_mask.
        Then, the model's forward pass is performed on the input_ids and attention_mask.
        Mean pooling is performed on the model's output to return a (384, ) vector.
        Finally, the embeddings are normalized.

        Args:
            text: The text to embed.
            batch: Whether to embed a batch of texts. If batch = True, the return value is a torch tensor of shape (batch_size, 384).
            return_numpy: Whether to return a numpy array. If return_numpy = True, the return value is a numpy array of shape (384, ). If return_numpy = False, the return value is a torch tensor of shape (384, ).
        Returns:
            A tensor or numpy array of shape (batch_size, 384) or (384, ) representing the embedding of the text.
        """
        tokenizer_output = self.tokenizer(text, padding=True, truncation=True, return_tensors="pt")
        input_ids = tokenizer_output["input_ids"]
        attention_mask = tokenizer_output["attention_mask"]
        with torch.no_grad():
            model_output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        mean_pooled_embeddings = self.mean_pooling(model_output, attention_mask)
        normalized_embeddings = F.normalize(mean_pooled_embeddings, p=2, dim=1)  # Normalize the embeddings to unit length

        if batch:
            if return_numpy:
                return normalized_embeddings.detach().cpu().numpy()
            else:
                return normalized_embeddings  # (batch_size, 384)
        else:
            if return_numpy:
                return normalized_embeddings.squeeze(0).detach().cpu().numpy()
            else:
                return normalized_embeddings.squeeze(0)  # (1, 384) -> (384, )


class SupabaseClient:
    def __init__(self, client: AsyncClient):
        self.client = client
        self.embedder = Embedder()

    @classmethod
    async def create(cls) -> "SupabaseClient":
        """Async factory to build the Supabase wrapper."""
        dotenv.load_dotenv()
        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_KEY")
        client = await acreate_client(url, key)
        return cls(client)

    async def query(self, query: str) -> SupabaseResponse:
        """
        Queries the vector database for the most similar vectors to the query.
        """
        try:
            embedding = self.embedder.embed_text(query)
            embedding = embedding.tolist()

            result = await self.client.rpc(
                "match_vectors",
                {
                    "query_embedding": embedding,
                    "match_threshold": 0.5,
                    "match_count": 10,
                },
            ).execute()

            if result.data:
                matches = []
                for match in result.data:
                    matches.append(
                        Match(
                            id=match["id"],
                            description=match["description"],
                            workflow=json.loads(match["workflow"]),
                            similarity=match["similarity"],
                        )
                    )

                return SupabaseResponse(
                    success=True,
                    message=f"Found {result.count} matches for query {query}",
                    matches=matches,
                )

        except Exception as e:
            return SupabaseResponse(
                success=False,
                message="Failed to query data from Supabase",
                error=str(e),
            )

    async def insert(self, document: SupabaseInsert) -> SupabaseResponse:
        """
        Inserts a new document into the vector database.
        The new document should have the following fields:
        - embedding: list[float]
        - description: str
        - workflow: dict

        Args:
            document: The document to insert.
        Returns:
            A SupabaseResponse object.
        """
        try:
            result = await self.client.table("vector_db").upsert(document.model_dump()).execute()
            if result:
                return SupabaseResponse(
                    success=True,
                    message="Data inserted successfully",
                    inserted_id=result.data[0]["id"],
                )
            else:
                return SupabaseResponse(
                    success=False,
                    message="Failed to insert data into Supabase",
                )
        except Exception as e:
            return SupabaseResponse(
                success=False,
                message="Error inserting data into Supabase",
                error=str(e),
            )


async def main():
    supabase = await SupabaseClient.create()

    supabase_data = await supabase.query("What is the weather in Japan?")

    for match in supabase_data.matches:
        print("match id", match.id)
        print("match description", match.description)
        print("match workflow", match.workflow)
        print("match similarity", match.similarity)


if __name__ == "__main__":
    asyncio.run(main())

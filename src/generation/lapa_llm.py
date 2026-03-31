"""
Local Lapa LLM wrapper for RAG answer generation.

Model: lapa-llm/lapa-v0.1.2-instruct (Gemma3-based, 12B parameters)
Designed for GPU deployment (NVIDIA RTX 4090, 24GB VRAM) via Vast.ai.
"""

import torch
from loguru import logger
from typing import List, Dict, Optional


class LapaLLM:
    """
    Wrapper for local Lapa LLM v0.1.2 (Ukrainian-optimized Gemma3 12B).

    Uses HuggingFace transformers with bfloat16 precision for reduced
    VRAM usage. Supports both pipeline and manual generation modes.
    """

    MODEL_ID = "lapa-llm/lapa-v0.1.2-instruct"

    def __init__(
        self,
        model_id: str = MODEL_ID,
        device: str = "cuda",
        max_new_tokens: int = 512,
    ):
        self.model_id = model_id
        self.device = device
        self.max_new_tokens = max_new_tokens
        self._model = None
        self._processor = None

        logger.info(f"LapaLLM initialized (model={model_id}, device={device})")

    def _load_model(self):
        """Lazy-load model and processor on first use."""
        if self._model is not None:
            return

        from transformers import AutoProcessor, Gemma3ForConditionalGeneration

        logger.info(f"Loading model {self.model_id}...")
        self._model = Gemma3ForConditionalGeneration.from_pretrained(
            self.model_id,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        ).eval()
        self._processor = AutoProcessor.from_pretrained(self.model_id)
        logger.info("Model loaded successfully.")

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """
        Generate a response given system and user prompts.

        Args:
            system_prompt: Instructions for the model (role, rules).
            user_prompt: The user query with context.

        Returns:
            Generated text response.
        """
        self._load_model()

        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_prompt}],
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": user_prompt}],
            },
        ]

        inputs = self._processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self._model.device, dtype=torch.bfloat16)

        input_len = inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            generation = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )
            generation = generation[0][input_len:]

        decoded = self._processor.decode(generation, skip_special_tokens=True)
        return decoded.strip()

    def generate_answer(
        self,
        query: str,
        retrieved_documents: List[Dict],
        max_context_docs: Optional[int] = None,
    ) -> str:
        """
        Generate a RAG answer from retrieved documents.

        Args:
            query: User question.
            retrieved_documents: List of dicts with 'title' and 'content'.
            max_context_docs: Limit number of context documents (None = use all).

        Returns:
            Generated answer string.
        """
        if not retrieved_documents:
            return (
                "Вибачте, не вдалося знайти відповідну інформацію "
                "в базі даних університету для вашого запитання."
            )

        docs = retrieved_documents[:max_context_docs] if max_context_docs else retrieved_documents

        contexts = []
        for i, doc in enumerate(docs, 1):
            title = doc.get("title", "Без назви")
            content = doc.get("content", "")
            contexts.append(f"Документ {i} ({title}):\n{content}")

        context_text = "\n\n---\n\n".join(contexts)

        system_prompt = (
            'Ти - асистент для університетської інформаційної системи '
            'Національного університету "Львівська політехніка". '
            "Відповідай ТІЛЬКИ на основі наданих контекстів. "
            "Використовуй конкретні факти, числа, адреси, назви. "
            "Якщо інформації недостатньо, скажи про це чесно. "
            "Відповідай українською мовою."
        )

        user_prompt = (
            f"Контексти з бази знань університету:\n{context_text}\n\n"
            f"Запитання: {query}"
        )

        return self.generate(system_prompt, user_prompt)

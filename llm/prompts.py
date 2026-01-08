def build_rag_prompt(question, context_chunks):
    system_message = """You are Brainy Binder, a helpful AI assistant that answers questions based on a personal knowledge base.

    Your task is to provide accurate, helpful answers based on the context provided below. Follow these guidelines:
    1. Answer the question using ONLY information from the provided context
    2. If the context doesn't contain enough information, say so clearly
    3. Cite sources by mentioning the document name when relevant
    4. Be concise but thorough
    5. If multiple sources provide related information, synthesize them

    Remember: This is a privacy-first system. All information is local and personal to the user."""

    context_text = "RETRIEVED CONTEXT\n\n"

    for i, chunk in enumerate(context_chunks, 1):
        source = chunk.get("source", "Unknown")
        content = chunk.get("content", "")
        context_text += f"[{i}] Source: {source}\n{content}\n\n"

    user_message = f"{context_text}\n## Question\n\n{question}"

    return [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message},
    ]


def build_summarization_prompt(document_text, title):

    system_message = """You are Brainy Binder, a helpful AI assistant that summarizes documents.

    Your task is to create clear, comprehensive summaries. Follow these guidelines:
    1. Capture the main ideas and key points
    2. Use bullet points for readability
    3. Keep the summary concise (3-7 bullets for most documents)
    4. Preserve important details, names, and facts
    5. Organize logically (introduction, main points, conclusion if applicable)"""

    title_text = f" titled '{title}'" if title else ""
    user_message = f"""Please summarize the following document{title_text}: {document_text}. Provide a clear, bullet-point summary."""

    return [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message},
    ]


def build_tagging_prompt(document_text, title):

    system_message = """You are Brainy Binder, a helpful AI assistant that generates semantic tags for documents.

    Your task is to suggest relevant tags/topics for the document. Follow these guidelines:
    1. Generate 3-7 concise tags
    2. Focus on main topics, themes, and key concepts
    3. Use single words or short phrases (2-3 words max)
    4. Be specific but not overly narrow
    5. Return ONLY a JSON array of strings, like: ["tag1", "tag2", "tag3"]
    6. Use lowercase for consistency"""

    title_text = f" titled '{title}'" if title else ""

    max_length = 2000
    if len(document_text) > max_length:
        document_text = document_text[:max_length] + "..."

    user_message = f"""Analyze the following document{title_text} and suggest semantic tags: {document_text}. Return only a JSON array of tag strings."""

    return [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message},
    ]


def build_chat_system_prompt():

    return """You are Brainy Binder, a helpful AI assistant with access to a personal knowledge base.

    You answer questions based on retrieved context from the user's documents (notes, PDFs, bookmarks).
    When provided with context, use it to give accurate, helpful answers. Always cite sources when possible.
    If the context doesn't contain relevant information, say so politely and offer to help in other ways.

    This is a privacy-first system - all data is local and belongs to the user."""
"""
PDF-to-JSON contract extraction.

Reads all PDFs from ./data/input/, extracts structured contract data using the
configured LLM provider, and writes JSON to ./data/output/.

Provider selection is controlled by the LLM_PROVIDER environment variable
(openai | anthropic | gemini | local).  See llm_provider.py for full env-var docs.
"""

import os
import json
from Utils import read_text_file, save_json_string_to_file, extract_json_from_string
from llm_provider import get_provider, Provider

system_instruction = read_text_file('./prompts/system_prompt.txt')
extraction_prompt = read_text_file('./prompts/contract_extraction_prompt.txt')


# ---------------------------------------------------------------------------
# Provider-specific extraction functions
# ---------------------------------------------------------------------------

def _extract_openai(pdf_path: str) -> str:
    """Use OpenAI Assistants API with file-search (best quality for OpenAI)."""
    from openai import OpenAI
    from openai.types.beta.threads.message_create_params import (
        Attachment,
        AttachmentToolFileSearch,
    )

    client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

    assistant = client.beta.assistants.create(
        model=os.getenv("LLM_MODEL_ID", "gpt-4o-2024-08-06"),
        description="An assistant to extract the information from contracts in PDF format.",
        tools=[{"type": "file_search"}],
        name="PDF assistant",
        instructions=system_instruction,
    )
    try:
        thread = client.beta.threads.create()
        file = client.files.create(file=open(pdf_path, "rb"), purpose="assistants")
        client.beta.threads.messages.create(
            thread_id=thread.id,
            role="user",
            attachments=[
                Attachment(file_id=file.id, tools=[AttachmentToolFileSearch(type="file_search")])
            ],
            content=extraction_prompt,
        )
        run = client.beta.threads.runs.create_and_poll(
            thread_id=thread.id, assistant_id=assistant.id, timeout=1000
        )
        if run.status != "completed":
            raise RuntimeError(f"Assistants run failed: {run.status}")
        messages = list(client.beta.threads.messages.list(thread_id=thread.id))
        return messages[0].content[0].text.value
    finally:
        client.beta.assistants.delete(assistant.id)


def _extract_anthropic(pdf_path: str) -> str:
    """Use Anthropic API with native PDF document support (Claude 3.5+)."""
    import anthropic
    import base64

    client = anthropic.Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))

    with open(pdf_path, "rb") as f:
        pdf_data = base64.standard_b64encode(f.read()).decode("utf-8")

    response = client.messages.create(
        model=os.getenv("LLM_MODEL_ID", "claude-3-5-sonnet-20241022"),
        max_tokens=4096,
        system=system_instruction,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "document",
                        "source": {
                            "type": "base64",
                            "media_type": "application/pdf",
                            "data": pdf_data,
                        },
                    },
                    {"type": "text", "text": extraction_prompt},
                ],
            }
        ],
    )
    return response.content[0].text


def _extract_gemini(pdf_path: str) -> str:
    """Use Google Gemini API with native PDF support."""
    import google.generativeai as genai

    genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
    model = genai.GenerativeModel(
        model_name=os.getenv("LLM_MODEL_ID", "gemini-1.5-pro"),
        system_instruction=system_instruction,
    )

    uploaded = genai.upload_file(pdf_path, mime_type="application/pdf")
    response = model.generate_content([uploaded, extraction_prompt])
    # Clean up the temporary file from Google's servers
    genai.delete_file(uploaded.name)
    return response.text


def _extract_local(pdf_path: str) -> str:
    """Extract PDF text locally with pdfplumber, then call the OpenAI-compatible server."""
    import pdfplumber
    from openai import OpenAI

    with pdfplumber.open(pdf_path) as pdf:
        text = "\n".join(page.extract_text() or "" for page in pdf.pages)

    client = OpenAI(
        api_key=os.getenv("LOCAL_API_KEY", "local"),
        base_url=os.getenv("LOCAL_BASE_URL", "http://localhost:11434/v1"),
    )
    response = client.chat.completions.create(
        model=os.getenv("LLM_MODEL_ID", "llama3.2"),
        messages=[
            {"role": "system", "content": system_instruction},
            {
                "role": "user",
                "content": (
                    f"{extraction_prompt}\n\n<document>\n{text}\n</document>"
                ),
            },
        ],
        temperature=0,
    )
    return response.choices[0].message.content


_EXTRACTORS = {
    Provider.OPENAI: _extract_openai,
    Provider.ANTHROPIC: _extract_anthropic,
    Provider.GEMINI: _extract_gemini,
    Provider.LOCAL: _extract_local,
}


def process_pdf(pdf_path: str) -> str:
    provider = get_provider()
    extractor = _EXTRACTORS[provider]
    return extractor(pdf_path)


def main():
    pdf_files = [f for f in os.listdir('./data/input/') if f.endswith('.pdf')]

    for pdf_filename in pdf_files:
        print(f"Processing {pdf_filename}...")
        try:
            complete_response = process_pdf('./data/input/' + pdf_filename)
        except Exception as e:
            print(f"Extraction failed for {pdf_filename}: {e}")
            continue

        # Log raw response for debugging
        save_json_string_to_file(
            complete_response,
            f'./data/debug/complete_response_{pdf_filename}.json',
        )

        # Parse and save valid JSON
        try:
            contract_json = extract_json_from_string(complete_response)
            json_string = json.dumps(contract_json, indent=4)
            save_json_string_to_file(json_string, f'./data/output/{pdf_filename}.json')
        except json.JSONDecodeError as e:
            print(f"Failed to decode JSON for {pdf_filename}: {e}")


if __name__ == '__main__':
    main()

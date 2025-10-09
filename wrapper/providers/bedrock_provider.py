import json
import sys
import os
import boto3
from botocore.exceptions import ClientError
from wrapper.utils import ColorLogger, load_dotenv
from wrapper.base import BaseLLM
from wrapper.config import SHOW_LOGS

log = ColorLogger(enable_debug=SHOW_LOGS)

load_dotenv()

class BedrockProvider(BaseLLM):
    def __init__(self):
        """
        Initialize BedrockProvider using environment variables only.
        No interactive prompting. Safe for builds, Uvicorn, and CI/CD.
        """
        self.region = os.environ.get("AWS_REGION")
        self.aws_access_key_id = os.environ.get("AWS_ACCESS_KEY_ID")
        self.aws_secret_access_key = os.environ.get("AWS_SECRET_ACCESS_KEY")
        self.aws_session_token = os.environ.get("AWS_SESSION_TOKEN")

        self.client = boto3.client(
            "bedrock-runtime",
            region_name=self.region,
            aws_access_key_id=self.aws_access_key_id,
            aws_secret_access_key=self.aws_secret_access_key,
            aws_session_token=self.aws_session_token,
        )

        self.bedrock_client = boto3.client(
            "bedrock",
            region_name=self.region,
            aws_access_key_id=self.aws_access_key_id,
            aws_secret_access_key=self.aws_secret_access_key,
            aws_session_token=self.aws_session_token,
        )

    def generate(
        self,
        model: str,
        prompt: str = None,
        messages: list[dict] = None,
        temperature: float = 0.7,
        max_tokens: int = 200,
        top_p: float = 0.9,
        stream: bool = False,
        **kwargs
    ) -> str:
        """
        Generate a response from Bedrock model.
        Supports both prompt-based and messages-based inputs.
        """
        system_prompt = None
        final_messages = []

        if messages:
            for m in messages:
                if m["role"] == "system":
                    system_prompt = m["content"]
                else:
                    final_messages.append(m)
        else:
            system_prompt = "You are a helpful AI assistant."
            final_messages = [{"role": "user", "content": prompt}]

        body_dict = {
            "messages": final_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
            "anthropic_version": "bedrock-2023-05-31",
            **kwargs
        }

        if system_prompt:
            body_dict["system"] = system_prompt

        body = json.dumps(body_dict)

        if not stream:
            resp = self.client.invoke_model(
                modelId=model,
                body=body,
                accept="application/json",
                contentType="application/json"
            )
            response_body = json.loads(resp["body"].read().decode("utf-8"))
            return response_body["content"][0]["text"].strip()
        else:
            output = []
            stream_resp = self.client.invoke_model_with_response_stream(
                modelId=model,
                body=body,
                accept="application/json",
                contentType="application/json"
            )
            for event in stream_resp["body"]:
                chunk = json.loads(event["chunk"]["bytes"].decode("utf-8"))
                if chunk.get("type") == "content_block_delta":
                    delta = chunk["delta"]["text"]
                    sys.stdout.write(delta)
                    sys.stdout.flush()
                    output.append(delta)
            return "".join(output).strip()

    def list_models(self, by_provider: str = None, by_output_modality: str = None, **kwargs):
        """
        List all Bedrock foundation models.
        Optional filtering by provider or output modality.
        """
        try:
            params = {}
            if by_provider:
                params["byProvider"] = by_provider
            if by_output_modality:
                params["byOutputModality"] = by_output_modality

            resp = self.bedrock_client.list_foundation_models(**params)
            summaries = resp.get("modelSummaries", [])
            models = []
            for m in summaries:
                models.append({
                    "modelId": m.get("modelId"),
                    "modelName": m.get("modelName"),
                    "provider": m.get("providerName"),
                    "outputModalities": m.get("outputModalities", []),
                    "inputModalities": m.get("inputModalities", []),
                    "responseStreamingSupported": m.get("responseStreamingSupported", False),
                })
            return models
        except ClientError as e:
            log.error(f"Bedrock: failed to list foundation models: {e}")
            return []

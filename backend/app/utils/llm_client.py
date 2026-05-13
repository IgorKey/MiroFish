"""
LLM客户端封装
统一使用OpenAI格式调用
"""

import json
import re
from typing import Optional, Dict, Any, List
from openai import OpenAI

from ..config import Config


class LLMClient:
    """LLM客户端"""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None
    ):
        self.api_key = api_key or Config.LLM_API_KEY
        self.base_url = base_url or Config.LLM_BASE_URL
        self.model = model or Config.LLM_MODEL_NAME
        
        if not self.api_key:
            raise ValueError("LLM_API_KEY 未配置")
        
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 4096,
        response_format: Optional[Dict] = None
    ) -> str:
        """
        发送聊天请求
        
        Args:
            messages: 消息列表
            temperature: 温度参数
            max_tokens: 最大token数
            response_format: 响应格式（如JSON模式）
            
        Returns:
            模型响应文本
        """
        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        
        if response_format:
            kwargs["response_format"] = response_format
        
        response = self.client.chat.completions.create(**kwargs)
        content = response.choices[0].message.content
        # 部分模型（如MiniMax M2.5）会在content中包含<think>思考内容，需要移除
        content = re.sub(r'<think>[\s\S]*?</think>', '', content).strip()
        return content
    
    def chat_json(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.3,
        max_tokens: int = 4096
    ) -> Dict[str, Any]:
        """
        Send chat request and return parsed JSON.
        Adds a repair retry when the model ignores JSON mode.
        """
    json_instruction = {
        "role": "user",
        "content": (
            "IMPORTANT: Return valid JSON only. "
            "Do not include markdown, explanations, headings, bullet lists, or code fences. "
            "The response must start with '{' and end with '}'."
        )
    }

    response = self.chat(
        messages=messages + [json_instruction],
        temperature=temperature,
        max_tokens=max_tokens,
        response_format={"type": "json_object"}
    )

    cleaned_response = response.strip()
    cleaned_response = re.sub(r'^```(?:json)?\s*\n?', '', cleaned_response, flags=re.IGNORECASE)
    cleaned_response = re.sub(r'\n?```\s*$', '', cleaned_response).strip()

    try:
        return json.loads(cleaned_response)
    except json.JSONDecodeError:
        # Retry: ask the model to convert its own invalid response into strict JSON.
        repair_messages = [
            {
                "role": "system",
                "content": (
                    "You are a JSON repair tool. "
                    "Convert the provided text into valid JSON only. "
                    "Do not include markdown or explanations. "
                    "The JSON must match the schema requested in the previous task."
                )
            },
            {
                "role": "user",
                "content": (
                    "The previous response was not valid JSON. "
                    "Convert it into valid JSON only.\n\n"
                    f"Invalid response:\n{cleaned_response}"
                )
            }
        ]

        repaired = self.chat(
            messages=repair_messages,
            temperature=0,
            max_tokens=max_tokens,
            response_format={"type": "json_object"}
        )

        repaired = repaired.strip()
        repaired = re.sub(r'^```(?:json)?\s*\n?', '', repaired, flags=re.IGNORECASE)
        repaired = re.sub(r'\n?```\s*$', '', repaired).strip()

        try:
            return json.loads(repaired)
        except json.JSONDecodeError:
            raise ValueError(f"LLM returned invalid JSON even after repair: {repaired}")


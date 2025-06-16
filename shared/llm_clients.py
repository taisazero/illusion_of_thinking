from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple, Union
from openai import OpenAI
import json
import anthropic
from google import genai
import time
from pathlib import Path
import requests
from typing import Optional
from tqdm import tqdm
import tempfile
import os
import subprocess


class BaseLLMClient(ABC):
    @abstractmethod
    async def create_message(self, messages: List[Dict[str, str]], with_tools: bool = False) -> Any:
        pass

    @abstractmethod
    def get_tool_info(self, tool: Dict[str, Any]) -> Dict[str, str]:
        pass

    @abstractmethod
    def process_tool_result(self, tool_use: Any) -> Tuple[str, Dict[str, Any]]:
        pass
    
    @abstractmethod
    def create_batch_messages(self, message_batches: List[List[Dict[str, str]]], **kwargs) -> List[Any]:
        """Process multiple message conversations in batch for efficiency"""
        pass



class OpenAIClient(BaseLLMClient):
    def __init__(self, api_key: str, base_url: str = None):
        if base_url:
            self.client = OpenAI(api_key=api_key, base_url=base_url)
        else:
            self.client = OpenAI(api_key=api_key)

    def create_message(self, messages: List[Dict[str, str]], tools: List = None, schema: Union[str, Dict[str, Any]] = None, kwargs: dict = None, reasoning: bool = False, reasoning_effort: str = "medium") -> Any:
        if kwargs is None:
            kwargs = {
                "model": "gpt-4o",
                "messages": messages,
                "max_tokens": 4000,
                "temperature": 0.1,
            }
        else:
            kwargs["messages"] = messages if 'messages' not in kwargs else kwargs["messages"]
            kwargs["max_tokens"] = 4000 if "max_tokens" not in kwargs else kwargs["max_tokens"]
            kwargs["temperature"] = 0.1 if "temperature" not in kwargs else kwargs["temperature"]

        # Handle reasoning models (o1, o3, o3-mini, o4-mini series)
        model_name = kwargs.get("model", "gpt-4o")
        is_reasoning_model = any(model_name.startswith(prefix) for prefix in ["o1", "o3", "o4"])
        
        if is_reasoning_model:
            # For reasoning models, use max_completion_tokens instead of max_tokens
            # Need to allocate more tokens since they include both reasoning and output
            if "max_tokens" in kwargs:
                base_tokens = kwargs.pop("max_tokens")
                if reasoning:
                    # Allocate different amounts based on reasoning effort level
                    effort_multipliers = {
                        "low": 3000,      # Conservative for low effort
                        "medium": 5000,   # Moderate for medium effort  
                        "high": 8000      # Generous for high effort
                    }
                    extra_tokens = effort_multipliers.get(reasoning_effort, 5000)
                    kwargs["max_completion_tokens"] = base_tokens + extra_tokens
                else:
                    kwargs["max_completion_tokens"] = base_tokens
            
            # Add reasoning effort for compatible models
            if reasoning and any(model_name.startswith(prefix) for prefix in ["o1", "o3", "o4"]):
                kwargs["reasoning_effort"] = reasoning_effort
                
            # Remove unsupported parameters for reasoning models
            unsupported_params = ["temperature", "top_p", "presence_penalty", "frequency_penalty", 
                                "logprobs", "top_logprobs", "logit_bias"]
            for param in unsupported_params:
                kwargs.pop(param, None)

        if tools:
            kwargs["tools"] = tools

        if schema:
            kwargs["extra_body"] = {
                "guided_json": schema,
                "guided_decoding_backend": "lm-format-enforcer",
            }
        
        response = self.client.chat.completions.create(**kwargs)
        
        # Handle reasoning models response parsing
        if is_reasoning_model:
            response_content = response.choices[0].message.content
            

            # Check if we have reasoning token information and reasoning is enabled
            if (reasoning and hasattr(response, 'usage') and hasattr(response.usage, 'completion_tokens_details') 
                and hasattr(response.usage.completion_tokens_details, 'reasoning_tokens') 
                and response.usage.completion_tokens_details.reasoning_tokens > 0):
                reasoning_tokens = response.usage.completion_tokens_details.reasoning_tokens
                # Create a reasoning placeholder since OpenAI doesn't expose the actual reasoning content
                reasoning_content = f"The model used {reasoning_tokens} reasoning tokens to process this request."
                combined_content = f"<reasoning>\n{reasoning_content}\n</reasoning>\n\n{response_content}"
                return combined_content
            else:
                # Return the actual response content for reasoning models
                return response_content
        
        # Handle regular models
        response_content = response.choices[0].message.content
        if tools and response.choices[0].message.tool_calls:
            return {
                "tool_calls": [
                    {
                        "function": {
                            "name": tool_call.function.name,
                            "arguments": json.loads(tool_call.function.arguments),
                        }
                    }
                    for tool_call in response.choices[0].message.tool_calls
                ]
            }
        elif tools:
            tool_call = None
            # Check if response_content contains indicators of a tool call
            if response_content and isinstance(response_content, str) and '"name":' in response_content:
                # Clean up tokens that might interfere with JSON parsing
                cleaned_content = response_content.strip()
                for token in ["<|eom_id|>", "<|eot_id|>"]:
                    cleaned_content = cleaned_content.replace(token, "")
                
                try:
                    tool_call = json.loads(cleaned_content)
                except json.decoder.JSONDecodeError:
                    # Try adding a closing brace if it's missing
                    try:
                        tool_call = json.loads(cleaned_content + "}")
                    except json.decoder.JSONDecodeError:
                        pass
            
            if tool_call is not None:
                return {
                    "tool_calls": [
                        {
                            "function": {
                                "name": tool_call["name"],
                                "arguments": tool_call["parameters"],
                            }
                        }
                    ]
                }
            
        return response_content
    
    def get_tool_info(self, tool: Dict[str, Any]) -> Dict[str, str]:
        return {
            "name": tool["function"]["name"],
            "description": tool["function"]["description"]
        }

    def process_tool_result(self, tool_use: Any) -> Tuple[str, Dict[str, Any]]:
        return tool_use['function']['name'], json.loads(tool_use['function']['arguments'])
    
    def create_batch_messages(self, message_batches: List[List[Dict[str, str]]], reasoning: bool = False, reasoning_effort: str = "medium", **kwargs) -> List[Any]:
        """
        Process multiple message conversations using OpenAI's Batch API.
        """
        try:
            # Prepare batch input as per OpenAI Batch API
            batch_requests = []
            for i, messages in enumerate(message_batches):
                model_name = kwargs.get("model", "gpt-4o")
                is_reasoning_model = any(model_name.startswith(prefix) for prefix in ["o1", "o3", "o4"])
                
                body = {
                    "model": model_name,
                    "messages": messages,
                }
                
                if is_reasoning_model:
                    # For reasoning models, use max_completion_tokens with extra allocation
                    base_tokens = kwargs.get("max_tokens", 4000)
                    if reasoning:
                        # Allocate different amounts based on reasoning effort level
                        effort_multipliers = {
                            "low": 3000,      # Conservative for low effort
                            "medium": 5000,   # Moderate for medium effort  
                            "high": 8000      # Generous for high effort
                        }
                        extra_tokens = effort_multipliers.get(reasoning_effort, 5000)
                        body["max_completion_tokens"] = base_tokens + extra_tokens
                        body["reasoning_effort"] = reasoning_effort
                    else:
                        body["max_completion_tokens"] = base_tokens
                else:
                    # For regular models, use max_tokens and temperature
                    body["max_tokens"] = kwargs.get("max_tokens", 4000)
                    body["temperature"] = kwargs.get("temperature", 0.1)
                
                batch_requests.append({
                    "custom_id": str(i),
                    "method": "POST", 
                    "url": "/v1/chat/completions",
                    "body": body
                })
            
            # Write batch to a temp file
            with tempfile.NamedTemporaryFile(delete=False, mode="w", suffix=".jsonl") as f:
                for req in batch_requests:
                    f.write(json.dumps(req) + "\n")
                batch_file_path = f.name
            
            # Upload batch file
            with open(batch_file_path, "rb") as batch_file:
                batch_input_file = self.client.files.create(
                    file=batch_file,
                    purpose="batch"
                )
            
            # Create batch
            batch = self.client.batches.create(
                input_file_id=batch_input_file.id,
                endpoint="/v1/chat/completions",
                completion_window="24h"
            )
            
            print(f"OpenAI batch created: {batch.id}")
            
            # Poll for completion
            while batch.status not in ("completed", "failed", "expired", "canceled", "ended"):
                print(f"Batch status: {batch.status}, waiting...")
                time.sleep(10)
                batch = self.client.batches.retrieve(batch.id)
            
            if batch.status != "completed":
                print(f"Batch failed with status: {batch.status}")
                return [f"Batch failed: {batch.status}"] * len(message_batches)
            
            # Download results
            output_file_id = batch.output_file_id
            result = self.client.files.content(output_file_id)
            
            # Parse results and map by custom_id
            results = {}
            for line in result.content.decode('utf-8').strip().split('\n'):
                if line.strip():
                    obj = json.loads(line)
                    if "custom_id" in obj:
                        # Extract the actual response content
                        if obj.get("response") and obj["response"].get("body"):
                            response_body = obj["response"]["body"]
                            if response_body.get("choices"):
                                content = response_body["choices"][0]["message"]["content"]
                                results[obj["custom_id"]] = content
                            else:
                                results[obj["custom_id"]] = str(response_body)
                        else:
                            results[obj["custom_id"]] = str(obj)
            
            # Clean up temp file
            os.unlink(batch_file_path)
            
            # Return in input order
            return [results.get(str(i), f"No result for index {i}") for i in range(len(message_batches))]
            
        except Exception as e:
            print(f"OpenAI batch processing failed: {e}")
            # Fallback to sequential processing
            print("Falling back to sequential processing...")
            responses = []
            with tqdm(total=len(message_batches), desc="Processing OpenAI requests (sequential)", unit="req") as pbar:
                for i, messages in enumerate(message_batches):
                    try:
                        response = self.create_message(messages, kwargs=kwargs)
                        responses.append(response)
                    except Exception as e:
                        print(f"Error processing batch item {i}: {e}")
                        responses.append(f"Error: {str(e)}")
                    pbar.update(1)
            return responses
    

class VLLMClient(OpenAIClient):
    def __init__(self, api_key: str = "NONE", 
                 model_name: str = "", 
                 base_url: str = "http://localhost:8000/v1"
                 ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        
        available_model = self.get_llm_server_modelname()
    
        if model_name:
            if model_name != available_model:
                print(f"Requested model '{model_name}' is not available. Using the available model: '{available_model}'")
            self.model_name = model_name
        elif available_model:
            self.model_name = available_model
            print(f"Language model name not set, using the available model: '{available_model}'")
        else:
            raise ValueError("No model is available on the VLLM server. Please ensure that the VLLM server is running and is serving a language model.")
        
        super().__init__(api_key=api_key, base_url=base_url)

    def set_base_url(self, base_url: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.client.base_url = self.base_url
        
    def create_message(self, messages: List[Dict[str, str]], tools: List = None, schema: Union[str, Dict[str, Any]] = None, kwargs: dict = None, reasoning: bool = False, reasoning_effort: str = "medium") -> Any:
        if kwargs is None:
            kwargs = {
                "model": self.model_name,
                "messages": messages,
                "max_tokens": 4000,
                "temperature": 0.1,
                "top_p": 0.9,
            }
        else:
            kwargs["model"] = self.model_name if "model" not in kwargs else kwargs["model"]
            kwargs["messages"] = messages if 'messages' not in kwargs else kwargs["messages"]
            kwargs["max_tokens"] = 4000 if "max_tokens" not in kwargs else kwargs["max_tokens"]
            kwargs["temperature"] = 0.1 if "temperature" not in kwargs else kwargs["temperature"]
            kwargs["top_p"] = 0.9 if "top_p" not in kwargs else kwargs["top_p"]
        return super().create_message(messages, tools, schema, kwargs, reasoning, reasoning_effort)

    def get_llm_server_modelname(self) -> Optional[str]:
        base_url = self.base_url.replace("/v1", "").rstrip("/")
        try:
            if self.api_key:
                response = requests.get(
                    f"{base_url}/v1/models", headers={"Authorization": f"Bearer {self.api_key}"}
                )
            else:
                response = requests.get(f"{base_url}/v1/models")
            if response.status_code == 200:
                models = [m for m in response.json()["data"] if m["object"] == "model"]
                if len(models) == 0:
                    print("The vLLM server is running but not hosting any models.")
                    return None
                model_name = models[0]["id"]
                print(f"vLLM server is running. Selecting: {model_name}.")
                return model_name
            else:
                print(f"vLLM server is running but could not get the list of models. Status code: {response.status_code}")
                return None
        except requests.exceptions.ConnectionError:
            print("No vLLM server running at the specified URL.")
            return None
        except Exception as e:
            print(f"Error while trying to get the vLLM model name: {e}")
            return None

    def create_batch_messages(self, message_batches: List[List[Dict[str, str]]], timeout: int = 18000, reasoning: bool = False, reasoning_effort: str = "medium", **kwargs) -> List[Any]:
        """
        Process multiple message conversations using vLLM's offline batch processing.
        Based on: https://docs.vllm.ai/en/stable/examples/offline_inference/openai_batch.html
        """
        try:
            # Prepare batch input as per vLLM OpenAI batch format
            batch_requests = []
            for i, messages in enumerate(message_batches):
                batch_requests.append({
                    "custom_id": f"request-{i}",
                    "method": "POST",
                    "url": "/v1/chat/completions", 
                    "body": {
                        "model": kwargs.get("model", self.model_name),
                        "messages": messages,
                        "max_completion_tokens": kwargs.get("max_tokens", 4000),
                        "temperature": kwargs.get("temperature", 0.1),
                    }
                })
            
            # Write batch to temp file
            with tempfile.NamedTemporaryFile(delete=False, mode="w", suffix=".jsonl") as input_file:
                for req in batch_requests:
                    input_file.write(json.dumps(req) + "\n")
                input_file_path = input_file.name
            
            # Create output file path
            output_file_path = input_file_path.replace(".jsonl", "_output.jsonl")
            
            # Run vLLM batch processing
            cmd = [
                "python", "-m", "vllm.entrypoints.openai.run_batch",
                "-i", input_file_path,
                "-o", output_file_path,
                "--model", kwargs.get("model", self.model_name)
            ]
            
            print(f"Running vLLM batch command: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)  # 5 hour timeout
            
            if result.returncode != 0:
                print(f"vLLM batch processing failed: {result.stderr}")
                raise Exception(f"vLLM batch failed: {result.stderr}")
            
            # Read results
            results = {}
            if os.path.exists(output_file_path):
                with open(output_file_path, "r") as f:
                    for line in f:
                        if line.strip():
                            obj = json.loads(line)
                            if "custom_id" in obj:
                                # Extract response content
                                if obj.get("response") and obj["response"].get("body"):
                                    response_body = obj["response"]["body"]
                                    if response_body.get("choices"):
                                        content = response_body["choices"][0]["message"]["content"]
                                        # Extract request index from custom_id
                                        request_id = obj["custom_id"].replace("request-", "")
                                        results[request_id] = content
                                    else:
                                        request_id = obj["custom_id"].replace("request-", "")
                                        results[request_id] = str(response_body)
            
            # Clean up temp files
            os.unlink(input_file_path)
            if os.path.exists(output_file_path):
                os.unlink(output_file_path)
            
            # Return in input order
            return [results.get(str(i), f"No result for index {i}") for i in range(len(message_batches))]
            
        except Exception as e:
            print(f"vLLM batch processing failed: {e}")
            # Fallback to sequential processing
            print("Falling back to sequential processing...")
            responses = []
            with tqdm(total=len(message_batches), desc="Processing vLLM requests (sequential)", unit="req") as pbar:
                for i, messages in enumerate(message_batches):
                    try:
                        response = self.create_message(messages, kwargs=kwargs)
                        responses.append(response)
                    except Exception as e:
                        print(f"Error processing batch item {i}: {e}")
                        responses.append(f"Error: {str(e)}")
                    pbar.update(1)
            return responses

class AnthropicClient(BaseLLMClient):
    def __init__(self, api_key: str):
        self.client = anthropic.Anthropic(api_key=api_key)

    def create_message(self, messages: List[Dict[str, str]], tools: List = None, schema: Union[str, Dict[str, Any]] = None, kwargs: dict = None, reasoning: bool = False, budget_tokens: int = 1000) -> Any:
        if kwargs is None:
            kwargs = {
                "model": "claude-3-5-sonnet-20241022",
                "max_tokens": 4000,
                "temperature": 0.1,
            }
        else:
            kwargs["model"] = kwargs.get("model", "claude-3-5-sonnet-20241022")
            kwargs["max_tokens"] = kwargs.get("max_tokens", 4000)
            kwargs["temperature"] = kwargs.get("temperature", 0.1)

        # Convert messages format for Anthropic
        # Anthropic expects 'content' instead of 'content' and handles system messages differently
        anthropic_messages = []
        system_message = None
        
        for message in messages:
            if message.get("role") == "system":
                system_message = message.get("content", "")
            else:
                anthropic_messages.append({
                    "role": message.get("role"),
                    "content": message.get("content", "")
                })

        # Set up the request parameters
        request_params = {
            "model": kwargs["model"],
            "messages": anthropic_messages,
            "max_tokens": kwargs["max_tokens"],
            "temperature": kwargs["temperature"]
        }

        if system_message:
            request_params["system"] = system_message

        if tools:
            # Convert tools to Anthropic format
            anthropic_tools = []
            for tool in tools:
                if "function" in tool:
                    anthropic_tools.append({
                        "name": tool["function"]["name"],
                        "description": tool["function"]["description"],
                        "input_schema": tool["function"]["parameters"]
                    })
            request_params["tools"] = anthropic_tools

        # Add extended thinking if reasoning is enabled
        if reasoning:
            request_params['temperature'] = 1.0
            request_params['max_tokens'] = request_params['max_tokens'] + budget_tokens
            request_params["thinking"] = {
                "type": "enabled",
                "budget_tokens": budget_tokens,
            }

        # Enable streaming for long operations (>10k tokens total)
        total_tokens = request_params.get('max_tokens', 0)
        use_streaming = total_tokens > 10000
        
        if use_streaming:
            print(f"🔄 Using streaming for long operation ({total_tokens} max tokens)")

        try:
            if use_streaming:
                # Use streaming for long operations with proper thinking block handling
                stream = self.client.messages.create(
                    stream=True,
                    **request_params
                )
                
                # Collect streaming response with proper block tracking
                thinking_content = ""
                text_content = ""
                current_block_type = None
                
                for chunk in stream:
                    if hasattr(chunk, 'type'):
                        if chunk.type == 'content_block_start':
                            if hasattr(chunk, 'content_block') and hasattr(chunk.content_block, 'type'):
                                current_block_type = chunk.content_block.type
                        elif chunk.type == 'content_block_delta':
                            if hasattr(chunk, 'delta'):
                                if hasattr(chunk.delta, 'text') and current_block_type == 'text':
                                    text_content += chunk.delta.text
                                elif hasattr(chunk.delta, 'thinking') and current_block_type == 'thinking':
                                    thinking_content += chunk.delta.thinking
                
                # Create a mock response object for consistent handling
                class MockResponse:
                    def __init__(self, thinking, text):
                        self.content = []
                        if thinking:
                            self.content.append(type('ThinkingBlock', (), {'type': 'thinking', 'thinking': thinking})())
                        if text:
                            self.content.append(type('TextBlock', (), {'type': 'text', 'text': text})())
                
                response = MockResponse(thinking_content, text_content)
            else:
                # Standard non-streaming call
                response = self.client.messages.create(**request_params)
            
            # Handle tool calls
            if tools and hasattr(response, 'content') and response.content:
                for content_block in response.content:
                    if hasattr(content_block, 'type') and content_block.type == 'tool_use':
                        return {
                            "tool_calls": [
                                {
                                    "function": {
                                        "name": content_block.name,
                                        "arguments": content_block.input,
                                    }
                                }
                            ]
                        }
            
            # Extract thinking and text content
            if hasattr(response, 'content') and response.content:
                thinking_content = ""
                text_content = ""
                
                for content_block in response.content:
                    if hasattr(content_block, 'type'):
                        if content_block.type == 'thinking':
                            thinking_content += content_block.thinking
                        elif content_block.type == 'text':
                            text_content += content_block.text
                
                # Combine thinking and text with XML formatting
                combined_content = ""
                if thinking_content:
                    combined_content += f"<reasoning>\n{thinking_content}\n</reasoning>\n\n"
                combined_content += text_content
                
                return combined_content if combined_content else ""
            
            return ""
            
        except Exception as e:
            print(f"Error calling Anthropic API: {e}")
            return f"Error: {str(e)}"

    def get_tool_info(self, tool: Dict[str, Any]) -> Dict[str, str]:
        return {
            "name": tool["function"]["name"],
            "description": tool["function"]["description"]
        }

    def process_tool_result(self, tool_use: Any) -> Tuple[str, Dict[str, Any]]:
        return tool_use['function']['name'], tool_use['function']['arguments']
    
    def create_batch_messages(self, message_batches: List[List[Dict[str, str]]], reasoning: bool = False, budget_tokens: int = 1000, **kwargs) -> List[Any]:
        """
        Process multiple message conversations using Anthropic's Message Batches API.
        """
        try:
            # Prepare batch input as per Anthropic Message Batches API
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY environment variable not set")
                
            headers = {
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json"
            }
            
            batch_requests = []
            for i, messages in enumerate(message_batches):
                # Convert messages for Anthropic format
                anthropic_messages = []
                system_message = None
                
                for message in messages:
                    if message.get("role") == "system":
                        system_message = message.get("content", "")
                    else:
                        anthropic_messages.append({
                            "role": message.get("role"),
                            "content": message.get("content", "")
                        })
                
                params = {
                    "model": kwargs.get("model", "claude-3-5-sonnet-20241022"),
                    "max_tokens": kwargs.get("max_tokens", 4000),
                    "temperature": kwargs.get("temperature", 0.1),
                    "messages": anthropic_messages
                }
                
                if system_message:
                    params["system"] = system_message
                
                # Add extended thinking if reasoning is enabled
                if reasoning:
                    params['max_tokens'] = params['max_tokens'] + budget_tokens
                    params['temperature'] = 1.0
                    params["thinking"] = {
                        "type": "enabled",
                        "budget_tokens": budget_tokens,
                    }
                
                batch_requests.append({
                    "custom_id": str(i),
                    "params": params
                })
            
            # Create batch
            payload = {"requests": batch_requests}
            resp = requests.post(
                "https://api.anthropic.com/v1/messages/batches",
                headers=headers,
                data=json.dumps(payload)
            )
            
            if resp.status_code != 200:
                print(f"Anthropic batch creation failed: {resp.text}")
                raise Exception(f"Batch creation failed: {resp.text}")
            
            batch_data = resp.json()
            batch_id = batch_data["id"]
            print(f"Anthropic batch created: {batch_id}")
            
            # Poll for completion
            status = batch_data["processing_status"]
            while status not in ("completed", "failed", "expired", "canceled", "ended"):
                print(f"Batch status: {status}, waiting...")
                time.sleep(10)
                status_resp = requests.get(
                    f"https://api.anthropic.com/v1/messages/batches/{batch_id}", 
                    headers=headers
                )
                if status_resp.status_code == 200:
                    status = status_resp.json()["processing_status"]
                else:
                    print(f"Error checking batch status: {status_resp.text}")
                    break
            
            if status != "completed":
                print(f"Batch failed with status: {status}")
                raise Exception(f"Batch failed: {status}")
            
            # Get results
            results_resp = requests.get(
                f"https://api.anthropic.com/v1/messages/batches/{batch_id}/results", 
                headers=headers
            )
            
            if results_resp.status_code != 200:
                print(f"Failed to get batch results: {results_resp.text}")
                raise Exception(f"Failed to get results: {results_resp.text}")
            
            results_data = results_resp.json()
            
            # Map results by custom_id
            results = {}
            for item in results_data.get("results", []):
                custom_id = item.get("custom_id")
                result = item.get("result")
                
                if custom_id and result:
                    if result.get("type") == "message" and result.get("content"):
                        # Extract thinking and text content
                        thinking_content = ""
                        text_content = ""
                        
                        for content_block in result["content"]:
                            if content_block.get("type") == "thinking":
                                thinking_content += content_block.get("thinking", "")
                            elif content_block.get("type") == "text":
                                text_content += content_block.get("text", "")
                        
                        # Combine thinking and text with XML formatting
                        combined_content = ""
                        if thinking_content:
                            combined_content += f"<reasoning>\n{thinking_content}\n</reasoning>\n\n"
                        combined_content += text_content
                        
                        results[custom_id] = combined_content if combined_content else text_content
                    else:
                        results[custom_id] = str(result)
                elif custom_id:
                    error = item.get("error", "Unknown error")
                    results[custom_id] = f"Error: {error}"
            
            # Return in input order
            return [results.get(str(i), f"No result for index {i}") for i in range(len(message_batches))]
            
        except Exception as e:
            print(f"Anthropic batch processing failed: {e}")
            # Fallback to sequential processing
            print("Falling back to sequential processing...")
            responses = []
            with tqdm(total=len(message_batches), desc="Processing Anthropic requests (sequential)", unit="req") as pbar:
                for i, messages in enumerate(message_batches):
                    try:
                        response = self.create_message(messages, kwargs=kwargs, reasoning=reasoning, budget_tokens=budget_tokens)
                        responses.append(response)
                    except Exception as e:
                        print(f"Error processing batch item {i}: {e}")
                        responses.append(f"Error: {str(e)}")
                    pbar.update(1)
            return responses


class GeminiClient(BaseLLMClient):
    def __init__(self, api_key: str, model: str = "gemini-1.5-flash-latest"):
        self.model = model
        self.client = genai.Client(api_key=api_key)

    def create_message(self, messages: List[Dict[str, str]], tools: List = None, schema: Union[str, Dict[str, Any]] = None, kwargs: dict = None) -> Any:
        # Convert messages to the format expected by the new genai client
        contents = []
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            
            if role == "system":
                # System messages are handled via config in the new API
                contents.append(f"System: {content}")
            else:
                contents.append(content)
        
        # Combine all content parts
        combined_content = "\n".join(contents)
        
        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=combined_content
            )
            return response.text if hasattr(response, "text") else str(response)
        except Exception as e:
            return f"Error: {str(e)}"

    def get_tool_info(self, tool: Dict[str, Any]) -> Dict[str, str]:
        return {"name": tool.get("name", ""), "description": tool.get("description", "")}

    def process_tool_result(self, tool_use: Any) -> Tuple[str, Dict[str, Any]]:
        return tool_use.get("name", ""), tool_use.get("arguments", {})

    def create_batch_messages(self, message_batches: List[List[Dict[str, str]]], **kwargs) -> List[Any]:
        """
        Gemini does not support batch processing, so we process sequentially.
        """
        responses = []
        with tqdm(total=len(message_batches), desc="Processing Gemini requests (sequential)", unit="req") as pbar:
            for i, messages in enumerate(message_batches):
                try:
                    response = self.create_message(messages, kwargs=kwargs)
                    responses.append(response)
                except Exception as e:
                    print(f"Error processing batch item {i}: {e}")
                    responses.append(f"Error: {str(e)}")
                pbar.update(1)
        return responses
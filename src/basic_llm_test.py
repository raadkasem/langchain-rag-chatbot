#!/usr/bin/env python3
"""
Basic LLM integration test script.
This script demonstrates basic interaction with OpenAI's API.
"""

import os
from dotenv import load_dotenv
from openai import OpenAI

def test_basic_llm():
    load_dotenv()
    
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("Error: OPENAI_API_KEY not found in environment variables.")
        print("Please copy .env.example to .env and add your API key.")
        return False
    
    try:
        client = OpenAI(api_key=api_key)
        
        # Test basic chat completion
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello! Can you tell me a brief joke about programming?"}
            ],
            max_tokens=150,
            temperature=0.7
        )
        
        print("✅ LLM Integration Test Successful!")
        print(f"Model: {response.model}")
        print(f"Response: {response.choices[0].message.content}")
        print(f"Tokens used: {response.usage.total_tokens}")
        return True
        
    except Exception as e:
        print(f"❌ LLM Integration Test Failed: {str(e)}")
        return False

def test_different_prompts():
    """Test the LLM with different types of prompts"""
    load_dotenv()
    
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        return False
    
    client = OpenAI(api_key=api_key)
    
    prompts = [
        "Explain what a chatbot is in one sentence.",
        "List 3 benefits of using AI in customer service.",
        "What is the difference between supervised and unsupervised learning?"
    ]
    
    print("\n🧪 Testing different prompt types:")
    
    for i, prompt in enumerate(prompts, 1):
        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a concise and helpful AI assistant."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=100,
                temperature=0.5
            )
            
            print(f"\n{i}. Prompt: {prompt}")
            print(f"   Response: {response.choices[0].message.content}")
            
        except Exception as e:
            print(f"   Error: {str(e)}")

if __name__ == "__main__":
    print("🚀 Starting Basic LLM Integration Test")
    print("=" * 50)
    
    success = test_basic_llm()
    
    if success:
        test_different_prompts()
    
    print("\n" + "=" * 50)
    print("Test completed!")
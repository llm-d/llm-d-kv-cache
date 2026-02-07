#!/usr/bin/env python3
"""
Test script to verify that render_jinja_template_wrapper.py supports multi-modality content blocks.
This tests the OpenAI API format with image_url content blocks.
"""

import json
import sys
import os

# Add the current directory to the path (render_jinja_template_wrapper.py is in the same directory)
sys.path.insert(0, os.path.dirname(__file__))

from render_jinja_template_wrapper import render_jinja_template

def test_text_only():
    """Test 1: Text-only content (backward compatibility)"""
    print("=== Test 1: Text-only Content (Backward Compatibility) ===")
    
    request = {
        "messages": [
            {"role": "user", "content": "Hello, how are you?"},
            {"role": "assistant", "content": "I'm doing well, thank you!"}
        ],
        "chat_template": "{% for message in messages %}{{ message.role }}: {{ message.content }}\n{% endfor %}"
    }
    
    try:
        result_json = render_jinja_template(json.dumps(request))
        result = json.loads(result_json)
        print("Text-only content works")
        print(f"   Rendered: {result['rendered_chats'][0][:100]}...")
        return True
    except Exception as e:
        print(f"Error: Text-only content failed: {e}")
        return False

def test_multimodality_image_url():
    """Test 2: Multi-modality with image_url (URL format)"""
    print("\n=== Test 2: Multi-modality with Image URL ===")
    
    request = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is in this image?"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "https://example.com/image.jpg"
                        }
                    }
                ]
            }
        ],
        "chat_template": "{% for message in messages %}{{ message.role }}: {{ message.content }}\n{% endfor %}"
    }
    
    try:
        result_json = render_jinja_template(json.dumps(request))
        result = json.loads(result_json)
        print("Multi-modality with image URL works")
        print(f"   Rendered: {result['rendered_chats'][0][:200]}...")
        return True
    except Exception as e:
        print(f"Error: Multi-modality with image URL failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_multimodality_base64():
    """Test 3: Multi-modality with base64 image"""
    print("\n=== Test 3: Multi-modality with Base64 Image ===")
    
    # Small 1x1 red pixel PNG in base64
    base64_image = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
    
    request = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": base64_image
                        }
                    }
                ]
            }
        ],
        "chat_template": "{% for message in messages %}{{ message.role }}: {{ message.content }}\n{% endfor %}"
    }
    
    try:
        result_json = render_jinja_template(json.dumps(request))
        result = json.loads(result_json)
        print("Multi-modality with base64 image works")
        print(f"   Rendered: {result['rendered_chats'][0][:200]}...")
        return True
    except Exception as e:
        print(f"Error: Multi-modality with base64 image failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_multimodality_multiple_images():
    """Test 4: Multiple images in one message"""
    print("\n=== Test 4: Multiple Images in One Message ===")
    
    request = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What are the animals?"},
                    {"type": "image_url", "image_url": {"url": "https://example.com/image1.jpg"}},
                    {"type": "image_url", "image_url": {"url": "https://example.com/image2.jpg"}}
                ]
            }
        ],
        "chat_template": "{% for message in messages %}{{ message.role }}: {{ message.content }}\n{% endfor %}"
    }
    
    try:
        result_json = render_jinja_template(json.dumps(request))
        result = json.loads(result_json)
        print("Multiple images in one message works")
        print(f"   Rendered: {result['rendered_chats'][0][:200]}...")
        return True
    except Exception as e:
        print(f"Error: Multiple images failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("Testing Python wrapper multi-modality support\n")
    print("=" * 42)
    
    results = []
    results.append(("Text-only", test_text_only()))
    results.append(("Image URL", test_multimodality_image_url()))
    results.append(("Base64 Image", test_multimodality_base64()))
    results.append(("Multiple Images", test_multimodality_multiple_images()))
    
    print("\n" + "=" * 42)
    print("Test Summary:")
    print("=" * 42)
    
    all_passed = True
    for test_name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"{status}: {test_name}")
        if not passed:
            all_passed = False
    
    print("=" * 42)
    if all_passed:
        print("All tests passed! Python wrapper supports multi-modality.")
        return 0
    else:
        print("Some tests failed. Check output above for details.")
        return 1

if __name__ == "__main__":
    sys.exit(main())


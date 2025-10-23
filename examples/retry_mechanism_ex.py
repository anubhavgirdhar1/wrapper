from wrapper.utils import with_retry, ColorLogger
import random

log = ColorLogger(enable_debug=True)

# Example 1: Basic retry decorator
@with_retry(max_retries=3, initial_delay=1.0, backoff_factor=2.0, logger=log)
def flaky_api_call():
    """Simulates an API call that might fail"""
    if random.random() < 0.7:
        raise Exception("Temporary network error")
    return "Success!"

# Example 2: Retry with specific exceptions
@with_retry(
    max_retries=5,
    initial_delay=0.5,
    backoff_factor=1.5,
    retriable_exceptions=(ConnectionError, TimeoutError),
    logger=log
)
def connect_to_service():
    """Only retries on connection/timeout errors"""
    if random.random() < 0.6:
        raise ConnectionError("Failed to connect")
    return "Connected successfully"

# Example 3: Using retry with provider methods
from wrapper import Wrapper

# The retry decorator can be applied to provider methods
# For example, wrapping generate calls with retry logic:
def generate_with_retry(provider_name, model, prompt, max_retries=3):
    """Helper function to generate with automatic retry"""
    
    @with_retry(max_retries=max_retries, initial_delay=1.0, logger=log)
    def _generate():
        client = Wrapper(provider_name)
        return client.generate(model=model, user=prompt)
    
    return _generate()

if __name__ == "__main__":
    print("Example 1: Basic retry")
    try:
        result = flaky_api_call()
        print(f"Result: {result}")
    except Exception as e:
        print(f"Failed after retries: {e}")
    
    print("\n" + "="*50 + "\n")
    
    print("Example 2: Retry with specific exceptions")
    try:
        result = connect_to_service()
        print(f"Result: {result}")
    except Exception as e:
        print(f"Failed after retries: {e}")
    
    print("\n" + "="*50 + "\n")
    
    print("Example 3: Generate with retry (uncomment to test with real provider)")
    # result = generate_with_retry("openai", "gpt-3.5-turbo", "Say hello!")
    # print(f"Generated: {result}")

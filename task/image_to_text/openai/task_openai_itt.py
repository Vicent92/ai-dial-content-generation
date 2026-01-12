import base64
from pathlib import Path

from task._utils.constants import API_KEY, DIAL_CHAT_COMPLETIONS_ENDPOINT
from task._utils.model_client import DialModelClient
from task._models.role import Role
from task.image_to_text.openai.message import ContentedMessage, TxtContent, ImgContent, ImgUrl


def start() -> None:
    project_root = Path(__file__).parent.parent.parent.parent
    image_path = project_root / "dialx-banner.png"

    with open(image_path, "rb") as image_file:
        image_bytes = image_file.read()
    base64_image = base64.b64encode(image_bytes).decode('utf-8')

    # Create DialModelClient with GPT-4o model
    client = DialModelClient(
        endpoint=DIAL_CHAT_COMPLETIONS_ENDPOINT,
        deployment_name="gpt-4o",
        api_key=API_KEY
    )

    # Option 1: Analyze image using base64 encoded format
    print("\n" + "=" * 50 + " BASE64 IMAGE ANALYSIS " + "=" * 50)
    base64_data_url = f"data:image/png;base64,{base64_image}"

    message_base64 = ContentedMessage(
        role=Role.USER,
        content=[
            TxtContent(text="What do you see on this picture? Describe it in detail."),
            ImgContent(image_url=ImgUrl(url=base64_data_url))
        ]
    )

    response_base64 = client.get_completion(messages=[message_base64])
    print(f"\nAI Response (base64): {response_base64.content}")

    # Option 2: Analyze image using URL
    print("\n" + "=" * 50 + " URL IMAGE ANALYSIS " + "=" * 50)
    image_url = "https://a-z-animals.com/media/2019/11/Elephant-male-1024x535.jpg"

    message_url = ContentedMessage(
        role=Role.USER,
        content=[
            TxtContent(text="What do you see on this picture? Describe it in detail."),
            ImgContent(image_url=ImgUrl(url=image_url))
        ]
    )

    response_url = client.get_completion(messages=[message_url])
    print(f"\nAI Response (URL): {response_url.content}")


start()
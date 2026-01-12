import asyncio
from io import BytesIO
from pathlib import Path

from task._models.custom_content import Attachment, CustomContent
from task._utils.constants import API_KEY, DIAL_URL, DIAL_CHAT_COMPLETIONS_ENDPOINT
from task._utils.bucket_client import DialBucketClient
from task._utils.model_client import DialModelClient
from task._models.message import Message
from task._models.role import Role


async def _put_image() -> Attachment:
    file_name = 'dialx-banner.png'
    image_path = Path(__file__).parent.parent.parent / file_name
    mime_type_png = 'image/png'

    # Create DialBucketClient and upload image
    async with DialBucketClient(api_key=API_KEY, base_url=DIAL_URL) as client:
        # Open image file and load bytes
        with open(image_path, "rb") as image_file:
            image_bytes = BytesIO(image_file.read())

        # Upload file to bucket
        result = await client.put_file(
            name=file_name,
            mime_type=mime_type_png,
            content=image_bytes
        )

        # Return Attachment object with title, url and type
        return Attachment(
            title=file_name,
            url=result.get("url"),
            type=mime_type_png
        )


def start() -> None:
    # Create DialModelClient with GPT-4o model
    client = DialModelClient(
        endpoint=DIAL_CHAT_COMPLETIONS_ENDPOINT,
        deployment_name="gpt-4o",
        api_key=API_KEY
    )

    # Upload image using async function
    attachment = asyncio.run(_put_image())
    print(f"\nUploaded attachment: {attachment}")

    # Create message with attachment reference
    message = Message(
        role=Role.USER,
        content="What do you see on this picture?",
        custom_content=CustomContent(attachments=[attachment])
    )

    # Get completion from model
    print("\n" + "=" * 50 + " DIAL ATTACHMENT IMAGE ANALYSIS " + "=" * 50)
    response = client.get_completion(messages=[message])
    print(f"\nAI Response: {response.content}")

    # Try with different models (Claude-3-Sonnet)
    print("\n" + "=" * 50 + " CLAUDE-3-SONNET ANALYSIS " + "=" * 50)
    claude_client = DialModelClient(
        endpoint=DIAL_CHAT_COMPLETIONS_ENDPOINT,
        deployment_name="anthropic.claude-3-sonnet-20240229-v1:0",
        api_key=API_KEY
    )

    response_claude = claude_client.get_completion(messages=[message])
    print(f"\nClaude Response: {response_claude.content}")


start()

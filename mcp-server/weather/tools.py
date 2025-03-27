from typing import Any
import httpx
import logging
import sys
from mcp.server.fastmcp import FastMCP

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('mcp_server.log')
    ]
)
logger = logging.getLogger('mcp_server')

logger.info("Initializing MCP Server")
mcp = FastMCP("tools")
logger.info("FastMCP initialized with name 'tools'")


NER_API_URL = "http://127.0.0.1:5052/mask_entities"



async def make_nws_request(url: str, method: str = "GET", data: dict = None, files: dict = None, json: dict = None) -> \
        dict[str, Any] | None:
    logger.info(f"Making {method} request to {url}")
    logger.debug(f"Request params - data: {data}, files: {files}, json: {json}")

    headers = {
        "Accept": "application/json"
    }
    logger.debug(f"Request headers: {headers}")

    async with httpx.AsyncClient() as client:
        try:
            logger.info(f"Sending {method} request...")
            if method.upper() == "GET":
                response = await client.get(url, headers=headers, timeout=30.0)
                logger.debug("GET request sent")
            elif method.upper() == "POST":
                response = await client.post(
                    url,
                    headers=headers,
                    data=data,
                    files=files,
                    json=json,
                    timeout=30.0
                )
                logger.debug("POST request sent")
            else:
                logger.error(f"Unsupported HTTP method: {method}")
                raise ValueError(f"Unsupported HTTP method: {method}")

            logger.info(f"Response status code: {response.status_code}")
            response.raise_for_status()

            response_json = response.json()
            logger.debug(f"Response JSON: {response_json}")
            return response_json

        except httpx.RequestError as e:
            logger.error(f"Request error communicating with API: {e}")
            print(f"Error communicating with API: {e}")
            return None
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error: {e.response.status_code} - {e.response.text}")
            print(f"Error communicating with API: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error communicating with API: {e}", exc_info=True)
            print(f"Error communicating with API: {e}")
            return None


@mcp.tool()
async def get_ner(text: str) -> str | None:
    logger.info("get_ner tool called")
    logger.debug(f"Input text length: {len(text)}")
    logger.debug(f"Input text (truncated): {text[:100]}...")

    try:
        logger.info("Preparing NER payload")
        payload = {"input_text": text}

        logger.info(f"Sending request to NER API: {NER_API_URL}")
        response = await make_nws_request(
            url=NER_API_URL,
            method="POST",
            json=payload
        )

        if response is None:
            logger.error("NER API returned None response")
            return None

        logger.info("NER API response received successfully")
        masked_text = response['masked_text']
        logger.debug(f"Masked text length: {len(masked_text)}")
        logger.debug(f"Masked text (truncated): {masked_text[:100]}...")

        return masked_text
    except KeyError as e:
        logger.error(f"Key error in NER processing: {e}", exc_info=True)
        print(f"Key error in NER processing: {e}")
        return None
    except Exception as e:
        logger.error(f"Error in NER processing: {e}", exc_info=True)
        print(f"Error in NER processing: {e}")
        return None
@mcp.tool()
def calculate_total_amount(a: int, b: int) -> int:
    try:
        logger.info(f"Calculating total amount of {a} and {b}")
        return a + b
    except Exception as e:
        logger.error(f"Error in calculating total amount of {a} and {b}", exc_info=True)



if __name__ == "__main__":
    logger.info("Starting MCP server with stdio transport")
    try:
        mcp.run(transport='stdio')
    except Exception as e:
        logger.error(f"Error running MCP server: {e}", exc_info=True)
        sys.exit(1)
import os
import logging
import oci
from oci.generative_ai_inference import GenerativeAiInferenceClient
from app.services.secure_config import require_env, get_env
from oci.generative_ai_inference.models import (
    ChatDetails,
    CohereChatRequest,
    OnDemandServingMode,
)

# LOGGING
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# CONFIGURATION
# CONFIG_PROFILE = "GC3TEST02"
CONFIG_PROFILE = require_env("CONFIG_PROFILE")
# COMPARTMENT_ID = "ocid1.compartment.oc1..aaaaaaaa2pf2tel6ftytyrdkwaareqpcjfyfit6s62v4qdukfjiflqhlmura"
COMPARTMENT_ID = require_env("COMPARTMENT_ID")
# MODEL_ID = "ocid1.generativeaimodel.oc1.ap-hyderabad-1.amaaaaaask7dceyaaccktjkitpfn3zp3xnkg6yclc6izeahggh2hkwawfjna"
MODEL_ID = require_env("MODEL_ID")


config_path = os.path.expanduser("~/.oci/config")
logger.info(f"Loading OCI config from: {config_path}")

try:
    config = oci.config.from_file(config_path, CONFIG_PROFILE)
except Exception as e:
    logger.error(f"Failed to load OCI config: {e}")
    raise

# ENDPOINT = "https://inference.generativeai.ap-hyderabad-1.oci.oraclecloud.com"
ENDPOINT = require_env("ENDPOINT")


try:
    oci_client = GenerativeAiInferenceClient(
        config=config,
        service_endpoint=ENDPOINT
    )
    logger.info("OCI Generative AI Client initialized successfully.")
except Exception as e:
    logger.error(f"Failed to initialize OCI client: {e}")
    raise


def call_oci_chat(prompt: str) -> str:
    """
    Send a prompt to OCI Generative AI model and return generated text.
    """
    try:
        chat_request = CohereChatRequest(
            message=prompt,
            max_tokens=600,
            temperature=0.7,
            top_p=0.75,
            top_k=50,
            safety_mode="CONTEXTUAL"  
        )

        chat_detail = ChatDetails(
            chat_request=chat_request,
            serving_mode=OnDemandServingMode(model_id=MODEL_ID),
            compartment_id=COMPARTMENT_ID
        )

        response = oci_client.chat(chat_detail)

        return response.data.chat_response.text.strip()

    except Exception as e:
        logger.error(f"OCI LLM Error: {e}", exc_info=True)
        return "Error calling OCI LLM. Check logs for details."

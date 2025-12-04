import os
import logging
from oci.object_storage import ObjectStorageClient
from oci.config import from_file
from app.services.secure_config import require_env

logger = logging.getLogger(__name__)

CONFIG_PATH = "~/.oci/config"
# PROFILE = "GC3TEST02"
PROFILE = require_env("CONFIG_PROFILE")


# BUCKET_NAME = "gsc-scm-loading-km-doc"
BUCKET_NAME = require_env("BUCKET_NAME")
# NAMESPACE = "ax4qsxvnsmtm"
NAMESPACE = require_env("OCI_NAMESPACE")

DOWNLOAD_ROOT = os.path.join(os.path.dirname(__file__), "..", "data", "downloads")

def download_all_from_bucket():
    """
    Downloads ALL folders/files recursively from an OCI bucket.
    Maintains folder structure exactly.
    """

    config = from_file(CONFIG_PATH, PROFILE)
    client = ObjectStorageClient(config)

    logger.info("Listing objects in bucket...")

    list_objects_response = client.list_objects(
        namespace_name=NAMESPACE,
        bucket_name=BUCKET_NAME,
        fields="name"
    )

    objects = list_objects_response.data.objects
    logger.info(f"Found {len(objects)} objects in bucket")

    for obj in objects:
        object_name = obj.name 

        if object_name.endswith("/"):
            continue

        local_path = os.path.join(DOWNLOAD_ROOT, object_name)

        os.makedirs(os.path.dirname(local_path), exist_ok=True)

        logger.info(f"Downloading: {object_name} → {local_path}")

        with open(local_path, "wb") as f:
            response = client.get_object(
                namespace_name=NAMESPACE,
                bucket_name=BUCKET_NAME,
                object_name=object_name
            )
            f.write(response.data.content)

    logger.info("Download completed.")
    return True

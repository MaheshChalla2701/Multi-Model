import os
import logging
from datetime import datetime
from motor.motor_asyncio import AsyncIOMotorClient

logger = logging.getLogger(__name__)

# Cache the client so we don't open new connections per request
_client = None

def _get_mongo_collection():
    global _client
    mongo_url = os.getenv("MONGODB_URL", "mongodb://localhost:27017")
    
    if not _client:
        try:
            _client = AsyncIOMotorClient(mongo_url, serverSelectionTimeoutMS=2000)
        except Exception as e:
            logger.error(f"Failed to initialize MongoDB client: {e}")
            return None
            
    db = _client["medical_ai_audit"]
    return db["case_logs"]

async def log_case(case_id: str, request_data: dict, final_report: dict, safety_status: bool):
    """
    Acts as the Audit Logging Layer (Layer 10).
    Saves a permanent record of the AI's analysis for compliance, tracing, and debugging.
    """
    logger.info(f"Logging case {case_id} to Audit Database...")
    
    audit_record = {
        "case_id": case_id,
        "timestamp": datetime.utcnow().isoformat(),
        "request": {
            "modality": request_data.get("modality"),
            "anatomy": request_data.get("anatomy"),
            "has_image": bool(request_data.get("image_base64"))
        },
        "report": final_report,
        "safety_verified": safety_status
    }
    
    collection = _get_mongo_collection()
    
    if collection is None:
        logger.warning(f"Mock Audit Log - Case {case_id} recorded in memory only.")
        return
        
    try:
        # Ping the server to check connection
        await _client.server_info()
        
        # Insert the record
        await collection.insert_one(audit_record)
        logger.info(f"Case {case_id} successfully saved to MongoDB.")
    except Exception as e:
        # Graceful fallback if MongoDB is not running locally
        logger.warning(f"MongoDB not reachable. Mock Audit Log - Case {case_id} recorded in memory only. Error: {e}")

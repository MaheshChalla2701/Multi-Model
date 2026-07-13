import os
import logging
from dotenv import load_dotenv
from neo4j import GraphDatabase

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

# Mini clinical knowledge graph mapping Findings -> Diseases -> Symptoms/Guidelines
KNOWLEDGE_GRAPH = [
    {
        "finding": "Filling defect in left main pulmonary artery",
        "disease": "Pulmonary Embolism",
        "symptoms": ["Dyspnea", "Chest Pain", "Tachycardia"],
        "guideline": "Anticoagulation therapy. Consider CTPA."
    },
    {
        "finding": "Spiculated mass",
        "disease": "Invasive Ductal Carcinoma",
        "symptoms": ["Palpable breast lump", "Skin dimpling"],
        "guideline": "Core needle biopsy. Multidisciplinary tumor board review."
    },
    {
        "finding": "Extra-axial mass with homogeneous enhancement",
        "disease": "Meningioma",
        "symptoms": ["Headaches", "Seizures"],
        "guideline": "Surgical resection if symptomatic or growing. Serial imaging if small and asymptomatic."
    },
    {
        "finding": "Dorsally displaced extra-articular fracture",
        "disease": "Colles Fracture",
        "symptoms": ["Wrist pain", "Dinner fork deformity"],
        "guideline": "Closed reduction and casting. Orthopedic consult if unstable."
    }
]

def main():
    if not NEO4J_URI or not NEO4J_USERNAME or not NEO4J_PASSWORD:
        logger.error("Missing Neo4j AuraDB credentials in .env. Cannot seed.")
        return

    logger.info("Connecting to Neo4j...")
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
    
    try:
        driver.verify_connectivity()
    except Exception as e:
        logger.error(f"Failed to connect to Neo4j: {e}")
        return

    with driver.session() as session:
        # Clear existing data (destructive for demo purposes)
        logger.info("Clearing existing graph...")
        session.run("MATCH (n) DETACH DELETE n")

        logger.info("Seeding knowledge graph...")
        for item in KNOWLEDGE_GRAPH:
            # Create nodes and relationships
            query = """
            MERGE (f:Finding {name: $finding})
            MERGE (d:Disease {name: $disease})
            MERGE (g:Guideline {description: $guideline})
            MERGE (f)-[:INDICATES]->(d)
            MERGE (d)-[:HAS_GUIDELINE]->(g)
            WITH d
            UNWIND $symptoms AS sym
            MERGE (s:Symptom {name: sym})
            MERGE (d)-[:PRESENTS_WITH]->(s)
            """
            session.run(
                query,
                finding=item["finding"],
                disease=item["disease"],
                symptoms=item["symptoms"],
                guideline=item["guideline"]
            )
            
    logger.info("Seeding complete!")
    driver.close()

if __name__ == "__main__":
    main()

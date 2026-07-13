import os
import logging
from typing import Any
from neo4j import GraphDatabase

logger = logging.getLogger(__name__)

_driver = None

def _get_neo4j_driver():
    global _driver
    if _driver is None:
        uri = os.getenv("NEO4J_URI")
        user = os.getenv("NEO4J_USERNAME")
        password = os.getenv("NEO4J_PASSWORD")
        
        if uri and user and password:
            try:
                _driver = GraphDatabase.driver(uri, auth=(user, password))
                # Verify connectivity
                _driver.verify_connectivity()
                logger.info("Neo4j AuraDB driver initialized successfully.")
            except Exception as e:
                logger.error(f"Failed to initialize Neo4j driver: {e}")
                _driver = None
    return _driver


async def query_knowledge_graph(findings: list[str]) -> list[dict[str, Any]]:
    """
    Search the Neo4j Knowledge Graph for diseases and clinical guidelines 
    associated with the provided radiological findings.
    """
    if not findings:
        return []

    driver = _get_neo4j_driver()

    # MOCK MODE: If user hasn't set up Neo4j yet, return fake graph data
    if not driver:
        logger.warning("NEO4J_URI missing. Returning mock knowledge graph data.")
        return [
            {
                "disease": f"Mock Disease for {f}",
                "confidence_weight": 0.85,
                "related_symptoms": ["pain", "swelling"],
                "guideline": "Follow up with clinical correlation."
            } for f in findings[:2]
        ]

    # REAL MODE: Execute Cypher queries against the AuraDB graph
    graph_results = []
    
    # We execute a query that finds Diseases linked to the provided Findings,
    # and aggregates other symptoms that might confirm the disease.
    cypher_query = """
    UNWIND $findings AS finding_name
    MATCH (f:Finding {name: finding_name})-[:INDICATES]->(d:Disease)
    OPTIONAL MATCH (d)-[:PRESENTS_WITH]->(s:Symptom)
    OPTIONAL MATCH (d)-[:HAS_GUIDELINE]->(g:Guideline)
    RETURN d.name AS disease,
           count(f) AS evidence_count,
           collect(DISTINCT s.name) AS symptoms,
           collect(DISTINCT g.description) AS guidelines
    ORDER BY evidence_count DESC
    LIMIT 5
    """

    try:
        # Use a session to execute the read transaction
        with driver.session() as session:
            result = session.run(cypher_query, findings=findings)
            for record in result:
                graph_results.append({
                    "disease": record["disease"],
                    "evidence_count": record["evidence_count"],
                    "related_symptoms": record["symptoms"],
                    "guidelines": record["guidelines"][0] if record["guidelines"] else "No specific guideline found."
                })
    except Exception as e:
        logger.error(f"Neo4j Cypher query failed: {e}")

    return graph_results


def close_driver():
    """Ensure the Neo4j driver is closed cleanly on shutdown."""
    global _driver
    if _driver is not None:
        _driver.close()
        _driver = None

from neo4j import GraphDatabase
import json
import os
from llm_provider import get_graphrag_embedder, get_embedding_dimensions


CREATE_GRAPH_STATEMENT = """
WITH $data AS data
WITH data.agreement as a

// todo proper global id for the agreement, perhaps from filename
MERGE (agreement:Agreement {contract_id: a.contract_id})
ON CREATE SET
  agreement.name = a.agreement_name,
  agreement.effective_date = a.effective_date,
  agreement.expiration_date = a.expiration_date,
  agreement.agreement_type = a.agreement_type,
  agreement.renewal_term = a.renewal_term,
  agreement.most_favored_country = a.governing_law.most_favored_country
  //agreement.Notice_period_to_Terminate_Renewal = a.Notice_period_to_Terminate_Renewal


MERGE (gl_country:Country {name: a.governing_law.country})
MERGE (agreement)-[gbl:GOVERNED_BY_LAW]->(gl_country)
SET gbl.state = a.governing_law.state


FOREACH (party IN a.parties |
  // todo proper global id for the party
  MERGE (p:Organization {name: party.name})
  MERGE (p)-[ipt:IS_PARTY_TO]->(agreement)
  SET ipt.role = party.role
  MERGE (country_of_incorporation:Country {name: party.incorporation_country})
  MERGE (p)-[incorporated:INCORPORATED_IN]->(country_of_incorporation)
  SET incorporated.state = party.incorporation_state
)

WITH a, agreement, [clause IN a.clauses WHERE clause.exists = true] AS valid_clauses
FOREACH (clause IN valid_clauses |
  CREATE (cl:ContractClause {type: clause.clause_type})
  MERGE (agreement)-[clt:HAS_CLAUSE]->(cl)
  SET clt.type = clause.clause_type
  // ON CREATE SET c.excerpts = clause.excerpts
  FOREACH (excerpt IN clause.excerpts |
    MERGE (cl)-[:HAS_EXCERPT]->(e:Excerpt {text: excerpt})
  )
  //link clauses to a Clause Type label
  MERGE (clType:ClauseType{name: clause.clause_type})
  MERGE (cl)-[:HAS_TYPE]->(clType)
)"""

CREATE_FULL_TEXT_INDICES = [
    ("excerptTextIndex", "CREATE FULLTEXT INDEX excerptTextIndex IF NOT EXISTS FOR (e:Excerpt) ON EACH [e.text]"),
    ("agreementTypeTextIndex", "CREATE FULLTEXT INDEX agreementTypeTextIndex IF NOT EXISTS FOR (a:Agreement) ON EACH [a.agreement_type]"),
    ("clauseTypeNameTextIndex", "CREATE FULLTEXT INDEX clauseTypeNameTextIndex IF NOT EXISTS FOR (ct:ClauseType) ON EACH [ct.name]"),
    ("clauseNameTextIndex", "CREATE FULLTEXT INDEX contractClauseTypeTextIndex IF NOT EXISTS FOR (c:ContractClause) ON EACH [c.type]"),
    ("organizationNameTextIndex", "CREATE FULLTEXT INDEX organizationNameTextIndex IF NOT EXISTS FOR (o:Organization) ON EACH [o.name]"),
    ("contractIdIndex","CREATE INDEX agreementContractId IF NOT EXISTS FOR (a:Agreement) ON (a.contract_id) ")
]


def index_exists(driver, index_name):
    check_index_query = "SHOW INDEXES WHERE name = $index_name"
    result = driver.execute_query(check_index_query, {"index_name": index_name})
    return len(result.records) > 0


def create_full_text_indices(driver):
    for index_name, create_query in CREATE_FULL_TEXT_INDICES:
        if not index_exists(driver, index_name):
            print(f"Creating index: {index_name}")
            driver.execute_query(create_query)
        else:
            print(f"Index {index_name} already exists.")


def create_vector_index(driver, dimensions: int):
    # NOTE: If you switch embedding providers and need different dimensions,
    # first run  DROP INDEX excerpt_embedding  in your Neo4j console, then
    # re-run this script so the index is recreated with the correct size.
    if index_exists(driver, "excerpt_embedding"):
        print("Vector index already exists (skipping creation).")
        return
    print(f"Creating vector index with {dimensions} dimensions...")
    driver.execute_query(
        f"""
        CREATE VECTOR INDEX excerpt_embedding
            FOR (e:Excerpt) ON (e.embedding)
            OPTIONS {{indexConfig: {{`vector.dimensions`: {int(dimensions)},
                                    `vector.similarity_function`:'cosine'}}}}
        """
    )


def generate_embeddings(driver, embedder):
    """Embed all Excerpt nodes that don't have an embedding yet."""
    FETCH_QUERY = (
        "MATCH (e:Excerpt) WHERE e.text IS NOT NULL AND e.embedding IS NULL "
        "RETURN e.text AS text, id(e) AS node_id"
    )
    STORE_QUERY = (
        "MATCH (e:Excerpt) WHERE id(e) = $node_id SET e.embedding = $embedding"
    )

    records, _, _ = driver.execute_query(FETCH_QUERY)
    total = len(records)
    print(f"Generating embeddings for {total} excerpts...")

    for i, record in enumerate(records, 1):
        embedding = embedder.embed_query(record["text"])
        driver.execute_query(STORE_QUERY, {"node_id": record["node_id"], "embedding": embedding})
        if i % 10 == 0 or i == total:
            print(f"  {i}/{total} done")

    print("Embeddings complete.")


NEO4J_URI = os.getenv('NEO4J_URI', 'bolt://localhost:7687')
NEO4J_USER = os.getenv('NEO4J_USERNAME', 'neo4j')
NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD')
JSON_CONTRACT_FOLDER = './data/output/'

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))


json_contracts = [f for f in os.listdir(JSON_CONTRACT_FOLDER) if f.endswith('.json')]
contract_id = 1
for json_contract in json_contracts:
    with open(JSON_CONTRACT_FOLDER + json_contract, 'r') as file:
        json_data = json.loads(file.read())
        json_data['agreement']['contract_id'] = contract_id
        driver.execute_query(CREATE_GRAPH_STATEMENT, data=json_data)
        contract_id += 1


create_full_text_indices(driver)
embedding_dims = get_embedding_dimensions()
create_vector_index(driver, embedding_dims)
generate_embeddings(driver, get_graphrag_embedder())

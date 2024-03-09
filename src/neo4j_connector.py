from neo4j import GraphDatabase


class Neo4jConnector:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        if self.driver is not None:
            self.driver.close()
            self.driver = None

    def query(self, query, params=None):
        assert self.driver is not None, "Driver not initialized!"
        try:
            result = self.driver.execute_query(query, params)
        except Exception as e:
            print("Query failed:", e)
            return -1
        return result




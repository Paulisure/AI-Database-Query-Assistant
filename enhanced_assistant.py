from openai import OpenAI
import sqlite3
import pandas as pd
import chromadb
import datetime
import json
from typing import Dict, List, Tuple, Any

class EnhancedQueryAssistant:
    # Class-level sample queries
    SAMPLE_QUERIES = [
        "Show me the top 5 customers by total purchase amount",
        "Which genres generated the most revenue?",
        "List the most popular artists by number of tracks sold",
        "Calculate the average invoice amount by country",
        "Show sales trends by month for the year 2009",
        "Find employees who exceeded the average sales amount",
        "What's the distribution of track lengths by genre?",
        "Which customers bought classical music in 2009?",
        "Rank artists by average track price",
        "Show the most common media types by sales volume"
    ]

    def __init__(self, db_path: str, openai_api_key: str):
        """Initialize the Enhanced Query Assistant."""
        self.db_path = db_path
        self.client = OpenAI(api_key=openai_api_key)
        self.chroma_client = chromadb.PersistentClient(path="./chroma_db")
        
        # Initialize or get the collection
        try:
            self.collection = self.chroma_client.get_collection(name="query_history")
        except:
            self.collection = self.chroma_client.create_collection(name="query_history")
        
        # Cache the database schema
        self.schema = self._get_db_schema()
    
    def _get_db_schema(self) -> str:
        """Get the database schema with table and column information."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        schema_info = []
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        
        for table in tables:
            table_name = table[0]
            cursor.execute(f"PRAGMA table_info({table_name});")
            columns = cursor.fetchall()
            
            column_info = [f"{col[1]} ({col[2]})" for col in columns]
            schema_info.append(f"Table {table_name}:")
            schema_info.extend([f"  - {col}" for col in column_info])
        
        conn.close()
        return "\n".join(schema_info)
    
    def _store_query(self, natural_query: str, sql_query: str, metadata: Dict[str, Any]):
        """Store query in vector database with metadata."""
        timestamp = datetime.datetime.now().isoformat()
        self.collection.add(
            documents=[natural_query],
            metadatas=[{
                "sql_query": sql_query,
                "timestamp": timestamp,
                **metadata
            }],
            ids=[f"query_{timestamp}"]
        )
    
    def _find_similar_queries(self, query: str, n_results: int = 3) -> List[Dict[str, Any]]:
        """Find similar queries from history."""
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results
        )
        
        if not results["documents"][0]:  # No results found
            return []
        
        return [
            {
                "natural_query": doc,
                "sql_query": meta["sql_query"],
                "timestamp": meta["timestamp"]
            }
            for doc, meta in zip(results["documents"][0], results["metadatas"][0])
        ]
    
    def _generate_sql(self, natural_query: str) -> str:
        """Generate SQL from natural language query."""
        prompt = f"""Convert the following natural language query to SQL.
        Database Schema:
        {self.schema}
        
        Natural language query: {natural_query}
        
        Important: Respond with only the SQL query, no formatting or explanations."""
        
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a SQL expert. Convert natural language queries to SQL without any formatting or explanation."},
                {"role": "user", "content": prompt}
            ]
        )
        
        return response.choices[0].message.content.strip()
    
    def execute_query(self, natural_query: str) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
        """Execute natural language query and return results with similar historical queries."""
        # Find similar queries first
        similar_queries = self._find_similar_queries(natural_query)
        
        # Generate and execute SQL query
        sql_query = self._generate_sql(natural_query)
        
        conn = sqlite3.connect(self.db_path)
        result_df = pd.read_sql_query(sql_query, conn)
        conn.close()
        
        # Store the query and its results
        metadata = {
            "row_count": len(result_df),
            "execution_time": datetime.datetime.now().isoformat()
        }
        self._store_query(natural_query, sql_query, metadata)
        
        return result_df, similar_queries

# Example usage for testing
if __name__ == "__main__":
    # Replace with your OpenAI API key
    OPENAI_API_KEY = "your_api_key"
    DB_PATH = "chinook.db"
    
    assistant = EnhancedQueryAssistant(DB_PATH, OPENAI_API_KEY)
    
    # Test a sample query
    print("Testing sample query...")
    results, similar = assistant.execute_query(EnhancedQueryAssistant.SAMPLE_QUERIES[0])
    print("\nResults:")
    print(results.head())
    
    if similar:
        print("\nSimilar queries:")
        for query in similar:
            print(f"- {query['natural_query']}")


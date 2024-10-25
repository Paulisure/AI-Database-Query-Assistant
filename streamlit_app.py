import streamlit as st
import pandas as pd
from enhanced_assistant import EnhancedQueryAssistant
import plotly.express as px
from typing import Optional, Union

def create_visualization(df: pd.DataFrame) -> Optional[Union[px.bar, px.scatter]]:
    """Create a visualization based on the data."""
    if len(df) == 0:
        return None
        
    # If the dataframe has numeric columns and a potential category column
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    other_cols = df.select_dtypes(exclude=['float64', 'int64']).columns
    
    if len(numeric_cols) > 0 and len(other_cols) > 0:
        # Create a bar chart
        fig = px.bar(df, x=other_cols[0], y=numeric_cols[0],
                    title=f"{numeric_cols[0]} by {other_cols[0]}")
        return fig
    elif len(numeric_cols) >= 2:
        # Create a scatter plot
        fig = px.scatter(df, x=numeric_cols[0], y=numeric_cols[1],
                        title=f"{numeric_cols[1]} vs {numeric_cols[0]}")
        return fig
    
    return None

def main():
    st.title("AI-Powered Database Query Assistant")
    
    # Initialize session state for the assistant
    if 'assistant' not in st.session_state:
        openai_key = st.secrets["openai"]["api_key"]
        st.session_state.assistant = EnhancedQueryAssistant("chinook.db", openai_key)
    
    # Query input
    query = st.text_input("Enter your question about the music store data:",
                         placeholder="e.g., Show me the top 5 customers by total purchase amount")
    
    # Show sample queries
    with st.expander("Sample Queries"):
        for sample_query in st.session_state.assistant.SAMPLE_QUERIES:
            if st.button(sample_query, key=sample_query):  # Add unique key for each button
                query = sample_query
    
    if query:
        try:
            with st.spinner("Processing query..."):
                results, similar = st.session_state.assistant.execute_query(query)
            
            # Display results
            st.subheader("Query Results")
            st.dataframe(results)
            
            # Try to create a visualization
            fig = create_visualization(results)
            if fig:
                st.plotly_chart(fig)
            
            # Show similar queries
            if similar:
                st.subheader("Similar Previous Queries")
                for sq in similar:
                    st.write(f"- {sq['natural_query']}")
                    
        except Exception as e:
            st.error(f"Error processing query: {str(e)}")
    
    # Add information about the database
    with st.expander("About the Database"):
        st.write("""
        This demo uses the Chinook database, which represents a digital media store.
        It includes tables for:
        - Customers
        - Employees
        - Tracks
        - Albums
        - Artists
        - Invoices
        
        You can ask questions about sales, customers, music, and more!
        """)

if __name__ == "__main__":
    main()
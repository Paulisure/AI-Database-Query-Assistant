# AI-Powered Database Query Assistant

A Natural Language to SQL query system that uses OpenAI's GPT-3.5, ChromaDB for vector similarity search, and Streamlit for the user interface. This project demonstrates the integration of Large Language Models (LLMs), vector databases, and text-to-SQL conversion in a practical application.

https://github.com/user-attachments/assets/bca6fb38-120b-4673-a3a0-d50051027b97

## Features

- **Natural Language Query Processing**: Convert plain English questions into SQL queries
- **Vector Similarity Search**: Find similar historical queries using ChromaDB
- **Interactive Web Interface**: User-friendly Streamlit application
- **Automatic Visualization**: Dynamic charts based on query results
- **Sample Query Library**: Pre-built examples to demonstrate capabilities
- **Real-time Processing**: Instant query conversion and execution

## Technology Stack

- **LLM Integration**: OpenAI GPT-3.5
- **Vector Database**: ChromaDB
- **Frontend**: Streamlit
- **Data Visualization**: Plotly Express
- **Database**: SQLite (Chinook Database)
- **Language**: Python 3.8+

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/ai-database-query-assistant.git
cd ai-database-query-assistant
```

2. Create and activate a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

4. Set up your OpenAI API key:
   - Create a `.streamlit/secrets.toml` file
   - Add your OpenAI API key:
     ```toml
     [openai]
     api_key = "your-api-key-here"
     ```

## Usage

1. Start the Streamlit application:
```bash
streamlit run streamlit_app.py
```

2. Open your web browser and navigate to `http://localhost:8501`

3. Enter your question in natural language or select from sample queries

## Sample Queries

- Show me the top 5 customers by total purchase amount
- Which genres generated the most revenue?
- List the most popular artists by number of tracks sold
- Calculate the average invoice amount by country
- Show sales trends by month for the year 2009

## Project Structure

```
ai-database-query-assistant/
├── streamlit_app.py          # Streamlit interface
├── enhanced_assistant.py     # Core query assistant logic
├── requirements.txt          # Project dependencies
├── chinook.db               # SQLite database
├── .streamlit/              # Streamlit configuration
│   └── secrets.toml         # API keys (not in repo)
└── README.md                # Project documentation
```

## Database Schema

The project uses the Chinook database, which models a digital media store with tables for:
- Customers
- Employees
- Tracks
- Albums
- Artists
- Invoices
- Playlists

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Chinook Database
- OpenAI API
- Streamlit Framework
- ChromaDB Team

## Author

Paul Loupe
- GitHub: Paulisure (https://github.com/Paulisure)
- LinkedIn: (www.linkedin.com/in/paulloupe)


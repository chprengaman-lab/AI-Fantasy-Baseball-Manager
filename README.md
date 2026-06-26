# Fantasy Baseball Daily Pickup Dashboard

This project is a Streamlit dashboard for evaluating fantasy baseball daily pickup candidates.

Today, the project is only a skeleton. It does not connect to any APIs yet. The app displays placeholder player data so we can confirm the project structure works before adding real data sources.

## Project Files

- `app.py`: Main Streamlit application.
- `requirements.txt`: Python packages installed in the current environment.
- `.env.example`: Example environment variable file for future configuration.
- `.gitignore`: Files and folders Git should ignore.
- `README.md`: Project instructions and notes.

## How to Run the Project

1. Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

2. Install the required packages:

```bash
pip install -r requirements.txt
```

3. Start the Streamlit app:

```bash
streamlit run app.py
```

4. Open the local URL Streamlit prints in your terminal, usually:

```text
http://localhost:8501
```

## Current Status

- Basic Streamlit page is in place.
- Sidebar placeholder is in place.
- Placeholder dataframe is in place.
- No APIs are connected yet.

# CivicAI

**Intelligent Public Data Analysis for a More Informed Society**

CivicAI is an AI-powered agent designed to bring clarity to today's complex media landscape by providing direct access to factual public data. It transforms how citizens, journalists, and policymakers understand societal trends by making public data exploration as simple as having a conversation.

[![CivicAI Demo](https://img.shields.io/badge/Watch-Demo-red?style=for-the-badge&logo=youtube)](https://youtu.be/rEJTBHzLuAE)

## 🎯 Mission & Goals

In an era of misinformation and data overload, CivicAI exists to:
- **Democratize access** to public data for everyone, regardless of technical expertise
- **Provide neutral, fact-based insights** from trusted government sources
- **Combat misinformation** with verifiable data and transparent analysis
- **Demonstrate the power** of AI + data visualization for civic engagement

## 🚀 What CivicAI Does

**Ask natural questions → Get instant insights + visual dashboards**

### Example Queries:
- "Which states have the highest income?"
- "Show COVID cases in California"
- "Compare income levels across different states"
- "What's the wealth distribution in New York?"

### How It Works:
1. **Ask** - Type your question in plain English
2. **Process** - AI understands intent and queries trusted datasets
3. **Understand** - Get AI-powered analysis and key findings
4. **Visualize** - Explore automatic Looker Studio dashboards

## 🛠 Tech Stack

- **AI & NLP**: AI Data Agent (ADK) for natural language query processing
- **Data Platform**: Looker/Looker Studio for data modeling & visualization
- **Data Sources**: BigQuery Public Datasets (COVID, Census, Economic data)
- **Frontend**: Simple chat interface (React/Streamlit)
- **Backend**: Python-based query engine and API layer

## 📁 Project Structure
CivicAI/
├── looker/                 # LookML models and explores
├── agent/                  # AI agent backend and query processing
├── frontend/               # Chat interface and UI components
├── demo/                   # Demo assets
│   ├── script.md          # Demo presentation script
│   ├── screenshots/       # Application screenshots
│   └── video/             # Demo video files
├── docs/                   # Additional documentation
└── README.md

## 🎬 Demo & Getting Started

### Quick Demo
Check out our [Demo Video](https://youtu.be/rEJTBHzLuAE) to see CivicAI in action, answering real questions about public data and generating instant visualizations.

### Local Development
bash
# Clone the repository
git clone https://github.com/ZeroUndergroun/CivicAI.git

# Install dependencies
cd CivicAI
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Add your Looker and BigQuery credentials

# Run the application
python app.py

# Supported Data Sources

Currently integrating with:

- U.S. Census Data (Income, Demographics)

- COVID-19 Public Datasets

- Economic Indicators (BLS)

- Education Statistics

- More datasets coming soon!

# Contributing

We welcome contributions from developers, data scientists, and civic tech enthusiasts! Please see our Contributing Guidelines for details.
# License

This project is licensed under the MIT License - see the LICENSE file for details.

# Acknowledgments

    Built for the Google Hackathon submission

    Powered by Looker Studio and BigQuery Public Datasets

    Inspired by the need for transparent, accessible public data

    Good Music as well
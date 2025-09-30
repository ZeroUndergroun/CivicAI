# app.py - FIXED AND WORKING VERSION
import streamlit as st
import pandas as pd
import sys
import os
from datetime import datetime
import time
import plotly.express as px
import plotly.graph_objects as go

# Add the project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from dotenv import load_dotenv
load_dotenv()

# Import your CivicAI components
from agent.main import CivicAI, QueryConfig
from data.bigquery_client import BigQueryClient
from data.looker_client import SimpleLookerClient

def main():
    # Page configuration
    st.set_page_config(
        page_title="CivicAI + Looker",
        page_icon="🏛️", 
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for nice styling
    st.markdown("""
    <style>
    .big-font { font-size:300% !important; color: #4285F4; }
    .medium-font { font-size:150% !important; }
    </style>
    """, unsafe_allow_html=True)

    # Helper functions - DEFINED INSIDE MAIN BUT BEFORE USAGE
    def create_fallback_chart(result, query_type):
        """Create Plotly charts when Looker isn't available"""
        data = result['data']
    
        if data.empty:
            return None
    
        try:
            if query_type == "covid" and 'date' in data.columns and 'confirmed_cases' in data.columns:
                # COVID timeline chart
                fig = px.line(data, x='date', y='confirmed_cases', 
                             title="📈 COVID Cases Over Time",
                             labels={'confirmed_cases': 'Confirmed Cases', 'date': 'Date'})
                fig.update_layout(template="plotly_white")
                return fig
        
            elif query_type == "income" and 'state_name' in data.columns and 'median_income' in data.columns:
                # Income bar chart
                fig = px.bar(data, x='state_name', y='median_income',
                            title="💰 Median Income by State",
                            labels={'median_income': 'Median Income', 'state_name': 'State'})
                fig.update_layout(template="plotly_white", xaxis_tickangle=-45)
                return fig
        
            else:
                # Generic chart
                numeric_cols = data.select_dtypes(include=['number']).columns
                if len(numeric_cols) > 0:
                    fig = px.bar(data.head(10), y=numeric_cols[0], 
                                title=f"📊 {numeric_cols[0]} Distribution")
                    fig.update_layout(template="plotly_white")
                    return fig
        except Exception as e:
            st.warning(f"Could not create chart: {e}")
    
        return None

    # Header
    st.markdown('<p class="big-font">🏛️ CivicAI + Looker Studio</p>', unsafe_allow_html=True)
    st.markdown("### Google Hackathon Submission - Intelligent Public Data Analysis")
    
    # Initialize session state
    if 'civic_ai' not in st.session_state:
        try:
            bq_client = BigQueryClient()
            st.session_state.civic_ai = CivicAI(bq_client)
            st.session_state.looker_client = SimpleLookerClient()
            st.session_state.initialized = True
        except Exception as e:
            st.error(f"❌ Failed to initialize: {e}")
            st.session_state.initialized = False
            return
    
    # Sidebar
    with st.sidebar:
        st.title("🚀 Quick Demos")
        
        demo_queries = [
            "Show COVID cases in California",
            "Compare income levels across states", 
            "COVID trends in New York",
            "Which states have highest income?",
            "Show me wealth distribution data"
        ]
        
        for query in demo_queries:
            if st.button(query):
                st.session_state.demo_query = query
                st.rerun()
        
        st.markdown("---")
        st.info("""
        **How to use:**
        1. Click a demo query or type your own
        2. Click 'Analyze' to process
        3. View AI insights + Looker dashboard
        """)
    
    # Main content
    st.subheader("🔍 Ask About Public Data")
    
    # Query input
    default_query = st.session_state.get('demo_query', 'Show COVID cases in California')
    query = st.text_area(
        "Enter your question about COVID data or income statistics:",
        value=default_query,
        height=100,
        placeholder="e.g., 'Show COVID cases in California' or 'Compare income levels'",
        key="query_input"
    )
    
    # Process button
    col1, col2 = st.columns([3, 1])
    with col1:
        process_clicked = st.button("🚀 Analyze with AI + Looker", type="primary", use_container_width=True)
    with col2:
        if st.button("🔄 Clear", use_container_width=True):
            st.session_state.demo_query = ""
            st.session_state.last_result = None
            st.rerun()
    
    # Process query if needed
    if process_clicked and query:
        with st.spinner("🔄 Processing your query with AI..."):
            try:
                # Process with CivicAI
                result = st.session_state.civic_ai.process_query(query)
                
                # Add Looker URL
                if result.get('success'):
                    query_type = result['metadata'].get('intent', 'covid')
                    state = result['metadata'].get('state')
                    
                    looker_url = st.session_state.looker_client.get_embed_url(
                        query_type=query_type,
                        state=state
                    )
                    result['looker_url'] = looker_url
                    result['query_type'] = query_type
                
                st.session_state.last_result = result
                
            except Exception as e:
                st.error(f"❌ Processing failed: {e}")
    
    # Display results
    if 'last_result' in st.session_state:
        result = st.session_state.last_result
        
        if result.get('success'):
            st.success("✅ Analysis Complete!")
            
            # Two columns for results
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.subheader("🤖 AI Analysis")
                st.markdown(f"**Query Type:** {result.get('query_type', 'unknown').upper()}")
                st.write(result.get('summary', 'No summary available'))
                
                if not result['data'].empty:
                    st.subheader("📋 Data Preview")
                    st.dataframe(result['data'], use_container_width=True)
            
            with col2:
                st.subheader("📊 Data Visualizations")
                
                # Get the URL safely
                looker_url = result.get('looker_url')
                
                # Always show interactive charts (they work everywhere)
                chart = create_fallback_chart(result, result.get('query_type'))
                if chart:
                    st.plotly_chart(chart, use_container_width=True)
                
                # Optionally show Looker link
                if looker_url:
                    st.markdown("---")
                    st.markdown("### 🔗 Looker Studio Integration")
                    st.markdown(f"[📊 Open in Looker Studio]({looker_url})")
                    st.caption("Click to view advanced dashboards in Looker Studio")
        
        else:
            st.error(f"❌ {result.get('error', 'Unknown error occurred')}")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #5f6368;'>
        <p><strong>🏛️ CivicAI</strong> - Google Hackathon Submission, Ryan Torrez 2025</p>
        <p>Powered by BigQuery • OpenAI • Streamlit • Looker Studio</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
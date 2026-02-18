"""Streamlit frontend for the Intelligent Research Report Generator."""

import streamlit as st
import requests
import time
from datetime import datetime
import os

# Configuration
API_URL = os.getenv("API_URL", "http://localhost:8000")

# Page config
st.set_page_config(
    page_title="Research Report Generator",
    page_icon="🔬",
    layout="wide",
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }
    .source-card {
        background-color: #f8f9fa;
        border-left: 3px solid #4CAF50;
        padding: 0.75rem;
        margin-bottom: 0.5rem;
        border-radius: 0 5px 5px 0;
    }
    .fact-card {
        background-color: #fff3cd;
        border-left: 3px solid #ffc107;
        padding: 0.75rem;
        margin-bottom: 0.5rem;
        border-radius: 0 5px 5px 0;
    }
    .status-healthy {
        color: #28a745;
        font-weight: bold;
    }
    .status-unhealthy {
        color: #dc3545;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


def check_api_health():
    """Check if the API is healthy."""
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        if response.status_code == 200:
            return response.json()
        return None
    except requests.RequestException:
        return None


def generate_research(query: str, max_sources: int = 10, use_cache: bool = True):
    """Call the research API."""
    try:
        response = requests.post(
            f"{API_URL}/research",
            json={
                "query": query,
                "max_sources": max_sources,
                "use_cache": use_cache,
            },
            timeout=120,  # Research can take time
        )
        if response.status_code == 200:
            return response.json(), None
        return None, f"API error: {response.status_code} - {response.text}"
    except requests.RequestException as e:
        return None, f"Connection error: {str(e)}"


def get_research_history(limit: int = 10):
    """Get recent research history."""
    try:
        response = requests.get(f"{API_URL}/research/history?limit={limit}", timeout=10)
        if response.status_code == 200:
            return response.json()
        return []
    except requests.RequestException:
        return []


def get_research_by_id(request_id: int):
    """Get a specific research result."""
    try:
        response = requests.get(f"{API_URL}/research/{request_id}", timeout=10)
        if response.status_code == 200:
            return response.json(), None
        return None, f"Not found: {response.status_code}"
    except requests.RequestException as e:
        return None, str(e)


# Sidebar - Health Status
st.sidebar.markdown("---")
st.sidebar.subheader("🏥 System Health")

health = check_api_health()
if health:
    status_class = "status-healthy" if health["status"] == "healthy" else "status-unhealthy"
    st.sidebar.markdown(f"**Overall:** <span class='{status_class}'>{health['status'].upper()}</span>", unsafe_allow_html=True)
    
    for service, status in health.get("services", {}).items():
        icon = "✅" if status == "healthy" else "❌"
        st.sidebar.text(f"{icon} {service}: {status}")
else:
    st.sidebar.error("❌ API Unreachable")
    st.sidebar.text(f"Check if API is running at:\n{API_URL}")

# Main content
st.markdown('<p class="main-header">🔬 Intelligent Research Report Generator</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Generate comprehensive, well-sourced research reports using AI</p>', unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3 = st.tabs(["📝 New Research", "📚 History", "🔍 View Report"])

# Tab 1: New Research
with tab1:
    st.subheader("Generate a New Research Report")
    
    # Input form
    with st.form("research_form"):
        query = st.text_area(
            "Research Query",
            placeholder="Enter your research topic or question (minimum 10 characters)...",
            height=100,
        )
        
        col1, col2 = st.columns(2)
        with col1:
            max_sources = st.slider("Maximum Sources", min_value=1, max_value=20, value=10)
        with col2:
            use_cache = st.checkbox("Use cached results if available", value=True)
        
        submitted = st.form_submit_button("🚀 Generate Report", use_container_width=True)
    
    if submitted:
        if len(query) < 10:
            st.error("Query must be at least 10 characters long.")
        else:
            with st.spinner("Researching... This may take 30-60 seconds."):
                start_time = time.time()
                result, error = generate_research(query, max_sources, use_cache)
                elapsed = time.time() - start_time
            
            if error:
                st.error(f"❌ {error}")
            elif result:
                # Store result in session state
                st.session_state["last_result"] = result
                
                # Success message
                if result.get("cached"):
                    st.success(f"✅ Report retrieved from cache (instant)")
                else:
                    st.success(f"✅ Report generated in {result.get('processing_time_seconds', elapsed):.1f} seconds")
                
                # Metrics row
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Query Type", result.get("query_type", "N/A").replace("_", " ").title())
                with col2:
                    st.metric("Complexity", result.get("complexity", "N/A").title())
                with col3:
                    st.metric("Sources", len(result.get("sources", [])))
                with col4:
                    st.metric("Facts Extracted", len(result.get("facts", [])))
                
                st.markdown("---")
                
                # Report
                st.subheader("📄 Research Report")
                st.markdown(result.get("report", "No report generated."))
                
                # Sources
                st.markdown("---")
                st.subheader("📚 Sources")
                for i, source in enumerate(result.get("sources", []), 1):
                    with st.expander(f"{i}. {source.get('title', 'Untitled')}", expanded=False):
                        st.markdown(f"**URL:** [{source.get('url')}]({source.get('url')})")
                        if source.get("snippet"):
                            st.markdown(f"**Snippet:** {source.get('snippet')}")
                
                # Facts
                st.markdown("---")
                st.subheader("💡 Extracted Facts")
                for fact in result.get("facts", []):
                    confidence_pct = int(fact.get("confidence", 0) * 100)
                    st.markdown(f"""
                    <div class="fact-card">
                        <strong>{fact.get('claim')}</strong><br>
                        <small>Source: {fact.get('source_url')} | Confidence: {confidence_pct}%</small>
                    </div>
                    """, unsafe_allow_html=True)

# Tab 2: History
with tab2:
    st.subheader("Recent Research Requests")
    
    col1, col2 = st.columns([3, 1])
    with col2:
        if st.button("🔄 Refresh", use_container_width=True):
            st.rerun()
    
    history = get_research_history(limit=20)
    
    if not history:
        st.info("No research history found. Generate your first report!")
    else:
        for item in history:
            created_at = item.get("created_at", "")
            if created_at:
                try:
                    dt = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
                    created_at = dt.strftime("%Y-%m-%d %H:%M")
                except:
                    pass
            
            with st.expander(f"**{item.get('query', 'N/A')[:80]}...**" if len(item.get('query', '')) > 80 else f"**{item.get('query', 'N/A')}**"):
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.text(f"ID: {item.get('id')}")
                with col2:
                    st.text(f"Type: {item.get('query_type', 'N/A')}")
                with col3:
                    st.text(f"Sources: {item.get('sources_count', 0)}")
                with col4:
                    st.text(f"Time: {item.get('processing_time_seconds', 0):.1f}s")
                
                st.text(f"Created: {created_at}")
                
                if st.button(f"View Full Report", key=f"view_{item.get('id')}"):
                    st.session_state["view_id"] = item.get("id")
                    st.rerun()

# Tab 3: View Report
with tab3:
    st.subheader("View Report by ID")
    
    # Check if we have a report to view from history
    default_id = st.session_state.get("view_id", "")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        report_id = st.number_input("Report ID", min_value=1, value=int(default_id) if default_id else 1)
    with col2:
        st.write("")  # Spacing
        st.write("")
        fetch = st.button("📥 Fetch Report", use_container_width=True)
    
    if fetch or default_id:
        # Clear the view_id after using it
        if "view_id" in st.session_state:
            del st.session_state["view_id"]
        
        with st.spinner("Fetching report..."):
            result, error = get_research_by_id(int(report_id))
        
        if error:
            st.error(f"❌ {error}")
        elif result:
            # Metrics row
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Query Type", (result.get("query_type") or "N/A").replace("_", " ").title())
            with col2:
                st.metric("Complexity", (result.get("complexity") or "N/A").title())
            with col3:
                st.metric("Sources", len(result.get("sources", [])))
            with col4:
                st.metric("Facts", len(result.get("facts", [])))
            
            st.markdown("---")
            
            # Query
            st.markdown(f"**Query:** {result.get('query')}")
            
            st.markdown("---")
            
            # Report
            st.subheader("📄 Report")
            st.markdown(result.get("report", "No report content."))
            
            # Sources
            st.markdown("---")
            st.subheader("📚 Sources")
            for i, source in enumerate(result.get("sources", []), 1):
                st.markdown(f"{i}. [{source.get('title', 'Untitled')}]({source.get('url')})")


# Footer
st.markdown("---")
st.markdown(
    "<small>Built with LangGraph, FastAPI, and Streamlit | "
    "[API Docs]({}/docs)</small>".format(API_URL),
    unsafe_allow_html=True
)

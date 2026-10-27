import streamlit as st

def apply_custom_theme():
    """
    Apply clean, elegant, glassmorphic custom theme and sidebar styling.
    """
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');

        /* Global Font & Body Settings */
        html, body, [class*="css"] {
            font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
        }

        /* Hide Streamlit's Default Auto-Generated Sidebar Navigation Links */
        [data-testid="stSidebarNav"],
        ul[data-testid="stSidebarNavItems"],
        div[data-testid="stSidebarNavSeparator"] {
            display: none !important;
        }

        /* Sidebar Main Container */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0b0f19 0%, #111827 100%) !important;
            border-right: 1px solid rgba(255, 255, 255, 0.08) !important;
        }

        [data-testid="stSidebar"] > div:first-child {
            padding-top: 1.5rem;
            padding-bottom: 2rem;
        }

        /* Sidebar Headers */
        .sidebar-brand-container {
            padding: 0.5rem 0.5rem 1.25rem 0.5rem;
            margin-bottom: 1rem;
            border-bottom: 1px solid rgba(255, 255, 255, 0.08);
        }

        .sidebar-brand-title {
            font-size: 1.25rem;
            font-weight: 700;
            background: linear-gradient(135deg, #a5b4fc 0%, #818cf8 50%, #38bdf8 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin: 0;
            display: flex;
            align-items: center;
            gap: 8px;
        }

        .sidebar-brand-subtitle {
            font-size: 0.72rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: #94a3b8;
            margin-top: 4px;
            font-weight: 600;
        }

        /* Radio Navigation Group Styling */
        [data-testid="stSidebar"] div[role="radiogroup"] {
            gap: 6px;
        }

        [data-testid="stSidebar"] div[role="radiogroup"] label {
            background: rgba(255, 255, 255, 0.02) !important;
            border: 1px solid rgba(255, 255, 255, 0.05) !important;
            border-radius: 10px !important;
            padding: 10px 14px !important;
            transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
            cursor: pointer !important;
            margin-bottom: 2px !important;
        }

        [data-testid="stSidebar"] div[role="radiogroup"] label:hover {
            background: rgba(99, 102, 241, 0.12) !important;
            border-color: rgba(99, 102, 241, 0.3) !important;
            transform: translateX(4px) !important;
        }

        /* Selected Navigation Item */
        [data-testid="stSidebar"] div[role="radiogroup"] label[data-checked="true"],
        [data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) {
            background: linear-gradient(135deg, rgba(99, 102, 241, 0.22) 0%, rgba(59, 130, 246, 0.22) 100%) !important;
            border: 1px solid rgba(99, 102, 241, 0.5) !important;
            box-shadow: 0 4px 14px rgba(99, 102, 241, 0.18) !important;
        }

        [data-testid="stSidebar"] div[role="radiogroup"] label span p {
            font-size: 0.92rem !important;
            font-weight: 500 !important;
            color: #cbd5e1 !important;
            transition: color 0.2s ease !important;
        }

        [data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) span p {
            color: #ffffff !important;
            font-weight: 600 !important;
        }

        /* Clean Sidebar Project Footer Card */
        .sidebar-footer-card {
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid rgba(255, 255, 255, 0.07);
            border-radius: 12px;
            padding: 14px;
            margin-top: 1.5rem;
        }

        .sidebar-footer-title {
            font-size: 0.78rem;
            font-weight: 600;
            color: #e2e8f0;
            margin-bottom: 4px;
        }

        .sidebar-footer-desc {
            font-size: 0.72rem;
            color: #94a3b8;
            line-height: 1.4;
        }

        .sidebar-status-badge {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            background: rgba(34, 197, 94, 0.12);
            border: 1px solid rgba(34, 197, 94, 0.3);
            color: #4ade80;
            font-size: 0.68rem;
            font-weight: 600;
            padding: 2px 8px;
            border-radius: 20px;
            margin-top: 8px;
        }

        .status-dot {
            width: 6px;
            height: 6px;
            background-color: #22c55e;
            border-radius: 50%;
            box-shadow: 0 0 8px #22c55e;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

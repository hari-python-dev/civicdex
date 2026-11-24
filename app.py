
import streamlit as st
import joblib
from pathlib import Path
import uuid
from datetime import datetime
from database import (
    initialize_db,
    register_complaint,
    get_complaint_by_id,
    get_complaints_by_department,
    update_complaint_status,
    create_citizen_account,
    verify_citizen_login,
    get_complaints_by_email,
    create_staff_account,
    verify_staff_login,
    get_all_department_names
)
from complaint_improver import improve_complaint

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="CivicDex | Public Grievance Portal",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

BASE_DIR = Path(__file__).parent

# Create the SQLite database/table if they do not exist
initialize_db()
# ---------------- CITIZEN LOGIN SESSION ----------------

if "logged_in_citizen" not in st.session_state:
    st.session_state.logged_in_citizen = None

if "logged_in_staff" not in st.session_state:
    st.session_state.logged_in_staff = None

if "login_mode" not in st.session_state:
    st.session_state.login_mode = "Login"

# ---------------- LOAD ML MODEL ----------------
@st.cache_resource
def load_model():
    model = joblib.load(BASE_DIR / "models/civicdex_raw_model.pkl")
    vectorizer = joblib.load(BASE_DIR / "models/civicdex_raw_vectorizer.pkl")
    return model, vectorizer

try:
    model, vectorizer = load_model()
    model_loaded = True
except Exception:
    model_loaded = False


department_mapping = {
    "certificates": "Revenue Department",
    "drainage": "Drainage Department",
    "electricity": "Electricity Department",
    "public_health": "Public Health Department",
    "roads": "Roads Department",
    "sanitation": "Sanitation Department",
    "street_lighting": "Electricity Department",
    "transport": "Transport Department",
    "water_supply": "Water Supply Department",
    "welfare": "Welfare Department"
}


def predict_complaint(complaint, selected_category):
    if selected_category == "Welfare":
        return {
            "category": "welfare",
            "department": "Welfare Department",
            "confidence": None,
            "routing": "Direct Welfare Routing"
        }

    features = vectorizer.transform([complaint])
    prediction = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]

    return {
        "category": prediction,
        "department": department_mapping[prediction],
        "confidence": round(float(max(probabilities)) * 100, 2),
        "routing": "ML Classification"
    }


# ---------------- ROYAL DESIGN CSS ----------------
st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

.stApp {
    background: #eef2f8;
    font-family: 'Inter', sans-serif;
    color: #102b50;
}

.block-container {
    max-width: 100%;
    width: 100%;
    padding: 1.2rem 2%;
}

header, footer {
    visibility: hidden;
}

h1, h2, h3 {
    color: #092b55 !important;
    font-weight: 800 !important;
}

p, label {
    color: #294565;
}

.hero-heading {
    font-size: 38px;
    font-weight: 800;
    color: #092b55;
}

.section-heading {
    font-size: 25px;
    font-weight: 800;
    color: #123763;
    margin: 20px 0 8px;
    border-left: 5px solid #c6a04a;
    padding-left: 12px;
}

.section-description {
    color: #627895;
    font-size: 15px;
    margin-bottom: 18px;
}

.form-card {
    background: #ffffff;
    border: 1px solid #d8e0ed;
    border-top: 4px solid #c6a04a;
    border-radius: 16px;
    padding: 25px;
    box-shadow: 0 8px 24px rgba(10, 36, 70, 0.08);
}

div[data-testid="stTextInput"] label,
div[data-testid="stSelectbox"] label,
div[data-testid="stTextArea"] label {
    color: #173b66 !important;
    font-size: 16px !important;
    font-weight: 700 !important;
}

div[data-testid="stTextInput"] input,
div[data-testid="stTextArea"] textarea {
    background: #ffffff !important;
    color: #173b66 !important;
    border: 1.5px solid #bdcce0 !important;
    border-radius: 10px !important;
    font-size: 16px !important;
}

div[data-testid="stSelectbox"] div[data-baseweb="select"] {
    background: #ffffff !important;
    border-radius: 10px !important;
}

.stButton button,
.stFormSubmitButton button {
    background: linear-gradient(110deg, #153e70, #071e3d) !important;
    color: #ffda7b !important;
    border: 2px solid #c6a04a !important;
    border-radius: 11px !important;
    min-height: 58px !important;
    font-size: 18px !important;
    font-weight: 800 !important;
    width: 100% !important;
    box-shadow: 0 5px 15px rgba(8, 41, 79, 0.15);
}

.stButton button:hover,
.stFormSubmitButton button:hover {
    background: #d5ad50 !important;
    color: #092b55 !important;
}

.icon-badge {
    width: 42px;
    height: 42px;
    min-width: 42px;
    border-radius: 50%;
    background: #e5edf8;
    color: #123763;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 21px;
    margin: 28px auto 0;
    border: 1px solid #c6a04a;
}

div[data-testid="stAlert"] {
    border-radius: 12px;
}

.footer-text {
    text-align: center;
    color: #71839a;
    font-size: 13px;
    padding: 22px 0;
}

""", unsafe_allow_html=True)


# ---------------- HEADER / BANNER ----------------
banner_path = BASE_DIR / "assets" / "civicdex_banner.png"

if banner_path.exists():
    st.image(
        str(banner_path),
        use_container_width=True
    )
else:
    st.warning("Banner image not found.")

st.markdown("""
<div class="hero-heading">Citizen Grievance Portal</div>
<div class="hero-subtitle">
Register your complaint. Get it routed to the appropriate department.
Help build cleaner and better communities.
</div>
""", unsafe_allow_html=True)

st.write("")


# ---------------- FEATURE CARDS ----------------
features = [
    ("📝", "Register Grievance", "Simple and easy complaint registration"),
    ("🤖", "Smart Classification", "AI-assisted department routing"),
    ("📍", "Track Progress", "Follow your complaint status"),
    ("✉️", "Get Notifications", "Updates through email"),
    ("🌱", "Clean Communities", "Work towards a better tomorrow")
]

columns = st.columns(5)

for column, feature in zip(columns, features):
    icon, title, caption = feature
    with column:
        st.markdown(f"""
        <div class="feature-card">
            <div class="feature-icon">{icon}</div>
            <div class="feature-title">{title}</div>
            <div class="feature-caption">{caption}</div>
        </div>
        """, unsafe_allow_html=True)

# ---------------- PAGE NAVIGATION ----------------
st.write("")
st.divider()

def logout():
    st.session_state.logged_in_citizen = None
    st.session_state.logged_in_staff = None
    st.session_state.current_page = "Citizen Login"
    st.session_state.login_mode = "Login"

page = "Citizen Login"

if "current_page" not in st.session_state:
    st.session_state.current_page = "Citizen Login"

def go_to_page(target_page):
    st.session_state.current_page = target_page


if st.session_state.logged_in_staff is not None:

    staff = st.session_state.logged_in_staff

    nav_col, user_col, logout_col = st.columns([4, 2, 1])

    with nav_col:

        if staff["role"] == "admin":
            page = st.radio(
                "Admin Services",
                ["Admin Dashboard"],
                key="current_page",
                horizontal=True
            )

        elif staff["role"] == "department":
            page = st.radio(
                "Department Services",
                ["Department Dashboard"],
                key="current_page",
                horizontal=True
            )

    with user_col:
        st.write("")
        st.markdown(
            f"👤 **{staff['name']}**",
            unsafe_allow_html=True
        )

    with logout_col:
        st.write("")
        st.button("🚪 Logout", on_click=logout)

elif st.session_state.logged_in_citizen is not None:

    citizen = st.session_state.logged_in_citizen

    nav_col, user_col, logout_col = st.columns([4, 2, 1])

    with nav_col:
        page = st.radio(
            "Choose a Service",
            [
                "Citizen Home",
                "Register Complaint",
                "My Complaints",
                "Track Complaint"
            ],
            key="current_page",
            horizontal=True
        )

    with user_col:
        st.write("")
        st.markdown(
            f"👤 **{citizen['name']}**",
            unsafe_allow_html=True
        )

    with logout_col:
        st.write("")
        st.button("🚪 Logout", on_click=logout)
    
    # Un existing login code inga irukkanum
# ---------------- DEPARTMENT DASHBOARD ----------------
# ---------------- ADMIN DASHBOARD ----------------

if page == "Admin Dashboard":

    staff = st.session_state.logged_in_staff

    if not staff or staff.get("role") != "admin":
        st.error("Access denied. Admin login required.")
        st.stop()

    st.markdown(
        '<div class="section-heading">🏛️ CivicDex Admin Dashboard</div>',
        unsafe_allow_html=True
    )

    st.subheader("Department Overview")
     

    departments = [
    "Sanitation Department",
    "Water Supply Department",
    "Roads Department",
    "Drainage Department",
    "Electricity Department",
    "Revenue Department",
    "Public Health Department",
    "Transport Department",
    "Welfare Department"
    ]

    cols = st.columns(3)

    for index, department in enumerate(departments):

        complaints = get_complaints_by_department(department)

        with cols[index % 3]:

            st.metric(
                label=f"🏢 {department}",
                value=len(complaints)
            )

    st.stop()
# ---------------- MY COMPLAINTS ----------------

if page == "My Complaints":

    citizen = st.session_state.logged_in_citizen

    st.markdown("## 📋 My Complaints")
    st.write(f"Welcome, {citizen['name']}!")

    complaints = get_complaints_by_email(
    st.session_state.logged_in_citizen["email"]
    )

    if complaints:

        for complaint in complaints:

            with st.expander(
                f"🆔 {complaint['complaint_id']} - {complaint['category']}"
            ):

                st.write("**Department:**", complaint["department"])
                st.write("**Status:**", complaint["status"])
                st.write("**District:**", complaint["district"])
                st.write("**Date:**", complaint["created_at"])

    else:
        st.info("You have not registered any complaints yet.")

    st.stop()

# ---------------- CITIZEN LOGIN / REGISTRATION ----------------

if page == "Citizen Login":

    st.markdown(
        '<div class="section-heading">🔐 CivicDex Account</div>',
        unsafe_allow_html=True
    )

    st.write("Login to access your CivicDex account.")

    login_mode = st.radio(
        "Choose an option",
        ["Login", "Create Account", "Staff Login"],
        horizontal=True,
        key="login_mode"
    )

    if login_mode == "Create Account":

        st.markdown("### 📝 Create New Account")

        with st.form("citizen_register_form"):

            new_name = st.text_input("Full Name")
            new_email = st.text_input("Email Address")
            new_password = st.text_input(
                "Create Password", type="password"
            )
            confirm_password = st.text_input(
                "Confirm Password", type="password"
            )

            register_button = st.form_submit_button("Create Account")

        if register_button:

            if not new_name.strip() or not new_email.strip():
                st.error("Please enter your name and email.")

            elif "@" not in new_email or "." not in new_email.split("@")[-1]:
                st.error("Please enter a valid email address.")

            elif len(new_password) < 8:
                st.error("Password must contain at least 8 characters.")

            elif new_password != confirm_password:
                st.error("Passwords do not match.")

            else:
                created = create_citizen_account(
                    new_name, new_email, new_password
                )

                if created:
                    st.success("Account created successfully! Please login.")
                else:
                    st.error("This email is already registered.")

    elif login_mode == "Staff Login":

        st.markdown("### 🏛️ Admin / Department Officer Login")

        with st.form("staff_login_form"):

            staff_email = st.text_input("Staff Email Address")
            staff_password = st.text_input(
                "Password", type="password"
            )

            staff_login_button = st.form_submit_button("Staff Login")

        if staff_login_button:

            if not staff_email.strip() or not staff_password:
                st.error("Please enter your email and password.")

            else:
                staff = verify_staff_login(
                    staff_email, staff_password
                )

                if staff:
                    st.session_state.logged_in_staff = staff
                    st.session_state.logged_in_citizen = None

                    if staff["role"] == "admin":
                        st.session_state.current_page = "Admin Dashboard"
                    elif staff["role"] == "department":
                        st.session_state.current_page = "Department Dashboard"
                    else:
                        st.error("Invalid staff role.")
                        st.stop()

                    st.rerun()

                else:
                    st.error("Invalid staff email or password.")

    else:

        st.markdown("### 🔑 Citizen Login")

        with st.form("citizen_login_form"):

            login_email = st.text_input("Email Address")
            login_password = st.text_input(
                "Password", type="password"
            )

            login_button = st.form_submit_button("Login")

        if login_button:

            if not login_email.strip() or not login_password:
                st.error("Please enter your email and password.")

            else:
                citizen = verify_citizen_login(
                    login_email, login_password
                )

                if citizen:
                    st.session_state.logged_in_citizen = citizen
                    st.session_state.logged_in_staff = None
                    st.session_state.current_page = "Citizen Home"
                    st.rerun()

                else:
                    st.error("Invalid email or password.")

    st.stop()

if page == "Citizen Home":

        st.html("""
        <style>
        .civic-welcome {
            background: linear-gradient(120deg, #081A35, #163967);
            border: 2px solid #D4AF37;
            border-radius: 20px;
            padding: 40px 35px;
            margin-bottom: 30px;
            box-shadow: 0 8px 25px rgba(0,0,0,0.25);
            text-align: center;
        }

        .civic-logo {
            font-size: 55px;
            margin-bottom: 10px;
        }

        .civic-title {
            color: #FFD966 !important;
            font-size: 52px !important;
            font-weight: 900 !important;
            letter-spacing: 2px;
            margin: 15px 0;
        }

        .civic-welcome h2 {
            color: #FFFFFF !important;
            font-size: 27px !important;
        }

        .civic-welcome p {
            color: #E5EAF2 !important;
            font-size: 17px !important;
        }

        .civic-tagline {
            color: #FFD966 !important;
            font-size: 19px !important;
            font-weight: 700;
        }
        </style>

        <div class="civic-welcome">
            <div class="civic-logo">🏛️</div>
            <div class="civic-title">WELCOME TO CIVICDEX</div>
            <h2>Your Voice. Our Responsibility.</h2>
            <p>A Smart Digital Public Grievance Redressal Portal</p>
            <div class="civic-tagline">
                TRANSPARENT GOVERNANCE • SMARTER SERVICES
            </div>
        </div>
        """)

        st.markdown("### Citizen Services")

        col1, col2 = st.columns(2)

        with col1:
            with st.container(border=True):
                st.markdown("### 📝 Register Complaint")
                st.write("Submit your public grievance easily.")

                st.button(
                    "Register Now",
                    key="home_register",
                    use_container_width=True,
                    on_click=go_to_page,
                    args=("Register Complaint",)
                )

            with st.container(border=True):
                st.markdown("### 📍 Track Complaint")
                st.write("Check the current status of your complaint.")

                st.button(
                    "Track Now",
                    key="home_track",
                    use_container_width=True,
                    on_click=go_to_page,
                    args=("Track Complaint",)
                )

        with col2:
            with st.container(border=True):
                st.markdown("### 📂 My Complaints")
                st.write("View your previously registered complaints.")

                st.button(
                    "View Complaints",
                    key="home_my",
                    use_container_width=True
                )

            with st.container(border=True):
                st.markdown("### ❓ Help & FAQ")
                st.write("Get assistance and learn how CivicDex works.")

                st.button(
                    "Get Help",
                    key="home_help",
                    use_container_width=True
                )

        st.divider()

        st.markdown("### 🕒 Recent Complaints")
        st.info("Your recent complaints will appear here.")

        st.stop()
    # ---------------- TRACK COMPLAINT ----------------


if page == "Track Complaint":

    st.title("📍 Track Your Complaint")
    st.write(
        "Enter your CivicDex Complaint ID to check "
        "your complaint progress."
    )

    complaint_id_input = st.text_input(
        "Complaint ID",
        placeholder="Example: CDX-20261021-XXXXXXXX"
    )

    if st.button("🔍 Track Complaint", key="track_search"):

        if complaint_id_input.strip():

            complaint_data = get_complaint_by_id(
                complaint_id_input.strip()
            )

            if complaint_data:

                st.success("Complaint tracking details found!")

                st.markdown("## 📋 Complaint Information")

                col1, col2 = st.columns(2)

                with col1:
                    st.write("**Complaint ID**")
                    st.code(complaint_data["complaint_id"])

                    st.write("**Category**")
                    st.write(
                        complaint_data["category"].replace("_", " ").title()
                    )

                with col2:
                    st.write("**Assigned Department**")
                    st.write(complaint_data["department"])

                    st.write("**Current Status**")
                    st.write(complaint_data["status"])

                # ---------------- STATUS TIMELINE ----------------

                st.markdown("---")
                st.markdown("## 🕒 Complaint Status Timeline")

                current_status = str(
                    complaint_data["status"]
                ).strip().lower()

                stages = [
                    "Complaint Registered",
                    "AI Classification Completed",
                    "Forwarded to Department",
                    "Work In Progress",
                    "Complaint Resolved"
                ]

                # Identify current stage

                if current_status in ["resolved", "closed"]:
                    current_step = 4

                elif current_status in ["in progress", "processing"]:
                    current_step = 3

                elif current_status in ["forwarded", "assigned"]:
                    current_step = 2

                else:
                    # Registered: AI classification is already completed.
                    # Waiting for department forwarding.
                    current_step = 2

                st.progress(
                    (current_step + 1) / len(stages)
                )

                for i, stage in enumerate(stages):

                    if i < current_step:
                        st.success(
                            "✅ " + stage + " — Completed"
                        )

                    elif i == current_step:
                        st.info(
                            "🔵 " + stage + " — Current Stage"
                        )

                    else:
                        st.write(
                            "⚪ " + stage + " — Pending"
                        )

                if current_status == "registered":
                    st.warning(
                        "Your complaint is registered and "
                        "AI classification is completed. "
                        "It is awaiting department forwarding."
                    )

                elif current_status == "resolved":
                    st.success(
                        "Your complaint has been marked as resolved."
                    )

                st.caption(
                    "Complaint progress is displayed using "
                    "the latest status available in CivicDex."
                )

            else:
                st.error(
                    "Complaint ID not found. Please check your ID."
                )

        else:
            st.warning("Please enter your Complaint ID.")

    st.stop()




# ---------------- DEPARTMENT DASHBOARD ----------------

if page == "Department Dashboard":

    staff = st.session_state.logged_in_staff

    # Security check
    if not staff or staff.get("role") != "department":
        st.error("Access denied. Department officer login required.")
        st.stop()

    assigned_department = staff.get("department")

    if not assigned_department:
        st.error("No department assigned to this officer.")
        st.stop()

    st.markdown(
        '<div class="section-heading">🏛️ Department Dashboard</div>',
        unsafe_allow_html=True
    )

    st.markdown(f"### 🏢 {assigned_department}")
    st.write(f"Welcome, {staff['name']}!")

    complaints = get_complaints_by_department(assigned_department)

    st.metric("Total Complaints", len(complaints))

    if complaints:

        for item in complaints:

            with st.expander(
                f'{item["complaint_id"]} | '
                f'{item["category"].replace("_", " ").title()}'
            ):

                st.write("**Citizen:**", item["citizen_name"])
                st.write("**District:**", item["district"])
                st.write("**Email:**", item["email"])
                st.write("**Complaint:**", item["complaint_text"])
                st.write("**Current Status:**", item["status"])

                status_options = [
                    "Registered",
                    "Forwarded",
                    "In Progress",
                    "Resolved"
                ]

                current_status = item["status"]

                if current_status not in status_options:
                    status_options.append(current_status)

                new_status = st.selectbox(
                    "Update Complaint Status",
                    status_options,
                    index=status_options.index(current_status),
                    key=f"status_{item['complaint_id']}"
                )

                if st.button(
                    "Update Status",
                    key=f"update_{item['complaint_id']}"
                ):

                    updated = update_complaint_status(
                        item["complaint_id"],
                        new_status
                    )

                    if updated:
                        st.success("Complaint status updated successfully!")
                        st.rerun()
                    else:
                        st.error("Unable to update complaint status.")

    else:
        st.info("No complaints found for your department.")

    st.stop()


if page == "Register Complaint":

    citizen = st.session_state.logged_in_citizen
    # ---------------- CITIZEN FORM ----------------

   

    left_panel, right_panel = st.columns([2, 1], gap="large")

    with left_panel:
        with st.container(border=True):

            st.markdown("### 👤 Citizen Information")
            st.caption("Enter your details to register a public grievance.")

            with st.form("grievance_form"):

                col1, col2 = st.columns(2)

                with col1:
                    name = st.text_input(
                        "👤 Full Name",
                        value=citizen["name"],
                        disabled=True
                    )

                with col2:
                    email = st.text_input(
                        "✉️ Email Address",
                        value=citizen["email"],
                        disabled=True
                    )

                district_list = [
                    "Select District", "Ariyalur", "Chengalpattu", "Chennai",
                    "Coimbatore", "Cuddalore", "Dharmapuri", "Dindigul",
                    "Erode", "Kallakurichi", "Kancheepuram", "Kanniyakumari",
                    "Karur", "Krishnagiri", "Madurai", "Mayiladuthurai",
                    "Nagapattinam", "Namakkal", "Nilgiris", "Perambalur",
                    "Pudukkottai", "Ramanathapuram", "Ranipet", "Salem",
                    "Sivaganga", "Tenkasi", "Thanjavur", "Theni",
                    "Thoothukudi", "Tiruchirappalli", "Tirunelveli",
                    "Tirupathur", "Tiruppur", "Tiruvallur",
                    "Tiruvannamalai", "Tiruvarur", "Vellore",
                    "Viluppuram", "Virudhunagar"
                ]

                district = st.selectbox("📍 District", district_list)

                st.markdown("### 📝 Complaint Information")

                col3, col4 = st.columns(2)

                with col3:
                    language = st.selectbox(
                        "🌐 Complaint Language",
                        ["Tamil", "English", "Tanglish"]
                    )

                with col4:
                    selected_category = st.selectbox(
                        "🏛️ Complaint Department / Type",
                        [
                            "Let AI identify", "Welfare", "Roads",
                            "Water Supply", "Sanitation", "Drainage",
                            "Electricity", "Street Lighting",
                            "Certificates", "Public Health", "Transport"
                        ]
                    )

                complaint = st.text_area(
                    "💬 Describe Your Complaint",
                    placeholder="Example: road la romba pothole iruku...",
                    height=150
                )

                st.caption(
                    "Please provide clear details. Do not include passwords "
                    "or sensitive personal information."
                )

                submitted = st.form_submit_button(
                    "🔍 ANALYZE MY COMPLAINT",
                    use_container_width=True
                )

                # AI COMPLAINT IMPROVEMENT

        if st.button("✨ Improve My Complaint"):

            if not complaint.strip():
                st.warning("Please enter your complaint first.")

            else:
                improved_text = improve_complaint(complaint)

                st.session_state.improved_complaint = improved_text
                st.success("Complaint improved successfully!")

        if "improved_complaint" in st.session_state:

            st.markdown("### ✨ Improved Complaint Preview")

            st.markdown(
                f"""
                <div style="
                    background-color: #071E3D;
                    color: #FFFFFF;
                    padding: 18px;
                    border: 2px solid #C6A04A;
                    border-radius: 12px;
                    font-size: 16px;
                    line-height: 1.7;
                ">
                    {st.session_state.improved_complaint}
                </div>
                """,
                unsafe_allow_html=True
            )

    
    with right_panel:

        st.html("""
        <div style="
            background:linear-gradient(150deg,#071e3d,#163e70);
            border:2px solid #c6a04a;
            border-radius:18px;
            padding:25px;
            color:#FFFFFF;
            margin-top:15px;
        ">

            <div style="font-size:48px;text-align:center;">🏛️</div>

            <h2 style="
                color:#FFD700 !important;
                -webkit-text-fill-color:#FFD700 !important;
                font-size:30px !important;
                font-weight:800 !important;
                text-align:center;
            ">
                TAMIL NADU
            </h2>

            <h3 style="
                color:#FFE5A3 !important;
                -webkit-text-fill-color:#FFE5A3 !important;
                font-size:23px !important;
                font-weight:700 !important;
                text-align:center;
            ">
                Public Grievance Portal
            </h3>

            <hr style="border-color:#c6a04a;">

            <h3 style="
                color:#FFD700 !important;
                -webkit-text-fill-color:#FFD700 !important;
            ">
                ✨ Key Benefits
            </h3>

            <p style="color:#FFFFFF !important; -webkit-text-fill-color:#FFFFFF !important;">✓ AI-powered complaint classification</p>
            <p style="color:#FFFFFF !important; -webkit-text-fill-color:#FFFFFF !important;">✓ Tamil, English & Tanglish support</p>
            <p style="color:#FFFFFF !important; -webkit-text-fill-color:#FFFFFF !important;">✓ Direct department routing</p>
            <p style="color:#FFFFFF !important; -webkit-text-fill-color:#FFFFFF !important;">✓ Transparent complaint tracking</p>
            <p style="color:#FFFFFF !important; -webkit-text-fill-color:#FFFFFF !important;">✓ Digital grievance management</p>

            <hr style="border-color:#c6a04a;">

            <h3 style="
                color:#FFD700 !important;
                -webkit-text-fill-color:#FFD700 !important;
            ">
                📢 Your Voice Matters
            </h3>

            <p style="
                color:#FFFFFF !important;
                -webkit-text-fill-color:#FFFFFF !important;
                font-size:16px;
                line-height:1.6;
            ">
                Every complaint is an opportunity to improve
                public services and build a better Tamil Nadu.
            </p>

        </div>
        """)



    # ---------------- PREDICTION RESULT ----------------
    if submitted:

        if not name.strip() or not email.strip() or not complaint.strip():
            st.error("Please fill in your name, email and complaint.")

        elif "@" not in email or "." not in email.split("@")[-1]:
            st.error("Please enter a valid email address.")

        elif district == "Select District":
            st.error("Please select your district.")

        elif not model_loaded:
            st.error("ML model could not be loaded. Please check the model files.")

        else:
            if selected_category == "Let AI identify":
                category_for_model = None
            else:
                category_for_model = selected_category

            result = predict_complaint(complaint, category_for_model)

            st.write("")
            st.markdown('<div class="section-heading">🤖 AI Analysis Result</div>',
                        unsafe_allow_html=True)

            col5, col6 = st.columns(2)

            with col5:
                st.metric("Predicted Category", result["category"].replace("_", " ").title())

            with col6:
                st.metric("Assigned Department", result["department"])

            if result["confidence"] is not None:
                st.metric("Model Confidence", f'{result["confidence"]}%')
                st.caption(
                    "Confidence is the model's prediction score, not a guarantee of correctness."
                )

            st.success(f'Routing Method: {result["routing"]}')

            # Generate a unique CivicDex complaint reference
            complaint_id = (
                f"CDX-{datetime.now().strftime('%Y%m%d')}-"
                f"{uuid.uuid4().hex[:8].upper()}"
            )

            try:
                register_complaint(
                    complaint_id=complaint_id,
                    citizen_name=name.strip(),
                    email=citizen["email"].strip(),
                    district=district,
                    language=language,
                    complaint_text=complaint.strip(),
                    category=result["category"],
                    department=result["department"],
                    confidence=result["confidence"]
                )

                st.success("Your complaint has been registered successfully!")
                st.markdown("### Your Complaint ID")
                st.code(complaint_id, language=None)
                st.info(
                    "Please save this Complaint ID to track your grievance. "
                    "Your complaint is saved in the CivicDex prototype database; "
                    "it has not yet been sent to a government department."
                )

            except Exception as error:
                st.error(f"Could not register your complaint: {error}")


# ---------------- FOOTER ----------------
st.divider()

st.markdown("""
<div class="footer-text">
CivicDex | AI-Powered Public Grievance Management System<br>
Prototype • Cleaner Cities • Better Tomorrow
</div>
""", unsafe_allow_html=True)

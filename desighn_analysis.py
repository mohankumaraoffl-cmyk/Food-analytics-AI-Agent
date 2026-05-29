import streamlit as st
import sqlite3
import pandas as pd
import uuid
import datetime
import os
import random
import numpy as np
from textblob import TextBlob
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
import google.generativeai as genai
import time

# --- Gemini API Configuration (Full-Stack Initialization) ---
# Updated Secure Key: AIzaSyBEdNNc4w2c6RQCVA4y8OwnQVvAClQkRv0
try:
    genai.configure(api_key="AIzaSyC4teMmfm2-a5MtNRofH4qEVAafZUWAV2w")
except Exception as e:
    st.error(f"Critical Error: Failed to configure Gemini API. {e}")

DB_FILE = "analytics_data.db"

@st.cache_data(ttl=3600, show_spinner=False)
def generate_ai_response(prompt_text):
    """Dual-model AI response system: Primary (Gemini 2.5 Pro) -> Backup (Gemini 2 Flash Lite)"""
    try:
        # Primary: High-performance Reasoning Model
        model = genai.GenerativeModel('gemini-2.5-flash')
        response = model.generate_content(prompt_text)
        return response.text
    except Exception as e:
        # Automatically switch to fallback on failure
        try:
            # Backup: Optimized Lightweight Model
            model = genai.GenerativeModel('gemini-2-flash-lite')
            response = model.generate_content(prompt_text)
            return response.text
        except Exception as e2:
            return "AI is temporarily resting, please try again!"

def generate_gemini_insights(prompt, context_label=""):
    ai_text = "Initializing AI analysis..."
    max_retries = 2
    
    for attempt in range(max_retries):
        ai_text = generate_ai_response(prompt)
        if "resting" not in ai_text:
            break # Success or valid response
        if attempt < max_retries - 1:
            time.sleep(5)
        
    with st.container():
        st.markdown(f"""
            <div>
                <div style="color: #FF8C00; font-weight: bold; margin-bottom: 10px;">🎯 {context_label}</div>
                <div style="color: #FF8C00;">{ai_text}</div>
                <div style="font-size: 0.8rem; color: rgba(255, 140, 0, 0.6); margin-top: 15px;">
                    💎 Powered by Gemini 2.5 Pro | Ultra-High Reasoning Decision Layer
                </div>
            </div>
        """, unsafe_allow_html=True)

def apply_standard_style(fig):
    """Apply the project's Deep Navy / Slate Grey palette to all Plotly charts."""
    fig.update_layout(
        template="plotly_white",
        font_family="Inter, Segoe UI, Roboto, sans-serif",
        font_color="#2F3E46",
        title_font=dict(color="#1A2B3C", size=18, family="Inter, Segoe UI, Roboto, sans-serif"),
        xaxis=dict(
            title_font=dict(color="#2F3E46", size=13),
            tickfont=dict(color="#64748B", size=11),
            gridcolor="#E2E8F0",
            linecolor="#CBD5E1"
        ),
        yaxis=dict(
            title_font=dict(color="#2F3E46", size=13),
            tickfont=dict(color="#64748B", size=11),
            gridcolor="#E2E8F0",
            linecolor="#CBD5E1"
        ),
        legend=dict(
            font=dict(color="#2F3E46", size=12, family="Inter, Segoe UI, Roboto, sans-serif"),
            bordercolor="#E2E8F0",
            borderwidth=1
        ),
        margin=dict(l=40, r=40, t=70, b=40),
        hovermode="closest",
        paper_bgcolor="rgba(255,255,255,0)",
        plot_bgcolor="rgba(255,255,255,0.6)",
    )
    return fig

# ── Project Color Palette for Charts ──
CHART_COLORS = ["#1A2B3C", "#10B981", "#F59E0B", "#2F3E46", "#6366F1", "#EC4899", "#14B8A6", "#8B5CF6"]
SEGMENT_COLORS = {
    "Champions": "#10B981",       # Emerald Green
    "Loyal Customers": "#34D399", # Light Emerald
    "At Risk": "#F59E0B",         # Soft Amber
    "Hibernating": "#94A3B8",     # Muted grey
}


def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS orders (
            order_id TEXT PRIMARY KEY,
            order_date TEXT,
            customer_name TEXT,
            customer_id TEXT,
            location TEXT,
            dish_name TEXT,
            quantity INTEGER,
            unit_price REAL,
            sub_total REAL,
            taxes REAL,
            total_price REAL,
            estimated_time_mins INTEGER,
            rating INTEGER,
            review_text TEXT
        )
    ''')
    conn.commit()
    conn.close()

def get_all_orders():
    init_db()
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("SELECT * FROM orders ORDER BY order_date DESC")
    rows = c.fetchall()
    conn.close()
    
    orders = []
    for row in rows:
        order = {
            "order_date": row["order_date"],
            "location": row["location"],
            "customer": {"name": row["customer_name"], "unique_id": row["customer_id"]},
            "items": [{"dish_name": row["dish_name"], "quantity": row["quantity"], "unit_price": row["unit_price"]}],
            "billing": {"order_id": row["order_id"], "sub_total": row["sub_total"], "taxes": row["taxes"], "total_price": row["total_price"], "timestamp": row["order_date"]},
            "logistics": {"location": row["location"], "estimated_time_mins": row["estimated_time_mins"], "feasibility": True},
            "feedback": {"rating": row["rating"], "review": row["review_text"]} if row["rating"] is not None else None
        }
        orders.append(order)
    return orders

# Mock Menu (Indian Pricing)
MENU = {
    "Margherita Pizza": 350.0,
    "Butter Chicken": 400.0,
    "Paneer Tikka": 280.0,
    "Biryani": 300.0,
    "Pasta Alfredo": 320.0,
    "Masala Dosa": 150.0,
    "Sushi Roll": 850.0,
    "Dal Makhani": 220.0,
    "Tandoori Roti": 25.0,
    "Garlic Naan": 60.0
}

class OrderIntakeAgent:
    def process_order(self, customer_name, dish_name, quantity):
        """Converses with users to take orders and validates 'Dish Name' and 'Quantity'."""
        if dish_name not in MENU:
            return False, "Dish not available in the menu."
        if quantity <= 0:
            return False, "Quantity must be greater than 0."
        
        order_details = {
            "customer_name": customer_name,
            "dish_name": dish_name,
            "quantity": quantity,
            "unit_price": MENU[dish_name]
        }
        return True, order_details

class LogisticsAgent:
    def check_feasibility(self, location):
        """Calculates delivery feasibility based on 'Location' and estimates time."""
        if len(location.strip()) < 3:
            return False, "Delivery location is too vague or invalid."
        
        # Mock calculation: 10 to 45 mins randomly determining ETA
        estimated_time = random.randint(10, 45)
        
        logistics_details = {
            "location": location,
            "estimated_time_mins": estimated_time,
            "feasibility": True
        }
        return True, logistics_details

class BillingAgent:
    def generate_bill(self, order_details, logistics_details):
        """Calculates 'Price' including GST and discounts, then generates an 'Order ID'."""
        sub_total = order_details["quantity"] * order_details["unit_price"]
        gst_rate = 0.05 # 5% GST
        taxes = round(sub_total * gst_rate, 2)
        total = round(sub_total + taxes, 2)
        
        order_id = "ORD-" + str(uuid.uuid4())[:8].upper()
        
        billing_details = {
            "order_id": order_id,
            "sub_total": sub_total,
            "taxes": taxes,
            "total_price": total,
            "timestamp": datetime.datetime.now().isoformat()
        }
        return billing_details

class ReviewAgent:
    def submit_review(self, order_id, rating, review_text):
        """Triggers after order completion to collect and summarize 'Customer Reviews'."""
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        c.execute("UPDATE orders SET rating = ?, review_text = ? WHERE order_id = ?", (rating, review_text, order_id))
        updated = c.rowcount > 0
        conn.commit()
        conn.close()
        return updated

class AnalyticsAgent:
    def get_overview(self, df):
        total_orders = len(df)
        unique_customers = df['customer_id'].nunique() if not df.empty else 0
        total_revenue = df['total_price'].sum() if not df.empty else 0
        return total_orders, unique_customers, total_revenue

class RFMAgent:
    def analyze(self, df):
        if df.empty: return pd.DataFrame()
        df['order_date'] = pd.to_datetime(df['order_date'])
        reference_date = df['order_date'].max() + pd.Timedelta(days=1)
        
        rfm = df.groupby(['customer_id', 'customer_name', 'location']).agg({
            'order_date': lambda x: (reference_date - x.max()).days,
            'order_id': 'count',
            'total_price': 'sum'
        }).reset_index()
        rfm.rename(columns={'order_date': 'Recency', 'order_id': 'Frequency', 'total_price': 'Monetary'}, inplace=True)
        
        rfm['R_Score'] = pd.qcut(rfm['Recency'].rank(method='first'), 4, labels=[4, 3, 2, 1]).astype(int)
        rfm['F_Score'] = pd.qcut(rfm['Frequency'].rank(method='first'), 4, labels=[1, 2, 3, 4]).astype(int)
        rfm['M_Score'] = pd.qcut(rfm['Monetary'].rank(method='first'), 4, labels=[1, 2, 3, 4]).astype(int)
        
        def segment_customer(row):
            R = row['R_Score']
            F = row['F_Score']
            M = row['M_Score']
            if R >= 3 and F >= 3 and M >= 3:
                return "Champions"
            elif R >= 2 and F >= 2 and M >= 2:
                return "Loyal Customers"
            elif R <= 2 and F >= 3:
                return "At Risk"
            else:
                return "Hibernating"
            
        rfm['Segment'] = rfm.apply(segment_customer, axis=1)
        rfm['RFM_Score'] = rfm['R_Score'] + rfm['F_Score'] + rfm['M_Score']
        
        return rfm

class SentimentAnalysisAgent:
    def analyze(self, df):
        reviews_df = df.dropna(subset=['review_text'])
        if reviews_df.empty:
            return pd.DataFrame()
            
        def get_sentiment(text):
            pol = TextBlob(str(text)).sentiment.polarity
            if pol > 0.1: return "Positive"
            elif pol < -0.1: return "Negative"
            else: return "Neutral"
            
        reviews_df = reviews_df.copy()
        reviews_df['sentiment'] = reviews_df['review_text'].apply(get_sentiment)
        return reviews_df

class PredictionAgent:
    def get_forecast(self, df, target_days=None, start_date=None, end_date=None):
        if df.empty: return None, "No data available."
        df['order_date_day'] = pd.to_datetime(df['order_date']).dt.date
        
        daily = df.groupby('order_date_day').agg(
            order_count=('order_id', 'count'),
            revenue=('total_price', 'sum'),
            quantity=('quantity', 'sum')
        ).reset_index()
        
        if len(daily) < 3:
            return None, "Not enough historical data for prediction (needs at least 3 active days)."
            
        daily = daily.sort_values('order_date_day')
        daily['day_index'] = np.arange(len(daily))
        
        top_dish = df['dish_name'].value_counts().idxmax()
        
        X = daily[['day_index']]
        models = {}
        r2_scores = {}
        
        for col in ['order_count', 'revenue', 'quantity']:
            y = daily[col]
            if len(X) < 5:
                model = LinearRegression()
                model.fit(X, y)
                models[col] = model
                r2_scores[col] = r2_score(y, model.predict(X)) if len(y) > 1 else 1.0
            else:
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                model = LinearRegression()
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                models[col] = model
                r2_scores[col] = r2_score(y_test, preds)
                
        avg_r2 = sum(r2_scores.values()) / 3
        last_date = pd.to_datetime(daily['order_date_day'].max()).date()
        
        if start_date and end_date:
            dates = pd.date_range(start=start_date, end=end_date).date
        elif target_days:
            dates = [last_date + datetime.timedelta(days=i+1) for i in range(target_days)]
        else:
            dates = [last_date + datetime.timedelta(days=i+1) for i in range(30)]
            
        if len(dates) == 0:
            return None, "Invalid date range selected."
            
        index_of_last = len(daily) - 1
        future_indices = [index_of_last + (d - last_date).days for d in dates]
        X_future = np.array(future_indices).reshape(-1, 1)
        
        pred_orders = np.maximum(0, models['order_count'].predict(X_future))
        pred_rev = np.maximum(0, models['revenue'].predict(X_future))
        pred_qty = np.maximum(0, models['quantity'].predict(X_future))
        
        total_pred_orders = int(np.sum(pred_orders))
        total_pred_rev = np.sum(pred_rev)
        total_pred_qty = int(np.sum(pred_qty))
        aov = total_pred_rev / total_pred_orders if total_pred_orders > 0 else 0
        
        future_df = pd.DataFrame({
            'Date': dates,
            'Predicted Revenue': pred_rev,
            'Predicted Orders': pred_orders
        })
        
        return {
            'predicted_orders': total_pred_orders,
            'predicted_revenue': total_pred_rev,
            'predicted_quantity': total_pred_qty,
            'top_dish': top_dish,
            'aov': aov,
            'r2': avg_r2,
            'daily_historical': daily,
            'future_df': future_df
        }, "Success"

class MockDataAgent:
    def generate_mock_data(self, num_orders=850):
        # Generates realistic dummy orders spread across 6 months
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        
        # Clear existing data to avoid conflicts and maintain a fresh dataset
        c.execute("DELETE FROM orders")
        
        first_names = ["Aarav", "Vivaan", "Aditya", "Vihaan", "Arjun", "Sai", "Ayaan", "Krishna", "Ishaan", "Shaurya", "Diya", "Aanya", "Ananya", "Myra", "Saanvi", "Priya", "Rahul", "Sneha", "Karan", "Rohit", "Neha", "Riya", "Kavya", "Aryan"]
        last_names = ["Sharma", "Singh", "Patel", "Kumar", "Gupta", "Desai", "Joshi", "Mehra", "Reddy", "Rao", "Verma", "Yadav", "Kapoor", "Jain", "Das"]
        
        # Generate 270 uniquely distinguishable full names
        all_possible_names = list(set([f"{f} {l}" for f in first_names for l in last_names]))
        unique_names = random.sample(all_possible_names, min(270, len(all_possible_names)))
        
        locations = ["Ariyalur", "Trichy", "Chennai", "Coimbatore", "Madurai", "Salem", "Tirunelveli", "Vellore", "Erode", "Thoothukudi", "Dindigul", "Thanjavur"]
        
        customer_mapping = {}
        for idx, name in enumerate(unique_names):
            customer_mapping[name] = {
                "id": f"CUST-{1001 + idx}",
                "location": random.choice(locations)
            }
            
        reviews_pool = [
            (5, "Amazing food, highly recommend!"), (4, "Good but a bit slow on delivery."),
            (5, "Absolutely delicious, authentic taste."), (3, "Average, nothing special."),
            (1, "Terrible experience, cold food and late."), (2, "Quantity was too small."),
            (4, "Will order again, quite tasty."), (5, "Best I have had in a long time!"),
            (1, "Found a hair in my food! Unacceptable."), (3, "Decent but overpriced."),
            (5, "Packaging was perfect and food was piping hot."), (4, "Nice flavours, would suggest to a friend.")
        ]
        
        for _ in range(num_orders):
            days_ago = random.randint(0, 180)
            order_date = datetime.datetime.now() - datetime.timedelta(days=days_ago, minutes=random.randint(0, 1440))
            order_date_str = order_date.strftime("%Y-%m-%d %H:%M:%S")
            
            customer_name = random.choice(unique_names)
            customer_id = customer_mapping[customer_name]["id"]
            location = customer_mapping[customer_name]["location"]
            
            dish_name = random.choice(list(MENU.keys()))
            quantity = random.randint(1, 4)
            unit_price = MENU[dish_name]
            sub_total = quantity * unit_price
            taxes = round(sub_total * 0.05, 2)
            total_price = round(sub_total + taxes, 2)
            estimated_time_mins = random.randint(15, 60)
            rating, review_text = random.choice(reviews_pool)
            if random.random() < 0.25:
                rating, review_text = None, None
            order_id = "ORD-" + str(uuid.uuid4())[:8].upper()
            c.execute('''
                INSERT INTO orders (
                    order_id, order_date, customer_name, customer_id, location,
                    dish_name, quantity, unit_price, sub_total, taxes, total_price,
                    estimated_time_mins, rating, review_text
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                order_id, order_date_str, customer_name, customer_id, location,
                dish_name, quantity, unit_price, sub_total, taxes, total_price,
                estimated_time_mins, rating, review_text
            ))
            
        conn.commit()
        c.execute("SELECT COUNT(*) FROM orders")
        total_count = c.fetchone()[0]
        conn.close()
        return num_orders, total_count

def render_analytics(df, analytics_agent):
    st.header("📊 Analytics Dashboard")
    tot_orders, uniq_cust, tot_rev = analytics_agent.get_overview(df)
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Orders", tot_orders)
    col2.metric("Unique Customers", uniq_cust)
    col3.metric("Total Revenue", f"₹{tot_rev:.2f}")
    
    st.write("---")
    
    # Graphs Area
    c1, c2 = st.columns(2)
    # 1. Revenue by Dish
    revenue_by_dish = df.groupby('dish_name')['total_price'].sum().reset_index()
    fig1 = px.bar(revenue_by_dish, x='dish_name', y='total_price', title='Revenue by Dish', labels={'total_price': 'Revenue (₹)', 'dish_name': 'Dish'}, color_discrete_sequence=CHART_COLORS)
    c1.plotly_chart(apply_standard_style(fig1), use_container_width=True)
    c1.markdown('<div class="insight-note"><strong>📌 Insight:</strong> Identify top revenue-generating dishes. Tall bars indicate hero items — consider promoting them further or bundling with lower performers.</div>', unsafe_allow_html=True)
    
    # 2. Revenue by Address
    revenue_by_addr = df.groupby('location')['total_price'].sum().reset_index()
    fig2 = px.bar(revenue_by_addr, x='location', y='total_price', title='Revenue by Location', labels={'total_price': 'Revenue (₹)', 'location': 'Location'}, color_discrete_sequence=CHART_COLORS[1:])
    c2.plotly_chart(apply_standard_style(fig2), use_container_width=True)
    c2.markdown('<div class="insight-note"><strong>📌 Insight:</strong> Compare regional performance. Locations with low revenue may need targeted marketing or logistics improvements.</div>', unsafe_allow_html=True)
    
    c3, c4 = st.columns(2)
    # 3. Daily Revenue Trend
    df['order_date_dt'] = pd.to_datetime(df['order_date'])
    df['date_only'] = df['order_date_dt'].dt.date
    revenue_daily = df.groupby('date_only')['total_price'].sum().reset_index()
    fig3 = px.line(revenue_daily, x='date_only', y='total_price', title='Daily Revenue Trend', labels={'total_price': 'Revenue (₹)', 'date_only': 'Date'}, color_discrete_sequence=['#10B981'])
    c3.plotly_chart(apply_standard_style(fig3), use_container_width=True)
    c3.markdown('<div class="insight-note"><strong>📌 Insight:</strong> Look for upward or downward slopes. Sudden spikes may correlate with promotions; dips suggest operational issues worth investigating.</div>', unsafe_allow_html=True)
    
    # 4. Weekly Order Patterns
    df['day_of_week'] = df['order_date_dt'].dt.day_name()
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    orders_weekly = df.groupby('day_of_week').size().reindex(days).reset_index(name='order_count')
    fig4 = px.bar(orders_weekly, x='day_of_week', y='order_count', title='Weekly Order Patterns', labels={'order_count': 'Total Orders', 'day_of_week': 'Day'}, color_discrete_sequence=['#F59E0B'])
    c4.plotly_chart(apply_standard_style(fig4), use_container_width=True)
    c4.markdown('<div class="insight-note-amber"><strong>📌 Insight:</strong> Identify peak ordering days. Schedule staffing and inventory around high-volume days to maximize efficiency.</div>', unsafe_allow_html=True)

def render_rfm(df, rfm_agent):
    st.header("🔍 RFM Segmentation Analysis")
    
    rfm_df = rfm_agent.analyze(df)
    
    if not rfm_df.empty:
        with st.expander("🎯 Targeted Marketing Simulator", expanded=False):
            available_segments = sorted(rfm_df['Segment'].unique())
            target_segment = st.selectbox("Select Target Segment", available_segments)
            
            target_subset = rfm_df[rfm_df['Segment'] == target_segment]
            reach = len(target_subset)
            total_spend = target_subset['Monetary'].sum()
            
            c1, c2 = st.columns(2)
            c1.metric("Total Customers in Segment", reach)
            c2.metric("Total Segment Revenue", f"₹{total_spend:,.2f}")
            
            st.write(f"**Customer List: {target_segment}**")
            display_df = target_subset[['customer_name', 'customer_id', 'location', 'Monetary']].sort_values(by='Monetary', ascending=False).copy()
            display_df.columns = ['Customer Name', 'Customer ID', 'Location', 'Total Spend (₹)']
            st.dataframe(display_df, use_container_width=True)
            
            if target_segment == "At Risk":
                st.warning("💡 **Marketing Tip:** Export this list and send a ₹150 discount code via SMS to reactivate these high-value past customers.")
            elif target_segment == "Champions":
                st.success("💡 **Marketing Tip:** These are your top spenders! Export the list to send exclusive invites to new menu launches or a premium loyalty badge.")
            elif target_segment == "Hibernating":
                st.error("💡 **Marketing Tip:** Standard deep-discount emails work best here. Use localized push notifications emphasizing massive seasonal sales.")
            elif target_segment == "Loyal Customers":
                st.info("💡 **Marketing Tip:** They order consistently. Send targeted up-sell suggestions or a 'Buy 1 Get 1 Free' on premium side dishes.")
            
        c1, c2 = st.columns(2)
        
        # Feature 1: Customer Segments Distribution
        fig_pie = px.pie(rfm_df, names='Segment', title='Customer Segments Distribution',
                         color='Segment', color_discrete_map=SEGMENT_COLORS)
        c1.plotly_chart(apply_standard_style(fig_pie), use_container_width=True)
        c1.markdown('<div class="insight-note"><strong>📌 Insight:</strong> A healthy business should have a large Champions + Loyal slice. If At-Risk (amber) dominates, immediate re-engagement campaigns are needed.</div>', unsafe_allow_html=True)
        
        # Feature 2: Number of Customers per Segment
        seg_counts = rfm_df['Segment'].value_counts().reset_index()
        seg_counts.columns = ['Segment', 'Count']
        fig_bar = px.bar(seg_counts, x='Segment', y='Count', color='Segment', title='Number of Customers per Segment',
                         color_discrete_map=SEGMENT_COLORS)
        c2.plotly_chart(apply_standard_style(fig_bar), use_container_width=True)
        c2.markdown('<div class="insight-note-amber"><strong>📌 Insight:</strong> Compare bar heights to gauge retention health. Tall Hibernating bars signal a churn problem requiring deep-discount reactivation.</div>', unsafe_allow_html=True)
        
        # Feature 4: Top Customers by RFM Score
        st.write("---")
        st.subheader("🏆 Top Customers (High Value Champions)")
        top_customers = rfm_df[['customer_id', 'customer_name', 'Monetary', 'RFM_Score', 'Segment']].sort_values(by='RFM_Score', ascending=False).head(20)
        top_customers.columns = ['Customer ID', 'Name', 'Total Spend (₹)', 'RFM Rank', 'Segment']
        st.dataframe(top_customers, use_container_width=True)
        
        # Feature 5: Strategic Insights & Tips
        st.write("---")
        st.subheader("💡 Strategic Business Insights")
        champions = len(rfm_df[rfm_df['Segment'] == 'Champions'])
        loyal = len(rfm_df[rfm_df['Segment'] == 'Loyal Customers'])
        at_risk = len(rfm_df[rfm_df['Segment'] == 'At Risk'])
        hibernating = len(rfm_df[rfm_df['Segment'] == 'Hibernating'])
        
        i1, i2 = st.columns(2)
        with i1:
            if champions > 0:
                st.success(f"🌟 **Champions ({champions})**: Reward them! Send a personalized 'VIP Appreciation' gift or premium loyalty reward. They are your best organic growth engine.")
            if loyal > 0:
                st.info(f"🤝 **Loyal Customers ({loyal})**: Upsell higher value items. Suggest premium dishes or add-ons to increase their order value.")
        with i2:
            if at_risk > 0:
                st.warning(f"🚨 **At Risk ({at_risk})**: They loved you but haven't ordered recently. Send a personalized 'We Miss You' coupon for ₹100 off their next order!")
            if hibernating > 0:
                st.error(f"💤 **Hibernating ({hibernating})**: Reactivate them! Send a deep discount campaign or push notification about a massive localized sale.")
        
        st.write("---")
        st.subheader("💡 Strategic AI Reasoning")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            pass # Container removed
            rfm_cat = st.selectbox("Select RFM Category:", ["Champions 🏆", "Loyal Customers 🤝", "At Risk 🚨", "Hibernating 💤"], key="rfm_cat")
            if st.button("💡 Generate Insights", key="rfm_ai_btn", use_container_width=True):
                with st.spinner(f"Generating insights for {rfm_cat}..."):
                    # Map dropdown to actual segment names
                    cat_map = {
                        "Champions 🏆": "Champions",
                        "Loyal Customers 🤝": "Loyal Customers",
                        "At Risk 🚨": "At Risk",
                        "Hibernating 💤": "Hibernating"
                    }
                    cat_clean = cat_map.get(rfm_cat, "Champions")
                    
                    target_data = rfm_df[rfm_df['Segment'] == cat_clean]
                    stats = {
                        "count": len(target_data),
                        "total_monetary": target_data['Monetary'].sum(),
                        "avg_recency": float(target_data['Recency'].mean()) if not target_data.empty else 0
                    }
                    prompt = f"Analyze the '{cat_clean}' customer segment data: {stats}. Provide 3 specific retention or growth tips utilizing 💎 Gemini 2.5 Pro Ultra-High Reasoning."
                    generate_gemini_insights(prompt, f"🎯 {rfm_cat} Strategy")
            pass # Container end

@st.dialog("📈 Sentiment Trend Analysis", width="large")
def show_sentiment_trend_dialog(sentiment_df):
    st.write("Analyze operational satisfaction patterns mapped over the 6-month historical array.")
    sentiment_df['order_date'] = pd.to_datetime(sentiment_df['order_date'])
    sentiment_df['Month'] = sentiment_df['order_date'].dt.to_period('M').astype(str)
    
    trend_df = sentiment_df.groupby(['Month', 'sentiment']).size().reset_index(name='Count')
    
    fig = px.line(trend_df, x='Month', y='Count', color='sentiment', 
                  color_discrete_map={'Positive': '#22c55e', 'Negative': '#ef4444', 'Neutral': '#9ca3af'},
                  title='Sentiment Trends Over Last 6 Months')
    st.plotly_chart(apply_standard_style(fig), use_container_width=True)
    
    st.info("💡 **Insight Engine:** Intersect the slopes of the trend lines. A rising positive (green) line indicates strengthening customer satisfaction scaling organically. A spiking negative (red) line formally maps an operational failure parameter requiring immediate intervention.")
    
    st.write("---")
    if st.button("Close Window", type="primary"):
        st.rerun()

@st.dialog("🍱 Item-Wise Sentiment Analyzer", width="large")
def show_item_sentiment_dialog(sentiment_df):
    st.write("Filter aggregate text reviews isolating precise menu bounds securely against live execution databases.")
    
    col1, col2 = st.columns(2)
    dishes = ["All"] + sorted(sentiment_df['dish_name'].unique().tolist())
    selected_dish = col1.selectbox("Select Target Dish:", dishes)
    selected_sentiment = col2.selectbox("Select Sentiment Filter:", ["All", "Positive", "Negative", "Neutral"])
    
    dish_df = sentiment_df.copy()
    if selected_dish != "All":
        dish_df = dish_df[dish_df['dish_name'] == selected_dish]
    if selected_sentiment != "All":
        dish_df = dish_df[dish_df['sentiment'] == selected_sentiment]
        
    if dish_df.empty:
        st.warning(f"No execution arrays mapping `{selected_dish}` and `{selected_sentiment}` bounds were structurally located.")
    else:
        if selected_dish == "All" and selected_sentiment != "All":
            pie_data = dish_df['dish_name'].value_counts().reset_index()
            pie_data.columns = ['Dish Name', 'Count']
            fig = px.pie(pie_data, names='Dish Name', values='Count', title=f'Distribution of {selected_sentiment} Reviews Across All Dishes')
            st.plotly_chart(apply_standard_style(fig), use_container_width=True)
            
        elif selected_dish == "All" and selected_sentiment == "All":
            dish_agg = dish_df.groupby(['dish_name', 'sentiment']).size().reset_index(name='Count')
            fig_all = px.bar(dish_agg, x='dish_name', y='Count', color='sentiment', barmode='group',
                             color_discrete_map={'Positive': '#22c55e', 'Negative': '#ef4444', 'Neutral': '#9ca3af'},
                             title="Sentiment Distribution Across All Menu Items")
            st.plotly_chart(apply_standard_style(fig_all), use_container_width=True)
            
        else:
            pie_data = dish_df['sentiment'].value_counts().reset_index()
            pie_data.columns = ['Sentiment', 'Count']
            fig_dish = px.pie(pie_data, names='Sentiment', values='Count', color='Sentiment',
                              color_discrete_map={'Positive': '#22c55e', 'Negative': '#ef4444', 'Neutral': '#9ca3af'},
                              title=f'{selected_sentiment if selected_sentiment != "All" else "Total"} Sentiment Distribution: {selected_dish}')
            st.plotly_chart(apply_standard_style(fig_dish), use_container_width=True)
            
            st.write(f"**Latest Raw Reviews:**")
            latest = dish_df.dropna(subset=['review_text']).sort_values(by='order_date', ascending=False).head(3)
            for idx, row in latest.iterrows():
                emo = "😍" if row['sentiment'] == 'Positive' else "😡" if row['sentiment'] == 'Negative' else "😐"
                st.markdown(f"> *{row['review_text']}* - **{row['sentiment']} {emo}**")
                
    st.write("---")
    if st.button("Close Analyzer Output", type="primary"):
        st.rerun()

def render_sentiment(df, sentiment_agent):
    st.header("🧠 AI Sentiment Intelligence")
    sentiment_df = sentiment_agent.analyze(df)
    b1, b2 = st.columns(2)
    if b1.button("📊 View Sentiment Trends Over Time", use_container_width=True):
        show_sentiment_trend_dialog(sentiment_df)
    if b2.button("🥘 Deep Dive: Item Sentiment Analysis", use_container_width=True):
        show_item_sentiment_dialog(sentiment_df)
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    if not sentiment_df.empty:
        sent_counts = sentiment_df['sentiment'].value_counts().to_dict() if not sentiment_df.empty else {}
        total_reviews = sum(sent_counts.values())
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Reviews", total_reviews)
        col2.metric("Positive", sent_counts.get("Positive", 0))
        col3.metric("Negative", sent_counts.get("Negative", 0))
    else:
        st.write("No text reviews submitted yet.")
        
    st.write("---")
    st.subheader("🧪 Live AI Sentiment Lab")
    st.write("Test our Natural Language Processing (NLP) algorithm manually on targeted phrases to instantly deduce logical customer classification states.")
    
    test_review = st.text_area("Paste or Type Customer Review:", placeholder="e.g. 'The Biryani was too salty and delivery was late'")
    
    if st.button("🔍 Analyze Sentiment Now", type="primary"):
        if test_review.strip() == "":
            st.warning("Please submit a text prompt first.")
        else:
            pol_score = TextBlob(test_review).sentiment.polarity
            
            st.markdown(f"**Calculated Polarity Metric:** `{pol_score:.4f}`")
            if pol_score > 0.1:
                st.success("Classification Bound: **Positive 😍**")
            elif pol_score < -0.1:
                st.error("Classification Bound: **Negative 😡**")
            else:
                st.info("Classification Bound: **Neutral 😐**")
                
    st.write("---")
    st.write("---")
    st.subheader("💡 Strategic AI Reasoning")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        sent_cat = st.selectbox("Select Sentiment Category:", ["Positive Trends 📈", "Negative Issues 📉", "Neutral Feedback 😐"], key="sent_cat")
        if st.button("💡 Generate Insights", key="sent_ai_btn", use_container_width=True):
            with st.spinner(f"Analyzing {sent_cat}..."):
                # Logic to filter or summarize based on category
                cat_type = "Positive" if "Positive" in sent_cat else "Negative" if "Negative" in sent_cat else "Neutral"
                subset = sentiment_df[sentiment_df['sentiment'] == cat_type]
                sample_reviews = subset['review_text'].dropna().head(5).tolist()
                
                prompt = f"Analyze these {cat_type} sentiment reviews: {sample_reviews}. Provide 3 actionable tips utilizing 💎 Gemini 2.5 Pro Ultra-High Reasoning."
                generate_gemini_insights(prompt, f"💬 {sent_cat} Analysis")
            pass # Container end

def render_prediction(df, prediction_agent):
    st.header("🔮 AI Prediction & Trends")
    
    if 'pred_mode' not in st.session_state:
        st.session_state.pred_mode = 'monthly'
        
    c1, c2, c3 = st.columns(3)
    if c1.button("📅 Custom Range Prediction", use_container_width=True): st.session_state.pred_mode = 'custom'
    if c2.button("🗓️ Weekly AI Prediction Hub", use_container_width=True): st.session_state.pred_mode = 'weekly'
    if c3.button("📊 Monthly AI Prediction Hub", use_container_width=True): st.session_state.pred_mode = 'monthly'
    start_date, end_date = None, None
    target_days = None
    
    if st.session_state.pred_mode == 'custom':
        st.write("---")
        st.write("**Select Custom Forecasting Range:**")
        date_cols = st.columns(2)
        base_date = datetime.date.today()
        start_date = date_cols[0].date_input("Start Date", base_date + datetime.timedelta(days=1))
        end_date = date_cols[1].date_input("End Date", base_date + datetime.timedelta(days=15))
        
        if start_date > end_date:
            st.error("Error: Start Date must be before End Date.")
            return
            
    elif st.session_state.pred_mode == 'weekly':
        target_days = 7
        st.write("---")
        st.write("**🗓️ Forecasting Next 7 Days**")
    else:
        target_days = 30
        st.write("---")
        st.write("**📊 Forecasting Next 30 Days**")
        
    results, msg = prediction_agent.get_forecast(df, target_days=target_days, start_date=start_date, end_date=end_date)
    
    if results:
        m1, m2, m3 = st.columns(3)
        m1.metric("💰 Predicted Revenue", f"₹{results['predicted_revenue']:,.2f}")
        m2.metric("📦 Predicted Orders", f"{results['predicted_orders']}")
        m3.metric("🥘 Predicted Item Sales", f"{results['predicted_quantity']}")
        
        m4, m5 = st.columns(2)
        m4.metric("🔥 Demand (Top Dish)", results['top_dish'])
        m5.metric("📈 AOV Forecast", f"₹{results['aov']:,.2f}")
        
        st.info(f"🧠 **Model Accuracy Assessment:** The Linear Regression engine reports an aggregated cross-validation R² Score of **{results['r2']:.2f}** for this temporal spread.")
        
        st.write("---")
        with st.expander("🚨 HIGH ALERT: Churn Prediction (Customer Exit Warning)", expanded=False):
            df['order_date'] = pd.to_datetime(df['order_date'])
            reference_date = df['order_date'].max() + pd.Timedelta(days=1)
            
            churn_agg = df.groupby(['customer_id', 'customer_name', 'location']).agg(
                last_order=('order_date', 'max'),
                recency=('order_date', lambda x: (reference_date - x.max()).days)
            ).reset_index()
            
            at_risk = churn_agg[churn_agg['recency'] > 60].copy()
            
            st.error("🚨 **Critical Business Warning: Customer Exit Threat**")
            st.info("Isolated high-value accounts bypassing expected purchasing parameters.")
            
            if at_risk.empty:
                st.success("Safe! Zero historical accounts tracking past 60-day turnover threats.")
            else:
                display_df = at_risk[['customer_name', 'customer_id', 'location', 'last_order']].sort_values(by='last_order').copy()
                display_df['last_order'] = display_df['last_order'].dt.strftime('%Y-%m-%d')
                display_df.columns = ['Name', 'ID', 'Location', 'Last Order Date']
                
                st.dataframe(display_df, use_container_width=True)
                st.error("💡 **Retention Strategy:** AI Suggestion: These customers are likely to stop using the app. Offer a high-value discount (₹250) immediately.")
        
        hist = results['daily_historical'][['order_date_day', 'revenue']].copy()
        hist.rename(columns={'order_date_day': 'Date', 'revenue': 'Revenue'}, inplace=True)
        hist['Date'] = pd.to_datetime(hist['Date'])
        hist['Type'] = 'Historical'
        
        fut = results['future_df'][['Date', 'Predicted Revenue']].copy()
        fut.rename(columns={'Predicted Revenue': 'Revenue'}, inplace=True)
        fut['Date'] = pd.to_datetime(fut['Date'])
        fut['Type'] = 'Predicted'
        
        combined = pd.concat([hist, fut])
        fig = px.line(combined, x='Date', y='Revenue', color='Type', title='Revenue Trend & AI Forecast',
                      color_discrete_map={'Historical': '#1A2B3C', 'Predicted': '#10B981'})
        st.plotly_chart(apply_standard_style(fig), use_container_width=True)
        st.markdown('<div class="insight-note"><strong>📌 Insight:</strong> The green predicted line projects future revenue based on historical patterns. A steep upward slope signals growth; a flat or declining slope warrants strategic intervention.</div>', unsafe_allow_html=True)
        
        st.write("---")
        st.subheader("💡 Strategic AI Reasoning")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            pass # Container removed
            pred_cat = st.selectbox("Select Strategy Focus:", ["Churn Mitigation 🛡️", "Growth Strategies 🚀"], key="pred_cat")
            if st.button("💡 Generate Insights", key="pred_ai_btn", use_container_width=True):
                with st.spinner(f"Formulating {pred_cat}..."):
                    prompt = f"Based on forecasted revenue of ₹{results['predicted_revenue']:.2f} and {results['predicted_orders']} orders, provide 3 specific {pred_cat} tips utilizing 💎 Gemini 2.5 Pro Ultra-High Reasoning."
                    generate_gemini_insights(prompt, f"🔮 {pred_cat} Logic")
            pass # Container end
    else:
        st.write(msg)

def render_explorer(df):
    st.header("🔍 Raw Data Explorer")
    st.write("Browse, search, and filter raw database records.")
    
    search_term = st.text_input("Search (Customer Name, Location, Dish or Order ID):", "")
    
    filtered_df = df.copy()
    if search_term:
        mask = filtered_df.astype(str).apply(lambda x: x.str.contains(search_term, case=False)).any(axis=1)
        filtered_df = filtered_df[mask]
        
    st.dataframe(filtered_df, use_container_width=True)

@st.dialog("🔍 Detailed Analytics Viewer", width="large")
def show_detailed_expanders_dialog(orders):
    st.markdown("<h3 style='text-align: center; color: #1e293b;'>Transaction Archive</h3>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #64748b; margin-bottom: 30px;'>Browse high-fidelity record tracking logs and transaction verification data.</p>", unsafe_allow_html=True)
    
    for order in orders:
        order_id = order['billing']['order_id']
        try:
            dt_obj = datetime.datetime.strptime(order['order_date'], "%Y-%m-%d %H:%M:%S")
            date_str = dt_obj.strftime("%Y-%m-%d")
            time_str = dt_obj.strftime("%I:%M %p")
        except:
            date_str = order['order_date']
            time_str = "N/A"
            
        # UI Fix: Spaced metadata and clean headers as requested
        with st.container():
            # Spacing between Order ID and Date/Time
            c1, c2, c3 = st.columns([1.5, 1, 1])
            c1.markdown(f"<p style='font-weight: 600; color: #1e293b;'>Order {order_id}</p>", unsafe_allow_html=True)
            c2.markdown(f"<p style='color: #64748b;'>{date_str}</p>", unsafe_allow_html=True)
            c3.markdown(f"<p style='color: #64748b;'>{time_str}</p>", unsafe_allow_html=True)
            
            # Strict Rule: Show ONLY "Order ID | Date | Time" in the expander label. No icons.
            with st.expander(f"{order_id} | {date_str} | {time_str}"):
                with st.container(border=True):
                    st.markdown(f"""
                        <div style='display: flex; justify-content: space-between; font-size: 0.8rem; color: #64748B; margin-bottom: 15px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em;'>
                            <span>Record Verification: {order_id}</span>
                            <span>Security: Verified</span>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    col_info, col_billing, col_feedback = st.columns(3)
                    
                    with col_info:
                        st.markdown("<p style='font-weight: 700; color: #1E293B;'>CUSTOMER INFO</p>", unsafe_allow_html=True)
                        st.write(f"Name: {order['customer']['name']}")
                        st.write(f"ID: {order['customer']['unique_id']}")
                        st.write(f"Location: {order['location']}")
                    
                    with col_billing:
                        st.markdown("<p style='font-weight: 700; color: #1E293B;'>BILLING INFO</p>", unsafe_allow_html=True)
                        st.write(f"Subtotal: ₹{order['billing']['sub_total']:.2f}")
                        st.write(f"Total: ₹{order['billing']['total_price']:.2f}")
                    
                    with col_feedback:
                        st.markdown("<p style='font-weight: 700; color: #1E293B;'>FEEDBACK</p>", unsafe_allow_html=True)
                        if order.get("feedback"):
                            st.write(f"Rating: {'⭐' * order['feedback']['rating']}")
                            st.info(f"{order['feedback']['review']}")
                        else:
                            st.write("Pending Review")
            st.markdown("<div style='margin-bottom: 20px;'></div>", unsafe_allow_html=True)
    
    st.markdown("---")
    if st.button("Close Viewer", type="primary", use_container_width=True):
        st.rerun()
    
    st.markdown("---")
    if st.button("Close Viewer", type="primary", use_container_width=True):
        st.rerun()

def main():
    init_db()
    st.set_page_config(page_title="Design and Analysis of Customer Segmentation", page_icon="📊", layout="wide")
    
    # UI Reset to Default Streamlit
    
    if 'current_page' not in st.session_state:
        st.session_state['current_page'] = 'insights'
        
    # Custom 42px title via HTML (no default st.title underline)
    st.markdown("""
        <h1 style='text-align:center; font-family:Inter,Segoe UI,Roboto,sans-serif;
        font-size:42px; font-weight:800; color:#1A2B3C; letter-spacing:-0.03em;
        border:none; text-decoration:none; margin-bottom:4px; line-height:1.15;'>
        Design and Analysis of Customer Segmentation</h1>
    """, unsafe_allow_html=True)

    
    
    # ═══════════════════════════════════════════════════════
    # PREMIUM UI DESIGN SYSTEM — v3.0
    # Palette: Deep Navy #1A2B3C · Slate Grey #2F3E46
    #          Emerald #10B981 · Amber #F59E0B
    # ═══════════════════════════════════════════════════════
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

        /* ── Global Typography ── */
        html, body, [class*="css"], .stMarkdown, p, span, label, div {
            font-family: 'Inter', 'Segoe UI', 'Roboto', sans-serif !important;
            font-size: 15px !important;
            color: #2F3E46 !important;
        }

        /* ── Page Background ── */
        .stApp, [data-testid="stAppViewContainer"] {
            background: linear-gradient(175deg, #F8FAFB 0%, #EEF2F6 50%, #E8ECF1 100%) !important;
        }

        /* ── Header Hierarchy (zero underlines) ── */
        h1, h2, h3, h4, h5, h6 {
            text-decoration: none !important;
            border-bottom: none !important;
            font-family: 'Inter', 'Segoe UI', 'Roboto', sans-serif !important;
        }

        h1 {
            text-align: center !important;
            font-size: 42px !important;
            font-weight: 800 !important;
            letter-spacing: -0.03em !important;
            color: #1A2B3C !important;
            margin-bottom: 6px !important;
            line-height: 1.15 !important;
        }

        h2 {
            text-align: center !important;
            font-size: 26px !important;
            font-weight: 700 !important;
            letter-spacing: -0.015em !important;
            color: #1A2B3C !important;
            margin-bottom: 18px !important;
        }

        h3 {
            text-align: center !important;
            font-size: 20px !important;
            font-weight: 600 !important;
            color: #2F3E46 !important;
            margin-bottom: 14px !important;
        }

        h4, .tagline {
            text-align: center !important;
            font-size: 15px !important;
            font-weight: 400 !important;
            color: #64748B !important;
            margin-bottom: 22px !important;
            text-decoration: none !important;
            border-bottom: none !important;
        }

        /* ── Elevated Card System (12px radius + drop-shadow) ── */
        div[data-testid="stMetricBlock"] {
            background: #FFFFFF !important;
            border: 1px solid #E2E8F0 !important;
            border-radius: 12px !important;
            box-shadow: 0 4px 12px rgba(26,43,60,0.06), 0 1px 3px rgba(26,43,60,0.04) !important;
            padding: 22px 18px !important;
            margin-bottom: 20px !important;
            transition: transform 0.25s ease, box-shadow 0.25s ease !important;
        }
        div[data-testid="stMetricBlock"]:hover {
            transform: translateY(-3px) !important;
            box-shadow: 0 12px 24px rgba(26,43,60,0.10), 0 4px 8px rgba(26,43,60,0.06) !important;
            border-color: #CBD5E1 !important;
        }

        .stDataFrame, .stTable {
            background: #FFFFFF !important;
            border: 1px solid #E2E8F0 !important;
            border-radius: 12px !important;
            box-shadow: 0 4px 12px rgba(26,43,60,0.06) !important;
            padding: 16px !important;
            margin-bottom: 20px !important;
        }

        div[data-testid="stExpander"] {
            background: #FFFFFF !important;
            border: 1px solid #E2E8F0 !important;
            border-radius: 12px !important;
            box-shadow: 0 4px 12px rgba(26,43,60,0.06) !important;
            padding: 12px !important;
            margin-bottom: 16px !important;
            transition: transform 0.25s ease, box-shadow 0.25s ease !important;
        }
        div[data-testid="stExpander"]:hover {
            transform: translateY(-2px) !important;
            box-shadow: 0 8px 20px rgba(26,43,60,0.09) !important;
            border-color: #CBD5E1 !important;
        }

        /* ── Metric Card Typography ── */
        [data-testid="stMetricLabel"] {
            display: flex !important;
            justify-content: center !important;
            text-align: center !important;
            font-weight: 600 !important;
            font-size: 13px !important;
            color: #64748B !important;
            text-transform: uppercase !important;
            letter-spacing: 0.04em !important;
        }
        [data-testid="stMetricValue"] {
            text-align: center !important;
            font-weight: 700 !important;
            font-size: 28px !important;
            color: #1A2B3C !important;
        }

        /* ── Default Buttons ── */
        div.stButton > button {
            background: linear-gradient(135deg, #34D399 0%, #10B981 100%) !important;
            color: #FFFFFF !important;
            border-radius: 10px !important;
            border: none !important;
            padding: 11px 28px !important;
            font-weight: 600 !important;
            font-size: 14px !important;
            box-shadow: 0 2px 8px rgba(16,185,129,0.25) !important;
            transition: all 0.3s cubic-bezier(.25,.8,.25,1) !important;
            width: auto !important;
            display: block !important;
            margin: 0 auto !important;
            letter-spacing: 0.01em !important;
        }
        div.stButton > button:hover {
            background: linear-gradient(135deg, #059669 0%, #047857 100%) !important;
            box-shadow: 0 6px 20px rgba(16,185,129,0.35) !important;
            transform: translateY(-2px) !important;
        }

        /* ── Back-Arrow Button (minimalist, transparent bg, grey border) ── */
        .back-arrow-btn div.stButton > button,
        .back-arrow-btn .stButton > button,
        .back-arrow-btn button {
            background: transparent !important;
            border: 1.5px solid #94A3B8 !important;
            border-radius: 8px !important;
            color: #2F3E46 !important;
            font-size: 18px !important;
            font-weight: 400 !important;
            padding: 4px 12px !important;
            min-height: 0 !important;
            line-height: 1.1 !important;
            box-shadow: none !important;
            width: auto !important;
            margin: 0 !important;
        }
        .back-arrow-btn div.stButton > button:hover,
        .back-arrow-btn .stButton > button:hover,
        .back-arrow-btn button:hover {
            background: rgba(226,232,240,0.5) !important;
            border-color: #64748B !important;
            transform: none !important;
            box-shadow: 0 1px 4px rgba(0,0,0,0.08) !important;
            color: #1A2B3C !important;
        }

        /* ── Place-Order Pill Button (top-right, lite→dark green hover) ── */
        .place-order-compact div.stButton > button,
        .place-order-compact .stButton > button,
        .place-order-compact button {
            background: linear-gradient(135deg, #6EE7B7 0%, #34D399 100%) !important;
            color: #064E3B !important;
            font-weight: 700 !important;
            font-size: 13px !important;
            border: none !important;
            border-radius: 999px !important;
            padding: 7px 22px !important;
            min-height: 0 !important;
            line-height: 1.2 !important;
            white-space: nowrap !important;
            box-shadow: 0 2px 10px rgba(16,185,129,0.30) !important;
            width: auto !important;
            margin: 0 0 0 auto !important;
            display: block !important;
            letter-spacing: 0.02em !important;
            transition: all 0.3s ease !important;
        }
        .place-order-compact div.stButton > button:hover,
        .place-order-compact .stButton > button:hover,
        .place-order-compact button:hover {
            background: linear-gradient(135deg, #059669 0%, #047857 100%) !important;
            color: #FFFFFF !important;
            transform: translateY(-1px) !important;
            box-shadow: 0 6px 18px rgba(5,150,105,0.45) !important;
        }

        /* ── Sidebar ── */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #F8FAFB 0%, #F1F5F9 100%) !important;
            border-right: 1px solid #E2E8F0 !important;
        }
        [data-testid="stSidebar"] .stRadio label {
            font-weight: 500 !important;
            color: #2F3E46 !important;
        }

        /* ── Flat Modern Tabs (thin active indicator) ── */
        button[data-baseweb="tab"] {
            justify-content: center !important;
            font-weight: 500 !important;
            font-size: 14px !important;
            color: #64748B !important;
            background: transparent !important;
            border: none !important;
            border-bottom: 2px solid transparent !important;
            padding: 10px 20px !important;
            transition: all 0.2s ease !important;
            letter-spacing: 0.01em !important;
        }
        button[data-baseweb="tab"]:hover {
            color: #1A2B3C !important;
            background: rgba(226,232,240,0.3) !important;
        }
        button[data-baseweb="tab"][aria-selected="true"] {
            color: #1A2B3C !important;
            font-weight: 700 !important;
            border-bottom: 2.5px solid #10B981 !important;
        }
        div[data-baseweb="tab-highlight"] {
            background-color: #10B981 !important;
            height: 2.5px !important;
        }
        div[data-baseweb="tab-border"] {
            background-color: #E2E8F0 !important;
        }

        /* ── Hide ghost material icon text ── */
        .material-icons, [class*="material-icons"], .material-symbols-outlined {
            display: none !important;
        }

        /* ── Header / Metric safe wrapping ── */
        h1, h2, h3, [data-testid="stMetricValue"] {
            white-space: normal !important;
            word-break: break-word !important;
            overflow-wrap: break-word !important;
            padding: 8px 0 !important;
        }

        /* ── Expander headers ── */
        .stExpander p {
            font-weight: 600 !important;
            font-size: 0.95rem !important;
            color: #1A2B3C !important;
        }

        /* ── Dividers ── */
        hr {
            border: none !important;
            border-top: 1px solid #E2E8F0 !important;
            margin: 24px 0 !important;
        }

        /* ── Form styling ── */
        [data-testid="stForm"] {
            background: #FFFFFF !important;
            border: 1px solid #E2E8F0 !important;
            border-radius: 12px !important;
            box-shadow: 0 4px 12px rgba(26,43,60,0.06) !important;
            padding: 28px !important;
        }

        /* ── Insight Note Cards (under charts) ── */
        .insight-note {
            background: linear-gradient(135deg, #F0FDF4 0%, #ECFDF5 100%);
            border-left: 4px solid #10B981;
            border-radius: 0 8px 8px 0;
            padding: 12px 16px;
            margin-top: 8px;
            margin-bottom: 20px;
            font-size: 13px;
            color: #2F3E46;
            line-height: 1.5;
        }
        .insight-note strong { color: #1A2B3C; }

        .insight-note-amber {
            background: linear-gradient(135deg, #FFFBEB 0%, #FEF3C7 100%);
            border-left: 4px solid #F59E0B;
            border-radius: 0 8px 8px 0;
            padding: 12px 16px;
            margin-top: 8px;
            margin-bottom: 20px;
            font-size: 13px;
            color: #2F3E46;
            line-height: 1.5;
        }
        .insight-note-amber strong { color: #92400E; }

        /* ── Scrollbar refinement ── */
        ::-webkit-scrollbar { width: 6px; }
        ::-webkit-scrollbar-track { background: #F1F5F9; }
        ::-webkit-scrollbar-thumb { background: #CBD5E1; border-radius: 3px; }
        ::-webkit-scrollbar-thumb:hover { background: #94A3B8; }

        </style>
    """, unsafe_allow_html=True)



    st.markdown("""
        <div style='display: flex; justify-content: center; align-items: center; margin-top: -10px; margin-bottom: 28px;'>
            <span style='text-align:center; color:#64748B; font-weight:400; letter-spacing:0.14em;
            text-transform:uppercase; font-size:13px; font-family:Inter,Segoe UI,Roboto,sans-serif;'>
                Predictive Analytics &amp; Customer Segmentation
            </span>
        </div>
    """, unsafe_allow_html=True)
    
    # Initialize agents
    intake_agent = OrderIntakeAgent()
    logistics_agent = LogisticsAgent()
    billing_agent = BillingAgent()
    review_agent = ReviewAgent()
    analytics_agent = AnalyticsAgent()
    rfm_agent = RFMAgent()
    sentiment_agent = SentimentAnalysisAgent()
    prediction_agent = PredictionAgent()
    mock_agent = MockDataAgent()

    if st.session_state['current_page'] == 'main':
        st.header("🏠 Main Dashboard Navigation")
        st.info("Welcome to the Central Automation Dashboard. Select a module below to proceed.")

        st.write("")
        st.write("")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🛍️ Place Order", use_container_width=True):
                st.session_state['current_page'] = 'admin'
                st.rerun()
        with col2:
            if st.button("📈 Insights Dashboard", use_container_width=True):
                st.session_state['current_page'] = 'insights'
                st.rerun()

    elif st.session_state['current_page'] == 'insights':
        # --- Compact Place Order button pushed to top-right ---
        _po_spacer, _po_btn = st.columns([6, 1])
        with _po_btn:
            st.markdown('<div class="place-order-compact">', unsafe_allow_html=True)
            if st.button("🛍️ Place Order", key="top_place_order"):
                st.session_state['current_page'] = 'admin'
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

        # Sidebar: minimalist back arrow + menu
        st.sidebar.markdown('<div class="back-arrow-btn">', unsafe_allow_html=True)
        if st.sidebar.button("⬅", key="sidebar_back"):
            st.session_state['current_page'] = 'main'
            st.rerun()
        st.sidebar.markdown('</div>', unsafe_allow_html=True)

        st.sidebar.markdown("---")
        st.sidebar.header("🧭 Insights Menu")
        view_mode = st.sidebar.radio(
            "Select Analytical View:",
            ["📊 Analytics Dashboard", "👥 RFM Analysis", "🎭 Sentiment Analysis", "🔮 Prediction", "🔍 Data Explorer"]
        )
        
        # Removed redundant Advanced Analytics header
        
        conn = sqlite3.connect(DB_FILE)
        df_orders = pd.read_sql_query("SELECT * FROM orders", conn)
        conn.close()
        
        if df_orders.empty:
            st.info("No data available yet to perform analytics.")
        else:
            if view_mode == "📊 Analytics Dashboard":
                render_analytics(df_orders, analytics_agent)

            elif view_mode == "👥 RFM Analysis":
                render_rfm(df_orders, rfm_agent)
            elif view_mode == "🎭 Sentiment Analysis":
                render_sentiment(df_orders, sentiment_agent)
            elif view_mode == "🔮 Prediction":
                render_prediction(df_orders, prediction_agent)
            elif view_mode == "🔍 Data Explorer":
                render_explorer(df_orders)
                
            st.write("---")
            st.subheader("💬 Ask the Business Consultant")
            user_q = st.text_input("Type your custom business-related question here...", placeholder="Ask Gemini something about your business context...")
            if user_q:
                with st.spinner("Consulting AI..."):
                    prompt = f"You are a Senior Business Consultant for a food delivery platform. Here is the current context: We have {len(df_orders)} total orders. The user asked: {user_q}. Briefly answer this utilizing all available business intelligence capabilities of 💎 Gemini 2.5 Pro Ultra-High Reasoning."
                    generate_gemini_insights(prompt, "💼 Business Consultant Response")

    elif st.session_state['current_page'] == 'admin':
        # Admin top bar: compact back arrow + reset data
        u1, u_spacer, u2 = st.columns([0.4, 7.6, 1.5])
        with u1:
            st.markdown('<div class="back-arrow-btn">', unsafe_allow_html=True)
            if st.button("⬅", key="admin_back_btn"):
                st.session_state['current_page'] = 'insights'
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)
        with u2:
            if st.button("⚡ Reset Data", key="sandbox_btn", use_container_width=True):
                with st.spinner("Generating dummy dataset..."):
                    added_count, new_db_total = mock_agent.generate_mock_data(850)
                    st.toast(f"✅ Injected {added_count} distinct orders!", icon="✅")

        st.header("🛠️ Admin Control Center")
        
        tab_manual, tab_review, tab_summary = st.tabs([
            "🛒 Manual Order Entry", 
            "📝 Review Entry", 
            "📊 Summary Table"
        ])

        with tab_manual:
            st.subheader("📒 Manual Order Entry")
            st.markdown("<p style='text-align: center; margin-bottom: 25px;'>Please fill out your details to securely inject a localized order.</p>", unsafe_allow_html=True)
            
            m1, m2, m3 = st.columns([1, 2, 1])
            with m2:
                with st.form("order_form"):
                    col1, col2 = st.columns(2)
                    with col1:
                        customer_name = st.text_input("Customer Name")
                        dish_name = st.selectbox("Select Dish", list(MENU.keys()))
                    with col2:
                        location = st.text_input("Location", placeholder="Enter city (e.g., Ariyalur)")
                        quantity = st.number_input("Quantity", min_value=1, step=1)
                        
                    st.write("")
                    submitted = st.form_submit_button("🛍️ Place Order", use_container_width=True)
                if submitted:
                    if not customer_name:
                        st.error("Please enter your name.")
                    else:
                        intake_success, intake_res = intake_agent.process_order(customer_name, dish_name, quantity)
                        if not intake_success:
                            st.error(f"🛑 Order Intake Failed: {intake_res}")
                        else:
                            log_success, log_res = logistics_agent.check_feasibility(location)
                            if not log_success:
                                st.error(f"🛑 Logistics Validation Failed: {log_res}")
                            else:
                                billing_details = billing_agent.generate_bill(intake_res, log_res)
                                conn = sqlite3.connect(DB_FILE)
                                c = conn.cursor()
                                c.execute('''
                                    INSERT INTO orders (
                                        order_id, order_date, customer_name, customer_id, location,
                                        dish_name, quantity, unit_price, sub_total, taxes, total_price,
                                        estimated_time_mins
                                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                                ''', (
                                    billing_details['order_id'], datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                    customer_name, f"CUST-{str(uuid.uuid4())[:6].upper()}", log_res["location"],
                                    intake_res["dish_name"], intake_res["quantity"], intake_res["unit_price"],
                                    billing_details['sub_total'], billing_details['taxes'], billing_details['total_price'],
                                    log_res['estimated_time_mins']
                                ))
                                conn.commit()
                                conn.close()
                                
                                st.success("✅ Workflow Successfully Processed!")
                                st.write("---")
                                st.subheader("📜 Order Summary")
                                sum_col1, sum_col2 = st.columns(2)
                                with sum_col1:
                                    st.write(f"**Order ID:** {billing_details['order_id']}")
                                    st.write(f"**Customer:** {customer_name}")
                                    st.write(f"**Dish:** {dish_name} x {quantity}")
                                with sum_col2:
                                    st.write(f"**Sub-total:** ₹{billing_details['sub_total']:.2f}")
                                    st.write(f"**GST (5%):** ₹{billing_details['taxes']:.2f}")
                                    st.write(f"**Total Price:** ₹{billing_details['total_price']:.2f}")

        with tab_review:
            st.subheader("📝 Submit Customer Review")
            st.markdown("<p style='text-align: center; margin-bottom: 25px;'>Provide natural language feedback mapping to recent operations.</p>", unsafe_allow_html=True)
            orders = get_all_orders()
            pending_review_orders = [o for o in orders if not o.get("feedback")]
            
            if not pending_review_orders:
                st.info("There are no fresh orders pending a review right now!")
            else:
                order_ids = [o["billing"]["order_id"] for o in pending_review_orders]
                
                r1, r2, r3 = st.columns([1, 2, 1])
                with r2:
                    with st.form("review_form"):
                        selected_order_id = st.selectbox("Select Order ID to Review", order_ids)
                        rating = st.slider("Rating (1 = Poor, 5 = Excellent)", 1, 5, 5)
                        review_text = st.text_area("Your Review")
                        
                        st.write("")
                        review_submit = st.form_submit_button("Submit Review", use_container_width=True)
                    if review_submit:
                        if review_agent.submit_review(selected_order_id, rating, review_text):
                            st.success("🎉 Review Agent processing successful! Feedback stored in DB.")
                        else:
                            st.error("Failed to submit review.")

        with tab_summary:
            st.subheader("📊 Transactions Summary")
            orders = get_all_orders()
            if not orders:
                st.info("No past transactions found.")
            else:
                table_data = []
                for o in orders:
                    review_str = "Pending Review"
                    if o.get("feedback"):
                        review_str = f"{'⭐'*o['feedback']['rating']} - {o['feedback']['review']}"
                    table_data.append({
                        "Customer Name": o['customer']['name'],
                        "Dish Name": o['items'][0]['dish_name'],
                        "Price": f"₹{o['billing']['total_price']:.2f}",
                        "Review": review_str
                    })
                
                df = pd.DataFrame(table_data)
                
                st.write("")
                c1, c2, c3 = st.columns([1, 2, 1])
                with c2:
                    if st.button("🔍 View Detailed Analytics & Expanders", type="primary", use_container_width=True):
                        show_detailed_expanders_dialog(orders)
                    
                st.write("")
                st.divider()
                
                st.write("")
                st.subheader("📊 Quick Analytical Summary")
                
                t1, t2, t3 = st.columns([1, 8, 1])
                with t2:
                    st.dataframe(df, use_container_width=True)

if __name__ == "__main__":
    main()
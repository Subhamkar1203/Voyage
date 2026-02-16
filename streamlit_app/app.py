"""
Voyage Analytics — Streamlit ML Dashboard
Multi-page app: EDA, Flight Price Predictor, Flight Class Classifier, Route Recommender.
Uses only flights.csv.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="Voyage Analytics — ML Platform",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw', 'flights.csv')


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df['month'] = df['date'].dt.month
        df['day_of_week'] = df['date'].dt.dayofweek
        df['route'] = df['from'] + ' → ' + df['to']
    return df


def main():
    st.sidebar.title("✈️ Voyage Analytics")
    st.sidebar.markdown("**Corporate Travel Intelligence**")

    page = st.sidebar.radio(
        "Navigate",
        ["🏠 Overview", "📊 Exploratory Analysis", "💰 Price Predictor",
         "🎟️ Class Classifier", "🗺️ Route Recommender", "📈 Model Performance"]
    )

    df = load_data()

    if page == "🏠 Overview":
        show_overview(df)
    elif page == "📊 Exploratory Analysis":
        show_eda(df)
    elif page == "💰 Price Predictor":
        show_price_predictor(df)
    elif page == "🎟️ Class Classifier":
        show_class_classifier(df)
    elif page == "🗺️ Route Recommender":
        show_route_recommender(df)
    elif page == "📈 Model Performance":
        show_model_performance()


# ──────────────────────────────────────────────────────────────
# Page: Overview
# ──────────────────────────────────────────────────────────────
def show_overview(df):
    st.title("✈️ Voyage Analytics — Platform Overview")
    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Flights", f"{len(df):,}")
    col2.metric("Unique Routes", df['route'].nunique() if 'route' in df.columns else "N/A")
    col3.metric("Avg Price (BRL)", f"R$ {df['price'].mean():,.0f}")
    col4.metric("Unique Users", df['userCode'].nunique())

    st.markdown("### Dataset Sample")
    st.dataframe(df.head(20), use_container_width=True)

    st.markdown("### Available ML Models")
    st.markdown("""
    | Model | Description | Endpoint |
    |-------|-------------|----------|
    | **Flight Price Regression** | Predicts ticket price from route, class, agency, timing | `/api/v1/flight-price/predict` |
    | **Flight Class Classification** | Classifies flight type (economic/premium/firstClass) | `/api/v1/flight-class/predict` |
    | **Route Recommendation** | Recommends routes based on user travel history | `/api/v1/route-recommend/<user_code>` |
    """)


# ──────────────────────────────────────────────────────────────
# Page: EDA
# ──────────────────────────────────────────────────────────────
def show_eda(df):
    st.title("📊 Exploratory Data Analysis")

    tab1, tab2, tab3, tab4 = st.tabs(["Distribution", "Routes", "Agencies", "Time"])

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            fig = px.histogram(df, x='price', color='flightType', nbins=50,
                               title="Price Distribution by Flight Type",
                               color_discrete_sequence=px.colors.qualitative.Set2)
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig = px.box(df, x='flightType', y='price', color='flightType',
                         title="Price Boxplot by Flight Type",
                         color_discrete_sequence=px.colors.qualitative.Set2)
            st.plotly_chart(fig, use_container_width=True)

        col3, col4 = st.columns(2)
        with col3:
            fig = px.histogram(df, x='distance', nbins=40, title="Distance Distribution",
                               color_discrete_sequence=['#636EFA'])
            st.plotly_chart(fig, use_container_width=True)
        with col4:
            fig = px.scatter(df.sample(min(5000, len(df))), x='distance', y='price',
                             color='flightType', opacity=0.5,
                             title="Price vs Distance (sampled)",
                             color_discrete_sequence=px.colors.qualitative.Set2)
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        top_routes = df['route'].value_counts().head(15)
        fig = px.bar(x=top_routes.values, y=top_routes.index, orientation='h',
                     title="Top 15 Routes by Flight Count",
                     labels={'x': 'Flight Count', 'y': 'Route'},
                     color_discrete_sequence=['#00CC96'])
        st.plotly_chart(fig, use_container_width=True)

        route_price = df.groupby('route')['price'].mean().sort_values(ascending=False).head(15)
        fig = px.bar(x=route_price.values, y=route_price.index, orientation='h',
                     title="Top 15 Most Expensive Routes (Avg Price)",
                     labels={'x': 'Avg Price (BRL)', 'y': 'Route'},
                     color_discrete_sequence=['#EF553B'])
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        agency_stats = df.groupby('agency').agg(
            flights=('travelCode', 'count'),
            avg_price=('price', 'mean'),
            avg_distance=('distance', 'mean')
        ).reset_index().sort_values('flights', ascending=False)

        fig = px.bar(agency_stats, x='agency', y='flights', color='avg_price',
                     title="Agency Performance: Flight Count & Avg Price",
                     color_continuous_scale='RdYlGn_r')
        st.plotly_chart(fig, use_container_width=True)

        fig = px.scatter(agency_stats, x='avg_distance', y='avg_price', size='flights',
                         text='agency', title="Agency: Distance vs Price",
                         color_discrete_sequence=['#AB63FA'])
        st.plotly_chart(fig, use_container_width=True)

    with tab4:
        if 'month' in df.columns:
            monthly = df.groupby('month').agg(
                flights=('travelCode', 'count'),
                avg_price=('price', 'mean')
            ).reset_index()
            fig = px.line(monthly, x='month', y='avg_price', markers=True,
                          title="Monthly Average Price Trend",
                          color_discrete_sequence=['#FFA15A'])
            st.plotly_chart(fig, use_container_width=True)

            fig = px.bar(monthly, x='month', y='flights', title="Monthly Flight Volume",
                         color_discrete_sequence=['#19D3F3'])
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Date information not available in dataset.")


# ──────────────────────────────────────────────────────────────
# Page: Price Predictor
# ──────────────────────────────────────────────────────────────
def show_price_predictor(df):
    st.title("💰 Flight Price Predictor")
    st.markdown("Predict the ticket price for a given flight configuration.")

    col1, col2 = st.columns(2)
    with col1:
        origin = st.selectbox("From", sorted(df['from'].unique()))
        destination = st.selectbox("To", sorted(df['to'].unique()))
        flight_type = st.selectbox("Flight Type", ['economic', 'premium', 'firstClass'])
        agency = st.selectbox("Agency", sorted(df['agency'].unique()))
    with col2:
        distance = st.number_input("Distance (km)", min_value=50.0, max_value=5000.0, value=1000.0)
        time_hrs = st.number_input("Duration (hours)", min_value=0.5, max_value=24.0, value=2.5)
        month = st.slider("Month", 1, 12, 6)
        day_of_week = st.slider("Day of Week (0=Mon)", 0, 6, 2)

    if st.button("🔮 Predict Price", type="primary"):
        try:
            from src.models.flight_price_regression import predict_price
            features = {
                'from': origin, 'to': destination, 'flightType': flight_type,
                'agency': agency, 'distance': distance, 'time': time_hrs,
                'month': month, 'day_of_week': day_of_week
            }
            predicted = predict_price(features)
            st.success(f"### Predicted Price: R$ {predicted:,.2f}")

            # Show similar flights
            similar = df[(df['from'] == origin) & (df['to'] == destination) & (df['flightType'] == flight_type)]
            if len(similar) > 0:
                st.markdown(f"**Historical stats** for {origin} → {destination} ({flight_type}):")
                scol1, scol2, scol3 = st.columns(3)
                scol1.metric("Avg Price", f"R$ {similar['price'].mean():,.0f}")
                scol2.metric("Min Price", f"R$ {similar['price'].min():,.0f}")
                scol3.metric("Max Price", f"R$ {similar['price'].max():,.0f}")
        except FileNotFoundError:
            st.error("⚠️ Model not trained yet. Run `python -m src.training.train_flight_price` first.")
        except Exception as e:
            st.error(f"Prediction error: {e}")


# ──────────────────────────────────────────────────────────────
# Page: Class Classifier
# ──────────────────────────────────────────────────────────────
def show_class_classifier(df):
    st.title("🎟️ Flight Class Classifier")
    st.markdown("Predict whether a flight is economic, premium, or firstClass.")

    col1, col2 = st.columns(2)
    with col1:
        origin = st.selectbox("From", sorted(df['from'].unique()), key='cc_from')
        destination = st.selectbox("To", sorted(df['to'].unique()), key='cc_to')
        agency = st.selectbox("Agency", sorted(df['agency'].unique()), key='cc_agency')
    with col2:
        price = st.number_input("Price (BRL)", min_value=50.0, max_value=10000.0, value=500.0)
        distance = st.number_input("Distance (km)", min_value=50.0, max_value=5000.0, value=1000.0, key='cc_dist')
        time_hrs = st.number_input("Duration (hours)", min_value=0.5, max_value=24.0, value=2.5, key='cc_time')

    month = st.slider("Month", 1, 12, 6, key='cc_month')
    day_of_week = st.slider("Day of Week (0=Mon)", 0, 6, 2, key='cc_dow')

    if st.button("🎯 Classify Flight", type="primary"):
        try:
            from src.models.flight_class_classification import predict_flight_class
            features = {
                'from': origin, 'to': destination, 'agency': agency,
                'price': price, 'distance': distance, 'time': time_hrs,
                'month': month, 'day_of_week': day_of_week
            }
            predicted_class, confidence = predict_flight_class(features)

            st.success(f"### Predicted Class: **{predicted_class}**")
            st.markdown("**Confidence Scores:**")
            for cls, conf in sorted(confidence.items(), key=lambda x: x[1], reverse=True):
                st.progress(conf, text=f"{cls}: {conf:.1%}")
        except FileNotFoundError:
            st.error("⚠️ Model not trained yet. Run `python -m src.training.train_flight_classifier` first.")
        except Exception as e:
            st.error(f"Classification error: {e}")


# ──────────────────────────────────────────────────────────────
# Page: Route Recommender
# ──────────────────────────────────────────────────────────────
def show_route_recommender(df):
    st.title("🗺️ Route Recommender")
    st.markdown("Get personalized flight route recommendations for any user.")

    user_codes = sorted(df['userCode'].unique())
    selected_user = st.selectbox("Select User Code", user_codes)
    top_n = st.slider("Number of recommendations", 3, 15, 5)

    if st.button("🔍 Get Recommendations", type="primary"):
        try:
            import joblib
            MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'models')

            user_route_matrix = joblib.load(os.path.join(MODEL_DIR, 'user_route_matrix.pkl'))
            route_sim_df = joblib.load(os.path.join(MODEL_DIR, 'route_similarity.pkl'))
            knn_model = joblib.load(os.path.join(MODEL_DIR, 'recommendation_knn.pkl'))
            route_profiles = joblib.load(os.path.join(MODEL_DIR, 'route_profiles.pkl'))

            from src.models.route_recommendation import hybrid_recommend
            recs = hybrid_recommend(selected_user, user_route_matrix, route_sim_df, knn_model, top_n=top_n)

            if recs:
                st.markdown("### Recommended Routes")
                rec_df = pd.DataFrame(recs, columns=['Route', 'Score'])
                rec_df['Rank'] = range(1, len(rec_df) + 1)
                rec_df = rec_df[['Rank', 'Route', 'Score']]

                st.dataframe(rec_df, use_container_width=True, hide_index=True)

                # Show user's existing routes
                if selected_user in user_route_matrix.index:
                    user_routes = user_route_matrix.loc[selected_user]
                    visited = user_routes[user_routes > 0].sort_values(ascending=False)
                    st.markdown(f"### User {selected_user}'s Travel History ({len(visited)} routes)")
                    st.dataframe(
                        pd.DataFrame({'Route': visited.index, 'Flights': visited.values}),
                        use_container_width=True, hide_index=True
                    )
            else:
                st.warning("No recommendations could be generated for this user.")
        except FileNotFoundError:
            st.error("⚠️ Recommendation model not trained. Run `python -m src.training.train_recommendation` first.")
        except Exception as e:
            st.error(f"Recommendation error: {e}")


# ──────────────────────────────────────────────────────────────
# Page: Model Performance
# ──────────────────────────────────────────────────────────────
def show_model_performance():
    st.title("📈 Model Performance Dashboard")

    import json
    MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'models')

    # Regression results
    reg_path = os.path.join(MODEL_DIR, 'regression_results.json')
    if os.path.exists(reg_path):
        with open(reg_path) as f:
            reg_results = json.load(f)
        st.markdown("### Flight Price Regression")
        reg_df = pd.DataFrame(reg_results).T
        st.dataframe(reg_df.style.highlight_min(subset=['rmse', 'mae'], color='lightgreen')
                      .highlight_max(subset=['r2'], color='lightgreen'),
                      use_container_width=True)

        fig = px.bar(reg_df.reset_index(), x='index', y='r2',
                     title="R² Score by Model", color='r2',
                     labels={'index': 'Model', 'r2': 'R²'},
                     color_continuous_scale='Greens')
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Regression model not yet trained.")

    # Classification results
    cls_path = os.path.join(MODEL_DIR, 'classification_results.json')
    if os.path.exists(cls_path):
        with open(cls_path) as f:
            cls_results = json.load(f)
        st.markdown("### Flight Class Classification")
        cls_df = pd.DataFrame(cls_results).T
        st.dataframe(cls_df.style.highlight_max(subset=['accuracy', 'f1'], color='lightgreen'),
                      use_container_width=True)

        fig = px.bar(cls_df.reset_index(), x='index', y='accuracy',
                     title="Accuracy by Classifier", color='accuracy',
                     labels={'index': 'Model', 'accuracy': 'Accuracy'},
                     color_continuous_scale='Blues')
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Classification model not yet trained.")

    st.markdown("---")
    st.markdown("*Train models first, then revisit this page to see performance metrics.*")


if __name__ == '__main__':
    main()

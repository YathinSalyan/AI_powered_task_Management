# AI Powered Task Management System - Streamlit App
import streamlit as st
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import pickle
import os
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="AI Powered Task Management",
    page_icon="✅",
    layout="wide",
    initial_sidebar_state="expanded"
)

# App title and intro
st.title("🚀 AI Powered Task Management System")
st.markdown("""
This application helps you manage your tasks intelligently using AI-powered:
- 💬 Sentiment Analysis
- 📊 Task Prioritization
- ⏱️ Completion Time Prediction
""")

# Create directories for models and data if they don't exist
os.makedirs('models', exist_ok=True)
os.makedirs('data', exist_ok=True)

# Download necessary NLTK resources
@st.cache_resource
def download_nltk_resources():
    resources = ['punkt', 'stopwords', 'wordnet']
    for resource in resources:
        try:
            nltk.download(resource, quiet=True)
        except Exception as e:
            st.warning(f"Error downloading NLTK resource {resource}: {e}")

download_nltk_resources()

# Define preprocessing function
def preprocess_text(text):
    """Preprocesses the input text for sentiment analysis."""
    if not isinstance(text, str) or pd.isna(text):
        return ''
    
    try:
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove user @ references and '#'
        text = re.sub(r'\@\w+|\#', '', text)
        
        # Remove punctuations and numbers
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\d+', '', text)
        
        # Tokenize the text
        try:
            tokens = word_tokenize(text)
        except LookupError:
            tokens = text.split()
        
        # Remove stopwords if available
        try:
            stop_words = set(stopwords.words('english'))
            tokens = [word for word in tokens if word not in stop_words]
        except LookupError:
            pass
        
        # Lemmatize if available
        try:
            lemmatizer = WordNetLemmatizer()
            tokens = [lemmatizer.lemmatize(word) for word in tokens]
        except LookupError:
            pass
        
        # Join tokens back into string
        text = ' '.join(tokens)
        return text
    except Exception as e:
        st.error(f"Error preprocessing text: {e}")
        return text

# Define sentiment prediction function
def predict_sentiment(text, model, vectorizer):
    """Predicts sentiment of a given text."""
    try:
        processed = preprocess_text(text)
        if not processed:
            return "Neutral", 0.5
            
        tfidf_vector = vectorizer.transform([processed])
        prediction = model.predict(tfidf_vector)[0]
        probability = model.predict_proba(tfidf_vector)[0]
        
        if prediction == 1:
            sentiment = "Positive"
            confidence = probability[1]
        else:
            sentiment = "Negative"
            confidence = probability[0]
            
        return sentiment, confidence
    
    except Exception as e:
        st.error(f"Error predicting sentiment: {e}")
        return "Neutral", 0.5

# Create task optimization model
def create_task_optimization_model():
    """Creates a task optimization model that scores tasks."""
    
    def optimize_tasks(tasks_df):
        """Optimize and prioritize tasks based on multiple factors."""
        # Create a copy to avoid modifying the original DataFrame
        df = tasks_df.copy()
        
        # Calculate days to deadline
        if 'deadline' in df.columns:
            try:
                df['days_to_deadline'] = pd.to_datetime(df['deadline'], errors='coerce') - pd.Timestamp.now()
                df['days_to_deadline'] = df['days_to_deadline'].dt.total_seconds() / (24 * 3600)
                
                # Handle missing or invalid dates
                df['days_to_deadline'] = df['days_to_deadline'].fillna(30)  # Default to 30 days
                
                # Normalize deadline score (closer deadline = higher score)
                max_days = max(df['days_to_deadline'].max(), 30)  # At least 30 days to avoid division by zero
                df['deadline_score'] = 1 - (df['days_to_deadline'] / max_days).clip(0, 1)
            except Exception as e:
                st.warning(f"Error calculating deadline score: {e}")
                df['deadline_score'] = 0.5  # Default score
        else:
            df['deadline_score'] = 0.5  # Default score
            
        # Convert priority to numeric score
        priority_map = {'urgent': 1.0, 'high': 0.75, 'medium': 0.5, 'low': 0.25}
        if 'priority' in df.columns:
            df['priority_score'] = df['priority'].str.lower().map(priority_map).fillna(0.5)
        else:
            df['priority_score'] = 0.5  # Default score
            
        # Add sentiment score if available
        if 'sentiment_score' not in df.columns:
            df['sentiment_score'] = 0.5  # Default score
            
        # Calculate final optimization score
        df['optimization_score'] = (
            df['deadline_score'] * 0.4 +  # Weight for deadline
            df['priority_score'] * 0.4 +  # Weight for priority
            df['sentiment_score'] * 0.2    # Weight for sentiment
        )
        
        # Sort tasks by optimization score
        return df.sort_values('optimization_score', ascending=False)
    
    return optimize_tasks

# Create predictive analytics model
def create_predictive_analytics_model():
    """Creates a predictive analytics model for estimating task completion time."""
    
    def predict_completion_time(task_history_df, new_task):
        """Predicts task completion time based on historical data"""
        if task_history_df is None or task_history_df.empty:
            # Default predictions if no history
            base_times = {'bug_fix': 4, 'feature': 8, 'documentation': 2, 'other': 6}
            priority_multipliers = {'urgent': 0.8, 'high': 0.9, 'medium': 1.0, 'low': 1.2}
            
            task_type = new_task.get('task_type', 'other').lower()
            priority = new_task.get('priority', 'medium').lower()
            
            base_time = base_times.get(task_type, 6)
            multiplier = priority_multipliers.get(priority, 1.0)
            
            return base_time * multiplier
        
        try:
            # Filter relevant history
            filtered_df = task_history_df.copy()
            
            # Handle potential missing columns
            required_columns = ['task_type', 'priority', 'assigned_to', 'actual_completion_time']
            for column in required_columns:
                if column not in filtered_df.columns:
                    if column == 'actual_completion_time':
                        # This is critical - we can't continue without it
                        raise ValueError("'actual_completion_time' column is required for prediction")
            
            # Apply filters if corresponding columns exist
            if 'task_type' in filtered_df.columns and 'task_type' in new_task:
                filtered_df = filtered_df[filtered_df['task_type'].str.lower() == new_task['task_type'].lower()]
            
            if 'priority' in filtered_df.columns and 'priority' in new_task:
                filtered_df = filtered_df[filtered_df['priority'].str.lower() == new_task['priority'].lower()]
                
            if 'assigned_to' in filtered_df.columns and 'assigned_to' in new_task:
                filtered_df = filtered_df[filtered_df['assigned_to'] == new_task['assigned_to']]
            
            # Calculate average completion time
            if filtered_df.empty:
                # Fall back to overall average if no matching tasks
                return task_history_df['actual_completion_time'].mean()
            else:
                return filtered_df['actual_completion_time'].mean()
                
        except Exception as e:
            st.warning(f"Error predicting completion time: {e}")
            # Return a reasonable default
            return 4.0  # Default 4 hours
    
    return predict_completion_time

# Train or load sentiment model
@st.cache_resource
def load_or_train_sentiment_model():
    """Load saved models or create new ones if they don't exist."""
    # Define paths
    model_path = 'models/sentiment_model.pkl'
    vectorizer_path = 'models/tfidf_vectorizer.pkl'
    
    # Try to load existing models
    if os.path.exists(model_path) and os.path.exists(vectorizer_path):
        try:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            with open(vectorizer_path, 'rb') as f:
                vectorizer = pickle.load(f)
            return model, vectorizer
        except Exception as e:
            st.warning(f"Failed to load saved models: {e}")
            st.info("Training new models...")
    else:
        st.info("No saved models found. Training new models...")
    
    # Create simple models for demonstration
    # This is a simplified version as the full training process requires the dataset
    vectorizer = TfidfVectorizer(max_features=5000)
    sample_texts = [
        "This is a great task that I'm excited to work on",
        "I love this project and I'm looking forward to it",
        "I'm excited to start this work",
        "This task is urgent and critical",
        "This is a frustrating and difficult task",
        "I hate working on this kind of project",
        "This deadline is too tight and unreasonable",
        "The requirements for this task are unclear"
    ]
    sample_sentiments = [1, 1, 1, 0, 0, 0, 0, 0]  # 1 = positive, 0 = negative
    
    # Fit the vectorizer and transform the texts
    X = vectorizer.fit_transform([preprocess_text(text) for text in sample_texts])
    
    # Create and train the model
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X, sample_sentiments)
    
    # Save the models
    try:
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        with open(vectorizer_path, 'wb') as f:
            pickle.dump(vectorizer, f)
    except Exception as e:
        st.warning(f"Failed to save models: {e}")
    
    return model, vectorizer

# Load or create task data
@st.cache_data
def load_task_data():
    """Load existing task data or create a new dataframe if it doesn't exist."""
    filepath = 'data/tasks.csv'
    
    if os.path.exists(filepath):
        try:
            tasks_df = pd.read_csv(filepath)
            # Ensure all columns exist
            required_columns = ['id', 'title', 'description', 'priority', 'deadline', 
                               'status', 'assigned_to', 'created_date', 'sentiment', 
                               'sentiment_score', 'estimated_hours']
            for col in required_columns:
                if col not in tasks_df.columns:
                    tasks_df[col] = ""
            return tasks_df
        except Exception as e:
            st.warning(f"Error loading task data: {e}")
    
    # Create empty DataFrame with required columns
    tasks_df = pd.DataFrame(columns=[
        'id', 'title', 'description', 'priority', 'deadline', 'status',
        'assigned_to', 'task_type', 'created_date', 'sentiment', 'sentiment_score',
        'estimated_hours'
    ])
    return tasks_df

@st.cache_data
def load_task_history():
    """Load existing task history or create a new dataframe if it doesn't exist."""
    filepath = 'data/task_history.csv'
    
    if os.path.exists(filepath):
        try:
            return pd.read_csv(filepath)
        except Exception as e:
            st.warning(f"Error loading task history: {e}")
    
    # Create empty DataFrame with required columns
    history_df = pd.DataFrame(columns=[
        'id', 'title', 'task_type', 'priority', 'assigned_to', 
        'actual_completion_time', 'completed_date'
    ])
    return history_df

# Save task data
def save_task_data(tasks_df):
    """Save task data to CSV file."""
    try:
        tasks_df.to_csv('data/tasks.csv', index=False)
    except Exception as e:
        st.error(f"Error saving task data: {e}")

# Save task history
def save_task_history(history_df):
    """Save task history to CSV file."""
    try:
        history_df.to_csv('data/task_history.csv', index=False)
    except Exception as e:
        st.error(f"Error saving task history: {e}")

# Load models and data
sentiment_model, vectorizer = load_or_train_sentiment_model()
task_optimizer = create_task_optimization_model()
completion_predictor = create_predictive_analytics_model()
tasks_df = load_task_data()
task_history_df = load_task_history()

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Dashboard", "Task Management", "Analytics", "Settings"])

# Dashboard page
if page == "Dashboard":
    st.header("📊 Dashboard")
    
    # Display task statistics
    col1, col2, col3, col4 = st.columns(4)
    
    # Task counts by status
    with col1:
        total_tasks = len(tasks_df)
        st.metric("Total Tasks", total_tasks)
    
    with col2:
        if 'status' in tasks_df.columns:
            completed_tasks = len(tasks_df[tasks_df['status'] == 'Completed'])
        else:
            completed_tasks = 0
        st.metric("Completed Tasks", completed_tasks)
    
    with col3:
        if 'status' in tasks_df.columns:
            pending_tasks = len(tasks_df[tasks_df['status'] == 'In Progress'])
        else:
            pending_tasks = 0
        st.metric("In Progress Tasks", pending_tasks)
    
    with col4:
        if 'deadline' in tasks_df.columns:
            overdue_tasks = len(tasks_df[(tasks_df['status'] != 'Completed') & 
                                       (pd.to_datetime(tasks_df['deadline'], errors='coerce') < pd.Timestamp.now())])
        else:
            overdue_tasks = 0
        st.metric("Overdue Tasks", overdue_tasks, delta=None, delta_color="inverse")
    
    # Display prioritized tasks
    st.subheader("🔝 Top Priority Tasks")
    
    if not tasks_df.empty and 'status' in tasks_df.columns:
        # Only consider non-completed tasks
        active_tasks = tasks_df[tasks_df['status'] != 'Completed'].copy()
        
        if not active_tasks.empty:
            # Add sentiment score if missing
            if 'sentiment_score' not in active_tasks.columns or active_tasks['sentiment_score'].isnull().any():
                for i, row in active_tasks.iterrows():
                    if pd.isna(row.get('sentiment_score')) or row.get('sentiment_score') == "":
                        description = row.get('description', '')
                        if description:
                            sentiment, confidence = predict_sentiment(description, sentiment_model, vectorizer)
                            # Convert sentiment to score (0-1 scale, higher = more urgent)
                            sentiment_score = 1 - confidence if sentiment == "Positive" else confidence
                            active_tasks.at[i, 'sentiment_score'] = sentiment_score
                            active_tasks.at[i, 'sentiment'] = sentiment
                        else:
                            active_tasks.at[i, 'sentiment_score'] = 0.5
                            active_tasks.at[i, 'sentiment'] = "Neutral"
            
            # Optimize tasks
            optimized_tasks = task_optimizer(active_tasks)
            
            # Display top 5 tasks
            top_tasks = optimized_tasks.head(5)
            
            # Create a more visual display of top tasks
            for i, task in top_tasks.iterrows():
                with st.expander(f"{task['title']} - Priority: {task['priority']}"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"**Description:** {task['description']}")
                        st.write(f"**Assigned to:** {task['assigned_to']}")
                        if 'estimated_hours' in task and task['estimated_hours']:
                            st.write(f"**Estimated time:** {task['estimated_hours']:.1f} hours")
                    with col2:
                        st.write(f"**Deadline:** {task['deadline']}")
                        st.write(f"**Sentiment:** {task['sentiment']}")
                        st.write(f"**Status:** {task['status']}")
                        
                        # Calculate progress percentage based on status
                        status_map = {'Not Started': 0, 'In Progress': 50, 'Completed': 100}
                        progress = status_map.get(task['status'], 0)
                        st.progress(progress/100)
        else:
            st.info("No active tasks found. Add some tasks in the Task Management tab!")
    else:
        st.info("No tasks found. Add some tasks in the Task Management tab!")
    
    # Chart showing task distribution by priority
    if not tasks_df.empty and 'priority' in tasks_df.columns:
        st.subheader("Task Distribution by Priority")
        
        # Calculate counts
        priority_counts = tasks_df['priority'].value_counts()
        
        # Create two columns layout
        col1, col2 = st.columns(2)
        
        # Pie chart for priority distribution
        with col1:
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.pie(priority_counts, labels=priority_counts.index, autopct='%1.1f%%', 
                   startangle=90, shadow=True)
            ax.axis('equal')
            st.pyplot(fig)
        
        # Bar chart for sentiment distribution
        with col2:
            if 'sentiment' in tasks_df.columns:
                sentiment_counts = tasks_df['sentiment'].value_counts()
                fig, ax = plt.subplots(figsize=(6, 6))
                sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, ax=ax)
                ax.set_title('Task Distribution by Sentiment')
                ax.set_ylabel('Count')
                ax.set_xlabel('Sentiment')
                st.pyplot(fig)
            else:
                st.info("Sentiment data not available yet.")

# Task Management page
elif page == "Task Management":
    st.header("✅ Task Management")
    
    # Create tabs for different task management functions
    tab1, tab2, tab3 = st.tabs(["Add Task", "View & Edit Tasks", "Complete Tasks"])
    
    # Add Task tab
    with tab1:
        st.subheader("Add New Task")
        
        with st.form("new_task_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                task_title = st.text_input("Task Title*")
                task_description = st.text_area("Task Description")
                task_priority = st.selectbox("Priority*", ["Urgent", "High", "Medium", "Low"])
                task_type = st.selectbox("Task Type", ["Feature", "Bug Fix", "Documentation", "Other"])
            
            with col2:
                task_assigned = st.text_input("Assigned To")
                task_deadline = st.date_input("Deadline", value=datetime.datetime.now() + datetime.timedelta(days=7))
                task_status = st.selectbox("Status", ["Not Started", "In Progress", "Completed"])
            
            submit_button = st.form_submit_button("Add Task")
            
            if submit_button:
                if not task_title:
                    st.error("Task title is required!")
                else:
                    # Generate unique ID
                    task_id = len(tasks_df) + 1
                    
                    # Analyze sentiment if description provided
                    sentiment = "Neutral"
                    sentiment_score = 0.5
                    if task_description:
                        sentiment, confidence = predict_sentiment(task_description, sentiment_model, vectorizer)
                        # Convert sentiment to score (0-1 scale, higher = more urgent)
                        sentiment_score = 1 - confidence if sentiment == "Positive" else confidence
                    
                    # Predict completion time
                    new_task = {
                        'task_type': task_type.lower(),
                        'priority': task_priority.lower(),
                        'assigned_to': task_assigned
                    }
                    estimated_hours = completion_predictor(task_history_df, new_task)
                    
                    # Create new task row
                    new_task = {
                        'id': task_id,
                        'title': task_title,
                        'description': task_description,
                        'priority': task_priority,
                        'deadline': task_deadline.strftime('%Y-%m-%d'),
                        'status': task_status,
                        'assigned_to': task_assigned,
                        'task_type': task_type,
                        'created_date': datetime.datetime.now().strftime('%Y-%m-%d'),
                        'sentiment': sentiment,
                        'sentiment_score': sentiment_score,
                        'estimated_hours': estimated_hours
                    }
                    
                    # Add to dataframe
                    tasks_df.loc[len(tasks_df)] = new_task
                    
                    # Save updated data
                    save_task_data(tasks_df)
                    
                    st.success(f"Task '{task_title}' added successfully! Estimated completion time: {estimated_hours:.1f} hours")
    
    # View & Edit Tasks tab
    with tab2:
        st.subheader("Task List")
        
        # Filter options
        col1, col2, col3 = st.columns(3)
        with col1:
            filter_status = st.multiselect("Filter by Status", 
                                          ["Not Started", "In Progress", "Completed"],
                                          default=["Not Started", "In Progress"])
        with col2:
            filter_priority = st.multiselect("Filter by Priority",
                                            ["Urgent", "High", "Medium", "Low"])
        with col3:
            filter_assigned = st.text_input("Filter by Assigned Person")
        
        # Apply filters
        filtered_df = tasks_df.copy()
        
        if filter_status:
            filtered_df = filtered_df[filtered_df['status'].isin(filter_status)]
        
        if filter_priority:
            filtered_df = filtered_df[filtered_df['priority'].isin(filter_priority)]
            
        if filter_assigned:
            filtered_df = filtered_df[filtered_df['assigned_to'].str.contains(filter_assigned, case=False, na=False)]
        
        # Display filtered tasks with edit option
        if not filtered_df.empty:
            for i, task in filtered_df.iterrows():
                with st.expander(f"{task['title']} - {task['status']}"):
                    # Create a form for each task to handle edits
                    with st.form(f"edit_task_{task['id']}"):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            edited_title = st.text_input("Task Title", value=task['title'])
                            edited_description = st.text_area("Description", value=task['description'])
                            edited_priority = st.selectbox("Priority", 
                                                        ["Urgent", "High", "Medium", "Low"], 
                                                        index=["Urgent", "High", "Medium", "Low"].index(task['priority']) if task['priority'] in ["Urgent", "High", "Medium", "Low"] else 0)
                            edited_type = st.selectbox("Task Type", 
                                                    ["Feature", "Bug Fix", "Documentation", "Other"],
                                                    index=["Feature", "Bug Fix", "Documentation", "Other"].index(task['task_type']) if task['task_type'] in ["Feature", "Bug Fix", "Documentation", "Other"] else 0)
                        
                        with col2:
                            edited_assigned = st.text_input("Assigned To", value=task['assigned_to'])
                            edited_deadline = st.date_input("Deadline", 
                                                          value=pd.to_datetime(task['deadline']) if pd.notna(task['deadline']) else datetime.datetime.now())
                            edited_status = st.selectbox("Status", 
                                                        ["Not Started", "In Progress", "Completed"],
                                                        index=["Not Started", "In Progress", "Completed"].index(task['status']) if task['status'] in ["Not Started", "In Progress", "Completed"] else 0)
                        
                        submit_edit = st.form_submit_button("Save Changes")
                        
                        if submit_edit:
                            # Re-analyze sentiment if description changed
                            sentiment = task['sentiment']
                            sentiment_score = task['sentiment_score']
                            if edited_description != task['description']:
                                sentiment, confidence = predict_sentiment(edited_description, sentiment_model, vectorizer)
                                # Convert sentiment to score (0-1 scale, higher = more urgent)
                                sentiment_score = 1 - confidence if sentiment == "Positive" else confidence
                            
                            # Update task data
                            tasks_df.at[i, 'title'] = edited_title
                            tasks_df.at[i, 'description'] = edited_description
                            tasks_df.at[i, 'priority'] = edited_priority
                            tasks_df.at[i, 'deadline'] = edited_deadline.strftime('%Y-%m-%d')
                            tasks_df.at[i, 'status'] = edited_status
                            tasks_df.at[i, 'assigned_to'] = edited_assigned
                            tasks_df.at[i, 'task_type'] = edited_type
                            tasks_df.at[i, 'sentiment'] = sentiment
                            tasks_df.at[i, 'sentiment_score'] = sentiment_score
                            
                            # Save updated data
                            save_task_data(tasks_df)
                            
                            st.success("Task updated successfully!")
                            st.experimental_rerun()
                    
                    # Delete button outside the form
                    if st.button(f"Delete Task {task['id']}"):
                        tasks_df = tasks_df[tasks_df['id'] != task['id']].reset_index(drop=True)
                        save_task_data(tasks_df)
                        st.success("Task deleted successfully!")
                        st.experimental_rerun()
        else:
            st.info("No tasks match the current filters.")
    
    # Complete Tasks tab
    with tab3:
        st.subheader("Mark Tasks as Completed")
        
        # Filter to show only non-completed tasks
        incomplete_tasks = tasks_df[tasks_df['status'] != 'Completed'].copy()
        
        if not incomplete_tasks.empty:
            for i, task in incomplete_tasks.iterrows():
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.write(f"**{task['title']}** - Priority: {task['priority']}, Deadline: {task['deadline']}")
                
                with col2:
                    if st.button(f"Complete #{task['id']}"):
                        # Update task status
                        tasks_df.at[i, 'status'] = 'Completed'
                        
                        # Add to task history
                        actual_hours = st.session_state.get(f'actual_hours_{task["id"]}', 4.0)  # Default to 4 hours if not set
                        
                        new_history_entry = {
                            'id': task['id'],
                            'title': task['title'],
                            'task_type': task['task_type'],
                            'priority': task['priority'],
                            'assigned_to': task['assigned_to'],
                            'actual_completion_time': actual_hours,
                            'completed_date': datetime.datetime.now().strftime('%Y-%m-%d')
                        }
                        
                        # Add entry to history
                        task_history_df.loc[len(task_history_df)] = new_history_entry
                        
                        # Save updated data
                        save_task_data(tasks_df)
                        save_task_history(task_history_df)
                        
                        st.success(f"Task '{task['title']}' marked as completed!")
                        st.experimental_rerun()
                
                # Input for actual completion time
                col1, col2 = st.columns([3, 1])
                with col1:
                    key = f'actual_hours_{task["id"]}'
                    if key not in st.session_state:
                        st.session_state[key] = 4.0
                    st.session_state[key] = st.slider(f"Actual hours spent on task #{task['id']}", 
                                                   min_value=0.5, max_value=40.0, 
                                                   value=st.session_state[key], 
                                                   step=0.5)
                st.divider()
        else:
            st.info("No incomplete tasks found. Great job!")

# Analytics page
elif page == "Analytics":
    st.header("📈 Analytics & Insights")
    
    tab1, tab2 = st.tabs(["Task Performance", "Sentiment Analysis"])
    
    # Task Performance tab
    with tab1:
        st.subheader("Task Completion Performance")
        
        if not task_history_df.empty:
            # Chart 1: Average completion time by task type
            st.write("#### Average Completion Time by Task Type")
            avg_by_type = task_history_df.groupby('task_type')['actual_completion_time'].mean().reset_index()
            
            if not avg_by_type.empty:
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.barplot(x='task_type', y='actual_completion_time', data=avg_by_type, ax=ax)
                ax.set_title('Average Completion Time by Task Type')
                ax.set_ylabel('Hours')
                ax.set_xlabel('Task Type')
                st.pyplot(fig)
            
            # Chart 2: Completion time by priority
            st.write("#### Completion Time by Priority")
            avg_by_priority = task_history_df.groupby('priority')['actual_completion_time'].mean().reset_index()
            priority_order = ['Urgent', 'High', 'Medium', 'Low']
            
            if not avg_by_priority.empty:
                # Sort by custom priority order
                avg_by_priority['priority'] = pd.Categorical(
                    avg_by_priority['priority'], 
                    categories=priority_order, 
                    ordered=True
                )
                avg_by_priority = avg_by_priority.sort_values('priority')
                
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.barplot(x='priority', y='actual_completion_time', data=avg_by_priority, ax=ax)
                ax.set_title('Average Completion Time by Priority')
                ax.set_ylabel('Hours')
                ax.set_xlabel('Priority')
                st.pyplot(fig)
                
            # Chart 3: Task completion over time
            if 'completed_date' in task_history_df.columns:
                st.write("#### Task Completion Timeline")
                
                # Convert dates to datetime
                task_history_df['completed_date'] = pd.to_datetime(task_history_df['completed_date'], errors='coerce')
                
                # Group by date and count
                completed_by_date = task_history_df.groupby(task_history_df['completed_date'].dt.date).size().reset_index(name='count')
                completed_by_date.columns = ['date', 'completed_tasks']
                
                # Sort by date
                completed_by_date = completed_by_date.sort_values('date')
                
                # Create line chart
                fig, ax = plt.subplots(figsize=(12, 6))
                ax.plot(completed_by_date['date'], completed_by_date['completed_tasks'], marker='o')
                ax.set_title('Tasks Completed Over Time')
                ax.set_ylabel('Number of Tasks')
                ax.set_xlabel('Date')
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(fig)
            
            # Table: Individual performance
            st.write("#### Individual Performance")
            if 'assigned_to' in task_history_df.columns:
                person_stats = task_history_df.groupby('assigned_to').agg({
                    'id': 'count',
                    'actual_completion_time': 'mean'
                }).reset_index()
                
                person_stats.columns = ['Person', 'Tasks Completed', 'Avg. Hours per Task']
                person_stats = person_stats.sort_values('Tasks Completed', ascending=False)
                
                st.dataframe(person_stats, use_container_width=True)
        else:
            st.info("No task history data available yet. Complete some tasks to see analytics!")
    
    # Sentiment Analysis tab
    with tab2:
        st.subheader("Task Description Sentiment Analysis")
        
        if not tasks_df.empty and 'sentiment' in tasks_df.columns:
            # Sentiment distribution
            st.write("#### Sentiment Distribution Across Tasks")
            
            sentiment_counts = tasks_df['sentiment'].value_counts().reset_index()
            sentiment_counts.columns = ['Sentiment', 'Count']
            
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.barplot(x='Sentiment', y='Count', data=sentiment_counts, ax=ax)
            ax.set_title('Task Distribution by Sentiment')
            st.pyplot(fig)
            
            # Relationship between sentiment and priority
            st.write("#### Sentiment vs. Priority")
            
            # Create a cross-tabulation
            if 'priority' in tasks_df.columns:
                sentiment_priority = pd.crosstab(tasks_df['sentiment'], tasks_df['priority'])
                
                fig, ax = plt.subplots(figsize=(10, 6))
                sentiment_priority.plot(kind='bar', stacked=True, ax=ax)
                ax.set_title('Sentiment Distribution by Priority')
                ax.set_ylabel('Number of Tasks')
                ax.legend(title='Priority')
                st.pyplot(fig)
            
            # Sentiment score distribution
            if 'sentiment_score' in tasks_df.columns:
                st.write("#### Sentiment Score Distribution")
                
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.histplot(tasks_df['sentiment_score'], bins=20, kde=True, ax=ax)
                ax.set_title('Distribution of Sentiment Scores')
                ax.set_xlabel('Sentiment Score (Higher = More Negative/Urgent)')
                ax.set_ylabel('Frequency')
                st.pyplot(fig)
            
            # Test your own text sentiment
            st.write("#### Test Sentiment Analysis")
            st.write("Enter a task description to analyze its sentiment:")
            
            test_text = st.text_area("Task description", "", key="sentiment_test")
            
            if test_text:
                sentiment, confidence = predict_sentiment(test_text, sentiment_model, vectorizer)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**Detected Sentiment:** {sentiment}")
                    st.write(f"**Confidence:** {confidence:.2f}")
                
                with col2:
                    # Create a gauge chart to visualize sentiment
                    if sentiment == "Positive":
                        color = "green"
                        gauge_value = confidence
                    else:
                        color = "red"
                        gauge_value = confidence
                    
                    fig, ax = plt.subplots(figsize=(4, 4), subplot_kw=dict(polar=True))
                    
                    # Create gauge chart
                    theta = np.linspace(0, np.pi, 100)
                    r = np.ones_like(theta)
                    ax.plot(theta, r, color='lightgray', linewidth=10, alpha=0.5)
                    gauge_theta = np.linspace(0, np.pi * gauge_value, 100)
                    gauge_r = np.ones_like(gauge_theta)
                    ax.plot(gauge_theta, gauge_r, color=color, linewidth=10, alpha=0.8)
                    
                    # Remove axis ticks and labels
                    ax.set_xticks([])
                    ax.set_yticks([])
                    ax.spines['polar'].set_visible(False)
                    
                    # Add confidence text in the middle
                    ax.text(np.pi/2, 0, f"{confidence:.2f}", ha='center', va='center', fontsize=24)
                    
                    st.pyplot(fig)
                
                # Provide feedback on the sentiment analysis
                if sentiment == "Positive":
                    st.write("This task description has a positive sentiment, suggesting it might be engaging or enjoyable to work on. It may be given lower urgency in prioritization.")
                else:
                    st.write("This task description has a negative sentiment, suggesting it might be challenging, urgent, or stress-inducing. It may be given higher urgency in prioritization.")
        else:
            st.info("No sentiment data available yet. Add tasks with descriptions to see sentiment analysis!")

# Settings page
elif page == "Settings":
    st.header("⚙️ Settings")
    
    # AI Model Settings
    st.subheader("AI Model Settings")
    
    # Task Prioritization Settings
    st.write("#### Task Prioritization Weights")
    st.write("Adjust the importance of different factors in task prioritization:")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        deadline_weight = st.slider("Deadline Proximity Weight", 0.1, 1.0, 0.4, 0.1)
    with col2:
        priority_weight = st.slider("Manual Priority Weight", 0.1, 1.0, 0.4, 0.1)
    with col3:
        sentiment_weight = st.slider("Sentiment Analysis Weight", 0.1, 1.0, 0.2, 0.1)
    
    # Ensure weights sum to 1.0
    total_weight = deadline_weight + priority_weight + sentiment_weight
    normalized_deadline = deadline_weight / total_weight
    normalized_priority = priority_weight / total_weight
    normalized_sentiment = sentiment_weight / total_weight
    
    st.write(f"Normalized weights: Deadline ({normalized_deadline:.2f}), Priority ({normalized_priority:.2f}), Sentiment ({normalized_sentiment:.2f})")
    
    # Update task optimizer function with new weights
    def update_optimizer():
        def optimize_tasks(tasks_df):
            # Create a copy to avoid modifying the original DataFrame
            df = tasks_df.copy()
            
            # Calculate days to deadline
            if 'deadline' in df.columns:
                try:
                    df['days_to_deadline'] = pd.to_datetime(df['deadline'], errors='coerce') - pd.Timestamp.now()
                    df['days_to_deadline'] = df['days_to_deadline'].dt.total_seconds() / (24 * 3600)
                    
                    # Handle missing or invalid dates
                    df['days_to_deadline'] = df['days_to_deadline'].fillna(30)  # Default to 30 days
                    
                    # Normalize deadline score (closer deadline = higher score)
                    max_days = max(df['days_to_deadline'].max(), 30)  # At least 30 days to avoid division by zero
                    df['deadline_score'] = 1 - (df['days_to_deadline'] / max_days).clip(0, 1)
                except Exception as e:
                    st.warning(f"Error calculating deadline score: {e}")
                    df['deadline_score'] = 0.5  # Default score
            else:
                df['deadline_score'] = 0.5  # Default score
                
            # Convert priority to numeric score
            priority_map = {'urgent': 1.0, 'high': 0.75, 'medium': 0.5, 'low': 0.25}
            if 'priority' in df.columns:
                df['priority_score'] = df['priority'].str.lower().map(priority_map).fillna(0.5)
            else:
                df['priority_score'] = 0.5  # Default score
                
            # Add sentiment score if available
            if 'sentiment_score' not in df.columns:
                df['sentiment_score'] = 0.5  # Default score
                
            # Calculate final optimization score with updated weights
            df['optimization_score'] = (
                df['deadline_score'] * normalized_deadline +
                df['priority_score'] * normalized_priority +
                df['sentiment_score'] * normalized_sentiment
            )
            
            # Sort tasks by optimization score
            return df.sort_values('optimization_score', ascending=False)
        
        return optimize_tasks
    
    if st.button("Save Weight Settings"):
        # Update the global optimizer function
        task_optimizer = update_optimizer()
        st.success("Prioritization weights updated successfully!")
    
    # Data Management Section
    st.subheader("Data Management")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Export Tasks Data"):
            # Create a download link for tasks data
            if not tasks_df.empty:
                csv = tasks_df.to_csv(index=False)
                st.download_button(
                    label="Download Tasks CSV",
                    data=csv,
                    file_name="task_management_data.csv",
                    mime="text/csv"
                )
            else:
                st.warning("No task data to export!")
    
    with col2:
        if st.button("Export Task History"):
            # Create a download link for task history
            if not task_history_df.empty:
                csv = task_history_df.to_csv(index=False)
                st.download_button(
                    label="Download Task History CSV",
                    data=csv,
                    file_name="task_history_data.csv",
                    mime="text/csv"
                )
            else:
                st.warning("No task history to export!")
    
    # Danger Zone
    st.subheader("Danger Zone", divider="red")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🧹 Clear All Tasks", type="primary"):
            confirm = st.checkbox("I understand this will delete all task data")
            if confirm:
                if st.button("Confirm Clear All Tasks", type="primary"):
                    tasks_df = pd.DataFrame(columns=[
                        'id', 'title', 'description', 'priority', 'deadline', 'status',
                        'assigned_to', 'task_type', 'created_date', 'sentiment', 'sentiment_score',
                        'estimated_hours'
                    ])
                    save_task_data(tasks_df)
                    st.success("All tasks cleared successfully!")
                    st.experimental_rerun()
    
    with col2:
        if st.button("🧹 Clear Task History", type="primary"):
            confirm = st.checkbox("I understand this will delete all task history data")
            if confirm:
                if st.button("Confirm Clear Task History", type="primary"):
                    task_history_df = pd.DataFrame(columns=[
                        'id', 'title', 'task_type', 'priority', 'assigned_to', 
                        'actual_completion_time', 'completed_date'
                    ])
                    save_task_history(task_history_df)
                    st.success("Task history cleared successfully!")
                    st.experimental_rerun()
    
    # About section
    st.subheader("About")
    st.write("""
    **AI Powered Task Management System**
    
    This application uses sentiment analysis and machine learning to help optimize your task management workflow.
    The system analyzes task descriptions for sentiment, prioritizes tasks based on multiple factors,
    and predicts completion times based on historical data.
    
    **Features:**
    - Sentiment analysis of task descriptions
    - Intelligent task prioritization
    - Task completion time prediction
    - Performance analytics and insights
    
    Built with Streamlit and scikit-learn.
    """)

# Footer
st.markdown("""
---
AI Powered Task Management System | Created with ❤️ and AI
""")
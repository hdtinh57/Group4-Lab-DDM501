"""
Airflow DAG for ML Training Pipeline.

This DAG orchestrates the movie rating prediction training pipeline:


Usage:
    Copy this file to your Airflow dags/ folder
    Access Airflow UI at http://localhost:8080
"""

from datetime import datetime, timedelta
import pickle
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.dummy import DummyOperator
default_args = {
    'owner': 'mlops-team',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'movie_rating_training',
    default_args=default_args,
    description='ML Training Pipeline for Movie Rating Prediction',
    schedule_interval='@weekly',  # Or '0 0 * * 0' for every Sunday
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=['ml', 'training', 'movie-rating'],
)

def load_data_task(**context):
    """
    Task 1: Load and prepare data.
    
    This function:
    """
    from pipeline.data_ingestion import load_and_split
    
    print("Loading data...")
    trainset, testset, stats = load_and_split()
    tmp_dir = '/tmp/airflow_ml_pipeline'
    os.makedirs(tmp_dir, exist_ok=True)
    
    with open(f'{tmp_dir}/trainset.pkl', 'wb') as f:
        pickle.dump(trainset, f)
    with open(f'{tmp_dir}/testset.pkl', 'wb') as f:
        pickle.dump(testset, f)
    context['ti'].xcom_push(key='data_stats', value=stats)
    context['ti'].xcom_push(key='data_path', value=tmp_dir)
    
    print(f"Data loaded: {stats['n_ratings']} ratings")
    return "Data loaded successfully"
def preprocess_data_task(**context):
    """
    Task 2: Preprocess and validate data.
    """
    
    from pipeline.preprocessing import preprocess_data

    tmp_dir = context['ti'].xcom_pull(key='data_path')
    if not tmp_dir:
        raise ValueError("Missing data_path from load_data task")
    with open(f'{tmp_dir}/trainset.pkl', 'rb') as data_file:
        trainset = pickle.load(data_file)
    with open(f'{tmp_dir}/testset.pkl', 'rb') as data_file:
        testset = pickle.load(data_file)

    report = preprocess_data(trainset, testset)
    context['ti'].xcom_push(key='preprocess_report', value=report)
    if not report['preprocessing_successful']:
        raise ValueError(f"Data validation failed: {report['trainset_validation']['issues']}")
    return "Preprocessing complete"
def train_model_task(**context):
    """
    Task 3: Train the model with MLflow tracking.
    
    Configuration:
    """
    
    from pipeline.training import setup_mlflow, train_model

    tmp_dir = context['ti'].xcom_pull(key='data_path')
    if not tmp_dir:
        raise ValueError("Missing data_path from load_data task")
    with open(f'{tmp_dir}/trainset.pkl', 'rb') as data_file:
        trainset = pickle.load(data_file)

    setup_mlflow()
    model, run_id = train_model(
        trainset,
        model_type='svd',
        run_name=f"airflow_run_{context['ds']}",
        n_factors=100,
        n_epochs=20,
    )
    with open(f'{tmp_dir}/model.pkl', 'wb') as model_file:
        pickle.dump(model, model_file)
    context['ti'].xcom_push(key='run_id', value=run_id)
    return f"Model trained. Run ID: {run_id}"
def evaluate_model_task(**context):
    """
    Task 4: Evaluate the trained model.
    """
    
    from pipeline.evaluation import evaluate_model

    tmp_dir = context['ti'].xcom_pull(key='data_path')
    run_id = context['ti'].xcom_pull(key='run_id')
    if not tmp_dir or not run_id:
        raise ValueError("Missing training artifacts or MLflow run ID")
    with open(f'{tmp_dir}/model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    with open(f'{tmp_dir}/testset.pkl', 'rb') as data_file:
        testset = pickle.load(data_file)

    metrics = evaluate_model(model, testset, run_id)
    context['ti'].xcom_push(key='metrics', value=metrics)
    return f"Evaluation complete. RMSE: {metrics['rmse']:.4f}"


def decide_registration(**context):
    """
    Branch task: Decide whether to register model based on performance.
    
    Returns 'register_model' if RMSE < 1.0, otherwise 'skip_registration'
    """
    metrics = context['ti'].xcom_pull(key='metrics')
    
    if metrics and metrics.get('rmse', float('inf')) < 1.0:
        return 'register_model'
    return 'skip_registration'


def register_model_task(**context):
    """
    Task 5: Register the best model.
    """
    from pipeline.registry import register_best_model
    
    result = register_best_model()
    print(f"Model registered: {result['model_name']} v{result['version']}")
    return result


def cleanup_task(**context):
    """
    Final task: Cleanup temporary files.
    """
    import shutil
    
    tmp_dir = context['ti'].xcom_pull(key='data_path')
    if tmp_dir and os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)
        print(f"Cleaned up: {tmp_dir}")
    
    return "Cleanup complete"
t_load_data = PythonOperator(
    task_id='load_data',
    python_callable=load_data_task,
    dag=dag,
)
t_preprocess = PythonOperator(
    task_id='preprocess_data',
    python_callable=preprocess_data_task,
    dag=dag,
)
t_train = PythonOperator(
    task_id='train_model',
    python_callable=train_model_task,
    dag=dag,
)
t_evaluate = PythonOperator(
    task_id='evaluate_model',
    python_callable=evaluate_model_task,
    dag=dag,
)
t_decide = BranchPythonOperator(
    task_id='decide_registration',
    python_callable=decide_registration,
    dag=dag,
)
t_register = PythonOperator(
    task_id='register_model',
    python_callable=register_model_task,
    dag=dag,
)
t_skip = DummyOperator(
    task_id='skip_registration',
    dag=dag,
)
t_cleanup = PythonOperator(
    task_id='cleanup',
    python_callable=cleanup_task,
    trigger_rule='none_failed',  # Run even if branch skipped
    dag=dag,
)

t_load_data >> t_preprocess >> t_train >> t_evaluate >> t_decide
t_decide >> [t_register, t_skip]
[t_register, t_skip] >> t_cleanup

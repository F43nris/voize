#!/usr/bin/env python3
"""
Minimal DAG-based Scheduler for ML Pipeline
===========================================
A lightweight DAG scheduler that demonstrates workflow orchestration
without requiring a full Airflow installation.

Features:
- Task dependencies and ordering
- Retry logic and error handling  
- Scheduling capabilities
- Integration with existing pipeline components
"""

import time
import logging
import json
import traceback
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import List, Callable, Any, Optional, Dict
from enum import Enum
import schedule
import threading
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running" 
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"

@dataclass
class TaskRun:
    """Represents a single task execution"""
    task_id: str
    status: TaskStatus = TaskStatus.PENDING
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    error_message: Optional[str] = None
    retry_count: int = 0
    output: Any = None

@dataclass
class Task:
    """Represents a DAG task"""
    task_id: str
    func: Callable
    dependencies: List[str] = field(default_factory=list)
    max_retries: int = 2
    retry_delay: int = 60  # seconds
    timeout: int = 3600   # seconds
    
    def can_run(self, completed_tasks: set) -> bool:
        """Check if all dependencies are completed"""
        return all(dep in completed_tasks for dep in self.dependencies)
    
    def execute(self, context: Dict = None) -> TaskRun:
        """Execute the task with retry logic"""
        run = TaskRun(task_id=self.task_id)
        context = context or {}
        
        for attempt in range(self.max_retries + 1):
            try:
                logger.info(f"🚀 Starting task: {self.task_id} (attempt {attempt + 1})")
                run.start_time = datetime.now()
                run.status = TaskStatus.RUNNING
                run.retry_count = attempt
                
                # Execute the function
                result = self.func(**context)
                
                run.end_time = datetime.now()
                run.status = TaskStatus.SUCCESS
                run.output = result
                
                duration = (run.end_time - run.start_time).total_seconds()
                logger.info(f"✅ Task {self.task_id} completed in {duration:.1f}s")
                return run
                
            except Exception as e:
                run.error_message = str(e)
                logger.error(f"❌ Task {self.task_id} failed (attempt {attempt + 1}): {e}")
                
                if attempt < self.max_retries:
                    logger.info(f"⏳ Retrying in {self.retry_delay}s...")
                    time.sleep(self.retry_delay)
                else:
                    run.end_time = datetime.now()
                    run.status = TaskStatus.FAILED
                    logger.error(f"💥 Task {self.task_id} failed after {self.max_retries + 1} attempts")
                    
        return run

class DAG:
    """Represents a Directed Acyclic Graph of tasks"""
    
    def __init__(self, dag_id: str, description: str = "", schedule: str = None):
        self.dag_id = dag_id
        self.description = description
        self.schedule = schedule
        self.tasks: Dict[str, Task] = {}
        self.runs: List[Dict] = []
        
    def add_task(self, task: Task):
        """Add a task to the DAG"""
        self.tasks[task.task_id] = task
        
    def validate(self) -> bool:
        """Validate DAG structure (no cycles, valid dependencies)"""
        # Simple cycle detection using DFS
        def has_cycle(task_id, visiting, visited):
            if task_id in visiting:
                return True
            if task_id in visited:
                return False
                
            visiting.add(task_id)
            for dep in self.tasks[task_id].dependencies:
                if dep not in self.tasks:
                    raise ValueError(f"Task {task_id} depends on non-existent task {dep}")
                if has_cycle(dep, visiting, visited):
                    return True
            visiting.remove(task_id)
            visited.add(task_id)
            return False
        
        visited = set()
        for task_id in self.tasks:
            if task_id not in visited:
                if has_cycle(task_id, set(), visited):
                    return False
        return True
    
    def get_execution_order(self) -> List[str]:
        """Get tasks in topological order"""
        in_degree = {task_id: 0 for task_id in self.tasks}
        
        # Calculate in-degrees
        for task in self.tasks.values():
            for dep in task.dependencies:
                in_degree[task.task_id] += 1
        
        # Topological sort
        queue = [task_id for task_id, degree in in_degree.items() if degree == 0]
        result = []
        
        while queue:
            task_id = queue.pop(0)
            result.append(task_id)
            
            # Reduce in-degree for dependent tasks
            for other_task_id, task in self.tasks.items():
                if task_id in task.dependencies:
                    in_degree[other_task_id] -= 1
                    if in_degree[other_task_id] == 0:
                        queue.append(other_task_id)
        
        return result
    
    def run(self, context: Dict = None) -> Dict:
        """Execute the entire DAG"""
        if not self.validate():
            raise ValueError(f"DAG {self.dag_id} contains cycles")
        
        logger.info(f"🎯 Starting DAG execution: {self.dag_id}")
        start_time = datetime.now()
        
        context = context or {}
        context['dag_id'] = self.dag_id
        context['run_date'] = start_time
        
        task_runs = {}
        completed_tasks = set()
        failed_tasks = set()
        
        execution_order = self.get_execution_order()
        
        for task_id in execution_order:
            task = self.tasks[task_id]
            
            # Check if we can run (dependencies completed)
            if not task.can_run(completed_tasks):
                # Skip if dependencies failed
                if any(dep in failed_tasks for dep in task.dependencies):
                    logger.warning(f"⏭️  Skipping {task_id} due to failed dependencies")
                    task_runs[task_id] = TaskRun(task_id=task_id, status=TaskStatus.SKIPPED)
                    continue
                else:
                    raise RuntimeError(f"Task {task_id} cannot run - dependency issue")
            
            # Execute task
            run = task.execute(context)
            task_runs[task_id] = run
            
            if run.status == TaskStatus.SUCCESS:
                completed_tasks.add(task_id)
                # Add task output to context for downstream tasks
                context[f'{task_id}_output'] = run.output
            else:
                failed_tasks.add(task_id)
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Summarize results
        success_count = sum(1 for run in task_runs.values() if run.status == TaskStatus.SUCCESS)
        failed_count = sum(1 for run in task_runs.values() if run.status == TaskStatus.FAILED)
        skipped_count = sum(1 for run in task_runs.values() if run.status == TaskStatus.SKIPPED)
        
        dag_status = "SUCCESS" if failed_count == 0 else "FAILED"
        
        logger.info(f"🏁 DAG {self.dag_id} completed: {dag_status}")
        logger.info(f"   Duration: {duration:.1f}s")
        logger.info(f"   Tasks: {success_count} success, {failed_count} failed, {skipped_count} skipped")
        
        run_summary = {
            'dag_id': self.dag_id,
            'status': dag_status,
            'start_time': start_time,
            'end_time': end_time,
            'duration_seconds': duration,
            'task_runs': task_runs,
            'success_count': success_count,
            'failed_count': failed_count,
            'skipped_count': skipped_count
        }
        
        self.runs.append(run_summary)
        return run_summary

class DAGScheduler:
    """Simple scheduler for running DAGs on schedule"""
    
    def __init__(self):
        self.dags: Dict[str, DAG] = {}
        self.running = False
        self.scheduler_thread = None
        
    def register_dag(self, dag: DAG):
        """Register a DAG with the scheduler"""
        self.dags[dag.dag_id] = dag
        
        if dag.schedule:
            self._schedule_dag(dag)
            
    def _schedule_dag(self, dag: DAG):
        """Set up scheduling for a DAG"""
        def run_dag():
            try:
                logger.info(f"📅 Scheduled execution of DAG: {dag.dag_id}")
                result = dag.run()
                return result
            except Exception as e:
                logger.error(f"💥 Scheduled DAG {dag.dag_id} failed: {e}")
                return None
        
        # Parse schedule string and set up with schedule library
        if dag.schedule == "daily":
            schedule.every().day.at("02:00").do(run_dag)
        elif dag.schedule == "weekly":
            schedule.every().monday.at("02:00").do(run_dag)
        elif dag.schedule.startswith("every"):
            # e.g., "every 6 hours"
            parts = dag.schedule.split()
            if len(parts) >= 3:
                interval = int(parts[1])
                unit = parts[2].rstrip('s')  # Remove 's' from 'hours'
                
                if unit == "hour":
                    schedule.every(interval).hours.do(run_dag)
                elif unit == "minute":
                    schedule.every(interval).minutes.do(run_dag)
        
        logger.info(f"📅 Scheduled DAG {dag.dag_id} with schedule: {dag.schedule}")
    
    def start(self):
        """Start the scheduler"""
        if self.running:
            logger.warning("Scheduler already running")
            return
            
        self.running = True
        
        def scheduler_loop():
            logger.info("🚀 DAG Scheduler started")
            while self.running:
                schedule.run_pending()
                time.sleep(1)
            logger.info("🛑 DAG Scheduler stopped")
        
        self.scheduler_thread = threading.Thread(target=scheduler_loop, daemon=True)
        self.scheduler_thread.start()
    
    def stop(self):
        """Stop the scheduler"""
        self.running = False
        if self.scheduler_thread:
            self.scheduler_thread.join(timeout=5)
    
    def run_dag_now(self, dag_id: str, context: Dict = None) -> Dict:
        """Run a specific DAG immediately"""
        if dag_id not in self.dags:
            raise ValueError(f"DAG {dag_id} not found")
        
        return self.dags[dag_id].run(context)
    
    def get_dag_status(self, dag_id: str) -> Dict:
        """Get the status and history of a DAG"""
        if dag_id not in self.dags:
            raise ValueError(f"DAG {dag_id} not found")
        
        dag = self.dags[dag_id]
        return {
            'dag_id': dag_id,
            'description': dag.description,
            'schedule': dag.schedule,
            'task_count': len(dag.tasks),
            'runs': dag.runs[-10:],  # Last 10 runs
            'last_run': dag.runs[-1] if dag.runs else None
        }

# Export main classes for use in other modules
__all__ = ['Task', 'DAG', 'DAGScheduler', 'TaskStatus'] 
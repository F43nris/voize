# ... existing code ...

# Add DAG scheduler imports and setup
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / "data-pipeline"))

try:
    from ml_dags import setup_ml_scheduler
    from dag_scheduler import DAGScheduler
    
    # Initialize the scheduler
    dag_scheduler = setup_ml_scheduler()
    
except ImportError as e:
    logger.warning(f"DAG scheduler not available: {e}")
    dag_scheduler = None

# ... existing code after the health and predict endpoints ...

@app.get("/dags")
async def list_dags():
    """List all available DAGs and their status"""
    if not dag_scheduler:
        raise HTTPException(status_code=503, detail="DAG scheduler not available")
    
    try:
        dags_info = []
        for dag_id in dag_scheduler.dags:
            dag_status = dag_scheduler.get_dag_status(dag_id)
            dags_info.append(dag_status)
        
        return {
            "dags": dags_info,
            "scheduler_running": dag_scheduler.running,
            "total_dags": len(dag_scheduler.dags)
        }
    except Exception as e:
        logger.error(f"Error listing DAGs: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/dags/{dag_id}/status")
async def get_dag_status(dag_id: str):
    """Get detailed status of a specific DAG"""
    if not dag_scheduler:
        raise HTTPException(status_code=503, detail="DAG scheduler not available")
    
    try:
        return dag_scheduler.get_dag_status(dag_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error getting DAG status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/dags/{dag_id}/run")
async def trigger_dag(dag_id: str):
    """Manually trigger a DAG execution"""
    if not dag_scheduler:
        raise HTTPException(status_code=503, detail="DAG scheduler not available")
    
    try:
        result = dag_scheduler.run_dag_now(dag_id)
        return {
            "message": f"DAG {dag_id} triggered successfully",
            "run_result": {
                "status": result["status"],
                "duration_seconds": result["duration_seconds"],
                "success_count": result["success_count"],
                "failed_count": result["failed_count"],
                "start_time": result["start_time"].isoformat(),
                "end_time": result["end_time"].isoformat()
            }
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error running DAG: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/scheduler/start")
async def start_scheduler():
    """Start the DAG scheduler"""
    if not dag_scheduler:
        raise HTTPException(status_code=503, detail="DAG scheduler not available")
    
    try:
        dag_scheduler.start()
        return {"message": "DAG scheduler started", "running": dag_scheduler.running}
    except Exception as e:
        logger.error(f"Error starting scheduler: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/scheduler/stop")
async def stop_scheduler():
    """Stop the DAG scheduler"""
    if not dag_scheduler:
        raise HTTPException(status_code=503, detail="DAG scheduler not available")
    
    try:
        dag_scheduler.stop()
        return {"message": "DAG scheduler stopped", "running": dag_scheduler.running}
    except Exception as e:
        logger.error(f"Error stopping scheduler: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ... existing code ... 
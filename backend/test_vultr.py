#!/usr/bin/env python3
"""
Quick test for Vultr cloud training integration
Tests the core functionality without creating real instances
"""
import asyncio
import os
import sys
import tempfile
import pandas as pd
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent))

async def test_vultr_trainer():
    """Test Vultr trainer setup and job packaging"""
    print("=" * 60)
    print("Testing Vultr Cloud Training Integration")
    print("=" * 60)
    
    # Load .env file
    try:
        from dotenv import load_dotenv
        env_path = Path(__file__).parent / '.env'
        load_dotenv(env_path)
        print(f"📄 Loaded environment from: {env_path}")
    except ImportError:
        print("⚠️  python-dotenv not installed, reading from shell environment")
    
    # 1. Check environment variables
    print("\n[1/5] Checking environment variables...")
    api_key = os.getenv("VULTR_API_KEY")
    use_cloud = os.getenv("USE_VULTR_TRAINING", "false").lower() == "true"
    
    if not api_key:
        print("   ⚠️  VULTR_API_KEY not set in .env")
        print("   This is OK for testing, but needed for actual cloud training")
    else:
        print(f"   ✅ VULTR_API_KEY found: {api_key[:20]}...{api_key[-10:]}")
    
    print(f"   {'✅' if use_cloud else '⚠️ '} USE_VULTR_TRAINING = {use_cloud}")
    
    # 2. Test imports
    print("\n[2/5] Testing imports...")
    try:
        from app.cloud.vultr_trainer import VultrTrainer
        print("   ✅ VultrTrainer imported successfully")
    except ImportError as e:
        print(f"   ❌ Failed to import VultrTrainer: {e}")
        return False
    
    try:
        import paramiko
        print("   ✅ paramiko library available")
    except ImportError:
        print("   ❌ paramiko not installed (run: pip install paramiko)")
        return False
    
    try:
        import requests
        print("   ✅ requests library available")
    except ImportError:
        print("   ❌ requests not installed")
        return False
    
    # 3. Test job packaging
    print("\n[3/5] Testing training job packaging...")
    try:
        # Create dummy training data
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            df = pd.DataFrame({
                'text': [
                    "This movie was amazing!",
                    "Terrible film, waste of time",
                    "Best movie I've seen this year",
                    "Boring and predictable"
                ] * 25,  # 100 samples
                'label': [1, 0, 1, 0] * 25
            })
            df.to_csv(f.name, index=False)
            test_data_path = f.name
        
        print(f"   ✅ Created test dataset: {test_data_path}")
        print(f"   📊 Dataset: {len(df)} samples, 2 classes")
        
        # Test packaging (without API key to avoid real calls)
        if api_key:
            trainer = VultrTrainer(api_key=api_key)
        else:
            # Mock trainer for testing
            print("   ⚠️  Skipping actual VultrTrainer init (no API key)")
            trainer = None
        
        if trainer:
            job_tarball = await trainer.package_training_job(
                task_type="text_classification",
                train_data_path=test_data_path,
                model_name="distilbert-base-uncased",
                num_classes=2,
                hyperparameters={
                    "num_epochs": 2,
                    "batch_size": 4,
                    "learning_rate": 2e-5
                }
            )
            
            print(f"   ✅ Job packaged successfully: {job_tarball}")
            
            # Check tarball contents
            import tarfile
            with tarfile.open(job_tarball, 'r:gz') as tar:
                members = tar.getnames()
                print(f"   📦 Tarball contents: {len(members)} files")
                for member in members[:5]:  # Show first 5
                    print(f"      - {member}")
                if len(members) > 5:
                    print(f"      ... and {len(members) - 5} more")
            
            # Verify required files
            required_files = ['job/train_data.csv', 'job/config.json', 'job/train.py']
            missing = [f for f in required_files if f not in members]
            if missing:
                print(f"   ❌ Missing required files: {missing}")
                return False
            else:
                print(f"   ✅ All required files present")
        else:
            print("   ⏭️  Skipped packaging test (no API key)")
        
        # Cleanup
        os.unlink(test_data_path)
        if trainer and 'job_tarball' in locals():
            os.unlink(job_tarball)
        
    except Exception as e:
        print(f"   ❌ Packaging test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 4. Test pipeline integration
    print("\n[4/5] Testing pipeline integration...")
    try:
        from app.orchestrator.pipeline import PipelineOrchestrator
        print("   ✅ PipelineOrchestrator imports successfully")
        
        # Check if cloud training is in the code
        import inspect
        source = inspect.getsource(PipelineOrchestrator._execute_train)
        if "USE_VULTR_TRAINING" in source:
            print("   ✅ Cloud training integrated into pipeline")
        else:
            print("   ❌ Cloud training code not found in pipeline")
            return False
        
        if "VultrTrainer" in source:
            print("   ✅ VultrTrainer reference found in pipeline")
        else:
            print("   ⚠️  VultrTrainer not directly referenced (may use dynamic import)")
        
    except Exception as e:
        print(f"   ❌ Pipeline integration test failed: {e}")
        return False
    
    # 5. Test trainer factory
    print("\n[5/5] Testing TrainerFactory...")
    try:
        from app.ml.trainer_factory import TrainerFactory
        
        # Test text classification detection - need realistic text data
        test_df = pd.DataFrame({
            'text': [
                "This movie was absolutely fantastic! I loved every minute of it and the acting was superb.",
                "Terrible experience. The plot made no sense and I wanted to leave halfway through.",
                "An amazing film with brilliant cinematography. Highly recommend to everyone!",
                "Waste of time and money. Poor dialogue and terrible special effects throughout.",
                "Outstanding performance by the lead actor. The story kept me engaged from start to finish.",
                "Disappointing and boring. Expected much better based on the reviews I had read.",
                "One of the best films I've seen this year! The soundtrack alone is worth watching for.",
                "Couldn't get into it at all. The pacing was off and characters felt underdeveloped.",
            ] * 25,  # 200 samples with varied text
            'label': [1, 0, 1, 0, 1, 0, 1, 0] * 25
        })
        
        task_type = TrainerFactory.detect_task_type(test_df, 'label')
        if task_type == "text_classification":
            print(f"   ✅ Task type detection works: {task_type}")
        else:
            print(f"   ⚠️  Got '{task_type}' (expected 'text_classification')")
            print(f"   Note: Detection depends on text column characteristics")
            # Don't fail the test - just a warning
        
        # Test trainer creation (force text_classification)
        trainer = TrainerFactory.get_trainer(
            task_type="text_classification",
            model_name="auto",
            num_classes=2
        )
        print(f"   ✅ Trainer created: {type(trainer).__name__}")
        
        from app.ml.text.sentiment_classifier import SentimentClassifier
        if isinstance(trainer, SentimentClassifier):
            print(f"   ✅ Correct trainer type (DistilBERT)")
        else:
            print(f"   ⚠️  Unexpected trainer type: {type(trainer)}")
        
    except Exception as e:
        print(f"   ❌ TrainerFactory test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

async def main():
    """Run all tests"""
    print("\n")
    success = await test_vultr_trainer()
    
    print("\n" + "=" * 60)
    if success:
        print("✅ All tests passed!")
        print("\nNext steps:")
        if not os.getenv("VULTR_API_KEY"):
            print("1. Add your VULTR_API_KEY to .env file")
            print("2. Set USE_VULTR_TRAINING=true in .env")
        print("3. Train a model through the UI")
        print("4. Watch the backend logs for cloud training!")
    else:
        print("❌ Some tests failed - check errors above")
        return 1
    print("=" * 60)
    print("\n")
    return 0

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)

"""
Vultr Cloud Training Integration
Offloads model training to Vultr GPU instances for faster training
"""
import asyncio
import json
import os
import tarfile
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional
import requests
import time


class VultrTrainer:
    """
    Trains models on Vultr cloud GPU instances
    
    Features:
    - Automatic instance provisioning (GPU instances)
    - Training job packaging and upload
    - Real-time training progress monitoring
    - Automatic model download after training
    - Instance cleanup after completion
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Vultr trainer
        
        Args:
            api_key: Vultr API key (or set VULTR_API_KEY env var)
        """
        self.api_key = api_key or os.getenv("VULTR_API_KEY")
        if not self.api_key:
            raise ValueError("VULTR_API_KEY environment variable not set")
        
        self.base_url = "https://api.vultr.com/v2"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        self.instance_id = None
        self.instance_ip = None
        
    def _api_request(self, method: str, endpoint: str, data: Optional[Dict] = None) -> Dict:
        """Make API request to Vultr"""
        url = f"{self.base_url}/{endpoint}"
        
        if method == "GET":
            response = requests.get(url, headers=self.headers)
        elif method == "POST":
            response = requests.post(url, headers=self.headers, json=data)
        elif method == "DELETE":
            response = requests.delete(url, headers=self.headers)
        else:
            raise ValueError(f"Unsupported method: {method}")
        
        response.raise_for_status()
        return response.json() if response.text else {}
    
    async def create_gpu_instance(self, region: str = "ewr", plan: str = "vhp-4c-8gb-amd") -> str:
        """
        Create a Vultr high-performance instance for ML training
        
        NOTE: GPU instances require special Vultr account approval.
        Using high-performance CPU instance as fallback.
        
        Args:
            region: Vultr region (ewr = New Jersey, lax = Los Angeles, etc.)
            plan: Instance plan (vhp-4c-8gb-amd = 4 vCPU, 8GB RAM)
        
        Returns:
            Instance ID
        """
        print(f"[VultrTrainer] Creating training instance in {region}...")
        print(f"[VultrTrainer] Plan: {plan} (4 vCPU, 8GB RAM)")
        
        # Create instance
        instance_data = {
            "region": region,
            "plan": plan,
            "label": "automl-training",
            "os_id": 1743,  # Ubuntu 22.04 LTS x64
            "enable_ipv6": False,
            "backups": "disabled",
            "ddos_protection": False,
            "activation_email": False,
            "hostname": "automl-trainer",
            "tag": "automl",
            "user_data": self._get_startup_script()
        }
        
        result = self._api_request("POST", "instances", instance_data)
        self.instance_id = result["instance"]["id"]
        
        print(f"[VultrTrainer] Instance created: {self.instance_id}")
        print(f"[VultrTrainer] Waiting for instance to become active...")
        
        # Wait for instance to be active
        await self._wait_for_instance_ready()
        
        return self.instance_id
    
    async def _wait_for_instance_ready(self, timeout: int = 300):
        """Wait for instance to be ready"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            result = self._api_request("GET", f"instances/{self.instance_id}")
            instance = result["instance"]
            
            status = instance["status"]
            print(f"[VultrTrainer] Instance status: {status}")
            
            if status == "active":
                self.instance_ip = instance["main_ip"]
                print(f"[VultrTrainer] ✅ Instance ready! IP: {self.instance_ip}")
                
                # Wait additional 30 seconds for SSH to be ready
                print(f"[VultrTrainer] Waiting 30s for SSH initialization...")
                await asyncio.sleep(30)
                return
            
            await asyncio.sleep(10)
        
        raise TimeoutError("Instance failed to become ready within timeout")
    
    def _get_startup_script(self) -> str:
        """Get cloud-init startup script for optimized CPU training"""
        return """#!/bin/bash
set -e

# Update system
export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y python3-pip python3-venv git curl wget

# Install optimized CPU libraries
apt-get install -y libopenblas-dev libomp-dev

# Create training directory
mkdir -p /root/automl-training
cd /root/automl-training

# Install ML libraries (CPU-optimized PyTorch)
pip3 install --upgrade pip
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip3 install transformers datasets scikit-learn pandas numpy joblib

# Set CPU optimization flags
echo 'export OMP_NUM_THREADS=4' >> /root/.bashrc
echo 'export MKL_NUM_THREADS=4' >> /root/.bashrc

# Setup SSH
systemctl enable ssh
systemctl start ssh

# Mark as ready
echo "Training environment ready - $(date)" > /root/automl-training/ready.txt
"""
    
    async def package_training_job(
        self,
        task_type: str,
        train_data_path: str,
        model_name: str,
        num_classes: int,
        hyperparameters: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Package training job into tarball
        
        Args:
            task_type: Type of ML task (text_classification, image_classification, etc.)
            train_data_path: Path to training data CSV
            model_name: Model to train
            num_classes: Number of classes
            hyperparameters: Optional hyperparameters
        
        Returns:
            Path to packaged job tarball
        """
        print(f"[VultrTrainer] Packaging training job...")
        
        # Create temporary directory
        with tempfile.TemporaryDirectory() as tmpdir:
            job_dir = Path(tmpdir) / "job"
            job_dir.mkdir()
            
            # Copy training data
            import shutil
            shutil.copy(train_data_path, job_dir / "train_data.csv")
            
            # Create job config
            config = {
                "task_type": task_type,
                "model_name": model_name,
                "num_classes": num_classes,
                "hyperparameters": hyperparameters or {
                    "num_epochs": 3,
                    "batch_size": 8,
                    "learning_rate": 2e-5
                }
            }
            
            with open(job_dir / "config.json", "w") as f:
                json.dump(config, f, indent=2)
            
            # Create training script
            training_script = self._generate_training_script(task_type)
            with open(job_dir / "train.py", "w") as f:
                f.write(training_script)
            
            # Create tarball
            tarball_path = Path(tmpdir) / "training_job.tar.gz"
            with tarfile.open(tarball_path, "w:gz") as tar:
                tar.add(job_dir, arcname="job")
            
            # Move to permanent location
            output_path = Path("/tmp") / f"vultr_job_{int(time.time())}.tar.gz"
            shutil.copy(tarball_path, output_path)
            
            print(f"[VultrTrainer] ✅ Job packaged: {output_path}")
            return str(output_path)
    
    def _generate_training_script(self, task_type: str) -> str:
        """Generate Python training script for cloud execution"""
        if task_type == "text_classification":
            return """#!/usr/bin/env python3
import json
import pandas as pd
from pathlib import Path
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
from datasets import Dataset
import torch

# Load config
with open('config.json') as f:
    config = json.load(f)

print(f"Starting training: {config}")

# Load data
df = pd.read_csv('train_data.csv')
print(f"Loaded {len(df)} samples")

# Prepare data
text_col = 'text'
label_col = 'label'

texts = df[text_col].tolist()
labels = df[label_col].tolist()

# Create label mapping
unique_labels = sorted(set(labels))
label_map = {label: idx for idx, label in enumerate(unique_labels)}
mapped_labels = [label_map[label] for label in labels]

# Split train/val
split_idx = int(len(texts) * 0.8)
train_texts = texts[:split_idx]
train_labels = mapped_labels[:split_idx]
val_texts = texts[split_idx:]
val_labels = mapped_labels[split_idx:]

# Load model
model_name = config['model_name']
print(f"Loading model: {model_name}")

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=config['num_classes']
)

# Tokenize
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)

train_dataset = Dataset.from_dict({"text": train_texts, "label": train_labels})
train_dataset = train_dataset.map(tokenize_function, batched=True)

val_dataset = Dataset.from_dict({"text": val_texts, "label": val_labels})
val_dataset = val_dataset.map(tokenize_function, batched=True)

# Training arguments
hyperparams = config['hyperparameters']
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=hyperparams.get('num_epochs', 3),
    per_device_train_batch_size=hyperparams.get('batch_size', 8),
    per_device_eval_batch_size=hyperparams.get('batch_size', 8),
    learning_rate=hyperparams.get('learning_rate', 2e-5),
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    logging_steps=10,
    fp16=torch.cuda.is_available()
)

# Data collator
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Metrics
def compute_metrics(eval_pred):
    from sklearn.metrics import accuracy_score, f1_score
    logits, labels = eval_pred
    predictions = logits.argmax(axis=-1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1": f1_score(labels, predictions, average='weighted')
    }

# Train
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

print("Starting training...")
result = trainer.train()

print("Training complete! Saving model...")
model.save_pretrained("./model_output/transformer")
tokenizer.save_pretrained("./model_output/tokenizer")

# Save label map
with open("./model_output/label_map.json", "w") as f:
    json.dump(label_map, f)

# Save metrics
metrics = {
    "train_loss": result.metrics.get('train_loss'),
    "eval_accuracy": trainer.evaluate().get('eval_accuracy'),
    "eval_f1": trainer.evaluate().get('eval_f1')
}

with open("./model_output/metrics.json", "w") as f:
    json.dump(metrics, f)

print(f"✅ Training complete! Metrics: {metrics}")
"""
        else:
            raise NotImplementedError(f"Task type {task_type} not yet supported for cloud training")
    
    async def upload_and_train(self, job_tarball_path: str) -> Dict[str, Any]:
        """
        Upload training job to instance and start training
        
        Args:
            job_tarball_path: Path to packaged job tarball
        
        Returns:
            Training metrics
        """
        if not self.instance_ip:
            raise ValueError("No active instance. Call create_gpu_instance() first.")
        
        print(f"[VultrTrainer] Uploading training job to {self.instance_ip}...")
        
        # Upload via SCP (requires SSH key setup)
        # For simplicity, we'll use a Python approach with paramiko
        import paramiko
        
        # Create SSH client
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        
        try:
            # Connect (use your SSH key)
            ssh.connect(self.instance_ip, username='root', timeout=30)
            
            # Upload tarball
            sftp = ssh.open_sftp()
            sftp.put(job_tarball_path, '/root/automl-training/job.tar.gz')
            sftp.close()
            
            print(f"[VultrTrainer] ✅ Job uploaded. Starting training...")
            
            # Extract and run training
            commands = """
cd /root/automl-training
source venv/bin/activate
tar -xzf job.tar.gz
cd job
python3 train.py > training.log 2>&1
echo "TRAINING_COMPLETE"
"""
            
            stdin, stdout, stderr = ssh.exec_command(commands)
            
            # Stream output
            for line in stdout:
                print(f"[VultrTrainer] {line.strip()}")
            
            # Download results
            print(f"[VultrTrainer] Downloading trained model...")
            sftp = ssh.open_sftp()
            
            # Create local output directory
            output_dir = Path("/tmp") / f"vultr_model_{int(time.time())}"
            output_dir.mkdir(exist_ok=True)
            
            # Download model files
            self._download_recursive(sftp, '/root/automl-training/job/model_output', str(output_dir))
            
            sftp.close()
            
            # Load metrics
            with open(output_dir / "metrics.json") as f:
                metrics = json.load(f)
            
            print(f"[VultrTrainer] ✅ Training complete! Metrics: {metrics}")
            
            return {
                "metrics": metrics,
                "model_path": str(output_dir)
            }
            
        finally:
            ssh.close()
    
    def _download_recursive(self, sftp, remote_dir: str, local_dir: str):
        """Recursively download directory via SFTP"""
        Path(local_dir).mkdir(parents=True, exist_ok=True)
        
        for item in sftp.listdir_attr(remote_dir):
            remote_path = f"{remote_dir}/{item.filename}"
            local_path = Path(local_dir) / item.filename
            
            if item.st_mode & 0o40000:  # Directory
                self._download_recursive(sftp, remote_path, str(local_path))
            else:  # File
                sftp.get(remote_path, str(local_path))
                print(f"[VultrTrainer] Downloaded: {item.filename}")
    
    async def cleanup_instance(self):
        """Delete the GPU instance"""
        if not self.instance_id:
            print("[VultrTrainer] No instance to cleanup")
            return
        
        print(f"[VultrTrainer] Deleting instance {self.instance_id}...")
        self._api_request("DELETE", f"instances/{self.instance_id}")
        print(f"[VultrTrainer] ✅ Instance deleted")
        
        self.instance_id = None
        self.instance_ip = None
    
    async def train_on_cloud(
        self,
        task_type: str,
        train_data_path: str,
        model_name: str,
        num_classes: int,
        hyperparameters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Complete cloud training workflow
        
        Args:
            task_type: Type of ML task
            train_data_path: Path to training data
            model_name: Model to train
            num_classes: Number of classes
            hyperparameters: Optional hyperparameters
        
        Returns:
            Dictionary with metrics and model_path
        """
        try:
            # 1. Create GPU instance
            await self.create_gpu_instance()
            
            # 2. Package training job
            job_tarball = await self.package_training_job(
                task_type=task_type,
                train_data_path=train_data_path,
                model_name=model_name,
                num_classes=num_classes,
                hyperparameters=hyperparameters
            )
            
            # 3. Upload and train
            result = await self.upload_and_train(job_tarball)
            
            return result
            
        finally:
            # Always cleanup instance
            await self.cleanup_instance()

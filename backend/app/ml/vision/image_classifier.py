"""
Image Classification using Transfer Learning
Uses PyTorch with pre-trained models (ResNet, EfficientNet, ViT)
This is the industry-standard approach for computer vision
"""
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from pathlib import Path
from typing import Dict, Any, Optional, Literal, List, Tuple
import numpy as np
from PIL import Image
import joblib
import json


class ImageDataset(Dataset):
    """Custom dataset for image classification"""
    
    def __init__(self, image_paths: List[str], labels: List[int], transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        label = self.labels[idx]
        return image, label


class ImageClassifier:
    """
    Image classifier using transfer learning with pre-trained CNNs
    
    Supported architectures:
    - ResNet50: Classic, reliable, 25M parameters
    - EfficientNet-B0: Modern, efficient, 5M parameters
    - ViT-B/16: Vision Transformer, state-of-the-art, 86M parameters
    
    This is the production-grade approach used in real-world applications
    """
    
    def __init__(
        self,
        model_name: Literal["resnet50", "efficientnet_b0", "vit_b_16"] = "resnet50",
        num_classes: int = 2,
        device: Optional[str] = None,
    ):
        self.model_name = model_name
        self.num_classes = num_classes
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.class_names = None
        
        print(f"[ImageClassifier] Device: {self.device}")
        if self.device == "cuda":
            print(f"[ImageClassifier] GPU: {torch.cuda.get_device_name(0)}")
        
    def _build_model(self):
        """Build model with pre-trained weights"""
        print(f"[ImageClassifier] Loading pre-trained {self.model_name}...")
        
        if self.model_name == "resnet50":
            # ResNet50 - Classic choice, very reliable
            weights = models.ResNet50_Weights.IMAGENET1K_V2
            model = models.resnet50(weights=weights)
            # Replace final layer
            num_features = model.fc.in_features
            model.fc = nn.Linear(num_features, self.num_classes)
            
        elif self.model_name == "efficientnet_b0":
            # EfficientNet - Modern, efficient
            weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
            model = models.efficientnet_b0(weights=weights)
            # Replace classifier
            num_features = model.classifier[1].in_features
            model.classifier[1] = nn.Linear(num_features, self.num_classes)
            
        elif self.model_name == "vit_b_16":
            # Vision Transformer - State-of-the-art
            weights = models.ViT_B_16_Weights.IMAGENET1K_V1
            model = models.vit_b_16(weights=weights)
            # Replace head
            num_features = model.heads.head.in_features
            model.heads.head = nn.Linear(num_features, self.num_classes)
        
        else:
            raise ValueError(f"Unknown model: {self.model_name}")
        
        model = model.to(self.device)
        print(f"[ImageClassifier] ✅ Model loaded: {self.model_name}")
        
        return model
    
    def _get_transforms(self, mode: Literal["train", "val"]):
        """Get image transformations"""
        if mode == "train":
            # Training: data augmentation
            return transforms.Compose([
                transforms.Resize(256),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],  # ImageNet stats
                    std=[0.229, 0.224, 0.225]
                ),
            ])
        else:
            # Validation/Test: no augmentation
            return transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
            ])
    
    def train(
        self,
        train_image_paths: List[str],
        train_labels: List[int],
        val_image_paths: Optional[List[str]] = None,
        val_labels: Optional[List[int]] = None,
        class_names: Optional[List[str]] = None,
        num_epochs: int = 10,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        freeze_backbone: bool = True,
    ) -> Dict[str, Any]:
        """
        Train image classifier using transfer learning
        
        Args:
            train_image_paths: List of paths to training images
            train_labels: List of training labels (0, 1, 2, ...)
            val_image_paths: List of paths to validation images
            val_labels: List of validation labels
            class_names: Names of classes (e.g., ['cat', 'dog'])
            num_epochs: Number of training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate
            freeze_backbone: If True, only train final layer (faster)
            
        Returns:
            Dictionary with training metrics
        """
        print(f"[ImageClassifier] Starting training...")
        print(f"[ImageClassifier] Training images: {len(train_image_paths)}")
        print(f"[ImageClassifier] Classes: {self.num_classes}")
        print(f"[ImageClassifier] Epochs: {num_epochs}")
        print(f"[ImageClassifier] Batch size: {batch_size}")
        print(f"[ImageClassifier] Freeze backbone: {freeze_backbone}")
        
        # Store class names
        self.class_names = class_names or [f"class_{i}" for i in range(self.num_classes)]
        
        # Build model
        self.model = self._build_model()
        
        # Freeze backbone if requested (transfer learning)
        if freeze_backbone:
            print(f"[ImageClassifier] Freezing backbone (only training final layer)...")
            for param in self.model.parameters():
                param.requires_grad = False
            
            # Unfreeze final layer
            if self.model_name == "resnet50":
                for param in self.model.fc.parameters():
                    param.requires_grad = True
            elif self.model_name == "efficientnet_b0":
                for param in self.model.classifier.parameters():
                    param.requires_grad = True
            elif self.model_name == "vit_b_16":
                for param in self.model.heads.parameters():
                    param.requires_grad = True
        
        # Create datasets
        train_transform = self._get_transforms("train")
        train_dataset = ImageDataset(train_image_paths, train_labels, train_transform)
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True if self.device == "cuda" else False
        )
        
        val_loader = None
        if val_image_paths and val_labels:
            val_transform = self._get_transforms("val")
            val_dataset = ImageDataset(val_image_paths, val_labels, val_transform)
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=4,
                pin_memory=True if self.device == "cuda" else False
            )
        
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=learning_rate
        )
        
        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=2, verbose=True
        )
        
        # Training loop
        best_val_acc = 0.0
        history = {"train_loss": [], "train_acc": [], "val_acc": [], "val_loss": []}
        
        for epoch in range(num_epochs):
            print(f"\n[ImageClassifier] Epoch {epoch+1}/{num_epochs}")
            
            # Training phase
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for batch_idx, (images, labels) in enumerate(train_loader):
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                # Statistics
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                train_total += labels.size(0)
                train_correct += predicted.eq(labels).sum().item()
                
                # Progress
                if (batch_idx + 1) % 10 == 0:
                    print(f"  Batch {batch_idx+1}/{len(train_loader)}: "
                          f"Loss={loss.item():.4f}, "
                          f"Acc={100.*train_correct/train_total:.2f}%")
            
            train_loss /= len(train_loader)
            train_acc = 100. * train_correct / train_total
            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            
            print(f"[ImageClassifier] Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            
            # Validation phase
            if val_loader:
                self.model.eval()
                val_loss = 0.0
                val_correct = 0
                val_total = 0
                
                with torch.no_grad():
                    for images, labels in val_loader:
                        images = images.to(self.device)
                        labels = labels.to(self.device)
                        
                        outputs = self.model(images)
                        loss = criterion(outputs, labels)
                        
                        val_loss += loss.item()
                        _, predicted = outputs.max(1)
                        val_total += labels.size(0)
                        val_correct += predicted.eq(labels).sum().item()
                
                val_loss /= len(val_loader)
                val_acc = 100. * val_correct / val_total
                history["val_loss"].append(val_loss)
                history["val_acc"].append(val_acc)
                
                print(f"[ImageClassifier] Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
                
                # Update learning rate
                scheduler.step(val_acc)
                
                # Save best model
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    print(f"[ImageClassifier] ✅ New best validation accuracy: {val_acc:.2f}%")
        
        print(f"\n[ImageClassifier] ✅ Training complete!")
        
        # Return metrics
        metrics = {
            "train_loss": history["train_loss"][-1],
            "train_accuracy": history["train_acc"][-1],
            "num_epochs": num_epochs,
            "num_samples": len(train_image_paths),
            "num_classes": self.num_classes,
            "model_name": self.model_name,
        }
        
        if val_loader:
            metrics["val_loss"] = history["val_loss"][-1]
            metrics["val_accuracy"] = history["val_acc"][-1]
            metrics["best_val_accuracy"] = best_val_acc
        
        return metrics
    
    def predict(self, image_paths: List[str]) -> np.ndarray:
        """
        Predict classes for images
        
        Args:
            image_paths: List of paths to images
            
        Returns:
            Array of predicted class indices
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        self.model.eval()
        transform = self._get_transforms("val")
        
        predictions = []
        
        with torch.no_grad():
            for image_path in image_paths:
                image = Image.open(image_path).convert('RGB')
                image = transform(image).unsqueeze(0).to(self.device)
                
                outputs = self.model(image)
                _, predicted = outputs.max(1)
                predictions.append(predicted.item())
        
        return np.array(predictions)
    
    def predict_proba(self, image_paths: List[str]) -> np.ndarray:
        """
        Get prediction probabilities for images
        
        Args:
            image_paths: List of paths to images
            
        Returns:
            Array of shape (n_images, n_classes) with probabilities
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        self.model.eval()
        transform = self._get_transforms("val")
        
        probabilities = []
        
        with torch.no_grad():
            for image_path in image_paths:
                image = Image.open(image_path).convert('RGB')
                image = transform(image).unsqueeze(0).to(self.device)
                
                outputs = self.model(image)
                probs = torch.softmax(outputs, dim=1)
                probabilities.append(probs.cpu().numpy()[0])
        
        return np.array(probabilities)
    
    def save(self, save_dir: Path):
        """Save model and metadata"""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model state
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_name': self.model_name,
            'num_classes': self.num_classes,
            'class_names': self.class_names,
        }, save_dir / "model.pth")
        
        # Save metadata
        metadata = {
            'model_name': self.model_name,
            'num_classes': self.num_classes,
            'class_names': self.class_names,
            'device': self.device,
        }
        with open(save_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"[ImageClassifier] ✅ Model saved to {save_dir}")
    
    def load(self, save_dir: Path):
        """Load model and metadata"""
        save_dir = Path(save_dir)
        
        # Load metadata
        with open(save_dir / "metadata.json", 'r') as f:
            metadata = json.load(f)
        
        self.model_name = metadata['model_name']
        self.num_classes = metadata['num_classes']
        self.class_names = metadata['class_names']
        
        # Build model
        self.model = self._build_model()
        
        # Load weights
        checkpoint = torch.load(save_dir / "model.pth", map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        print(f"[ImageClassifier] ✅ Model loaded from {save_dir}")

#!/usr/bin/env python3
"""
Transfer Learning Model for Touch Data Analysis

This module implements a transfer learning approach for touch interaction analysis:
1. Pre-training on synthetic/general touch data
2. Fine-tuning on Coloring-specific patterns
3. Quality assessment and behavioral classification
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
import json
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import pickle
import os

logger = logging.getLogger(__name__)

class TouchSequenceDataset(Dataset):
    """Dataset class for touch sequence data."""

    def __init__(self, sequences: List[np.ndarray], labels: List[Dict], max_length: int = 100):
        self.sequences = sequences
        self.labels = labels
        self.max_length = max_length

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        label = self.labels[idx]

        # Pad or truncate sequence to max_length
        if len(sequence) > self.max_length:
            sequence = sequence[:self.max_length]
        else:
            padding = np.zeros((self.max_length - len(sequence), sequence.shape[1]))
            sequence = np.vstack([sequence, padding])

        return torch.FloatTensor(sequence), label

class TouchTransformerModel(nn.Module):
    """
    Transformer-based model for touch sequence analysis.
    Designed for transfer learning with pre-training and fine-tuning phases.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 128, num_heads: int = 8,
                 num_layers: int = 4, max_length: int = 100, num_classes: int = 5):
        super(TouchTransformerModel, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.max_length = max_length

        # Input projection
        self.input_projection = nn.Linear(input_dim, hidden_dim)

        # Positional encoding
        self.positional_encoding = self._create_positional_encoding(max_length, hidden_dim)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output heads for different tasks
        self.quality_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

        self.pattern_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, num_classes)
        )

        self.anomaly_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

    def _create_positional_encoding(self, max_length: int, hidden_dim: int):
        """Create positional encoding for transformer."""
        pe = torch.zeros(max_length, hidden_dim)
        position = torch.arange(0, max_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_dim, 2).float() *
                           (-np.log(10000.0) / hidden_dim))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        return nn.Parameter(pe.unsqueeze(0), requires_grad=False)

    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape

        # Project input to hidden dimension
        x = self.input_projection(x)

        # Add positional encoding
        x = x + self.positional_encoding[:, :seq_len, :]

        # Apply transformer
        if mask is not None:
            transformer_output = self.transformer(x, src_key_padding_mask=mask)
        else:
            transformer_output = self.transformer(x)

        # Global average pooling for sequence-level representation
        sequence_repr = transformer_output.mean(dim=1)

        # Apply different heads
        quality_score = self.quality_head(sequence_repr)
        pattern_logits = self.pattern_head(sequence_repr)
        anomaly_score = self.anomaly_head(sequence_repr)

        return {
            'quality_score': quality_score,
            'pattern_logits': pattern_logits,
            'anomaly_score': anomaly_score,
            'sequence_representation': sequence_repr
        }

class TouchDataTransferLearner:
    """
    Transfer learning system for touch data analysis.
    Handles pre-training, fine-tuning, and inference.
    """

    def __init__(self, input_dim: int = 6, hidden_dim: int = 128, max_length: int = 100):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.max_length = max_length

        self.model = TouchTransformerModel(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            max_length=max_length
        )

        self.scaler = StandardScaler()
        self.pattern_encoder = LabelEncoder()
        self.is_pretrained = False
        self.is_finetuned = False

        # Device setup
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        logger.info(f"Initialized TouchDataTransferLearner on device: {self.device}")

    def generate_synthetic_data(self, num_sequences: int = 1000) -> Tuple[List[np.ndarray], List[Dict]]:
        """
        Generate synthetic touch data for pre-training.
        Creates diverse touch patterns to learn general touch behavior.
        """
        logger.info(f"Generating {num_sequences} synthetic touch sequences...")

        sequences = []
        labels = []

        for i in range(num_sequences):
            # Random sequence length
            seq_length = np.random.randint(5, self.max_length)

            # Generate different types of synthetic sequences
            pattern_type = np.random.choice(['linear', 'circular', 'random', 'stationary'])

            if pattern_type == 'linear':
                sequence, label = self._generate_linear_sequence(seq_length)
            elif pattern_type == 'circular':
                sequence, label = self._generate_circular_sequence(seq_length)
            elif pattern_type == 'stationary':
                sequence, label = self._generate_stationary_sequence(seq_length)
            else:
                sequence, label = self._generate_random_sequence(seq_length)

            sequences.append(sequence)
            labels.append(label)

        return sequences, labels

    def _generate_linear_sequence(self, length: int) -> Tuple[np.ndarray, Dict]:
        """Generate a linear movement sequence."""
        start_x, start_y = np.random.uniform(0, 1000, 2)
        end_x, end_y = np.random.uniform(0, 1000, 2)

        x_coords = np.linspace(start_x, end_x, length)
        y_coords = np.linspace(start_y, end_y, length)
        times = np.linspace(0, 1000, length)

        # Add some noise
        x_coords += np.random.normal(0, 5, length)
        y_coords += np.random.normal(0, 5, length)

        sequence = np.column_stack([
            x_coords, y_coords, times,
            np.random.uniform(0, 1, length),  # completion percentage
            np.random.normal(0, 0.1, length),  # acc_x
            np.random.normal(0, 0.1, length)   # acc_y
        ])

        label = {
            'quality_score': 0.8 + np.random.uniform(-0.1, 0.1),
            'pattern': 'linear',
            'anomaly_score': 0.1 + np.random.uniform(-0.05, 0.05)
        }

        return sequence, label

    def _generate_circular_sequence(self, length: int) -> Tuple[np.ndarray, Dict]:
        """Generate a circular movement sequence."""
        center_x, center_y = np.random.uniform(200, 800, 2)
        radius = np.random.uniform(50, 200)

        angles = np.linspace(0, 2 * np.pi, length)
        x_coords = center_x + radius * np.cos(angles)
        y_coords = center_y + radius * np.sin(angles)
        times = np.linspace(0, 1500, length)

        # Add noise
        x_coords += np.random.normal(0, 3, length)
        y_coords += np.random.normal(0, 3, length)

        sequence = np.column_stack([
            x_coords, y_coords, times,
            np.random.uniform(0, 1, length),
            np.random.normal(0, 0.1, length),
            np.random.normal(0, 0.1, length)
        ])

        label = {
            'quality_score': 0.9 + np.random.uniform(-0.05, 0.05),
            'pattern': 'circular',
            'anomaly_score': 0.05 + np.random.uniform(-0.02, 0.02)
        }

        return sequence, label

    def _generate_stationary_sequence(self, length: int) -> Tuple[np.ndarray, Dict]:
        """Generate a stationary touch sequence."""
        center_x, center_y = np.random.uniform(100, 900, 2)

        x_coords = np.full(length, center_x) + np.random.normal(0, 2, length)
        y_coords = np.full(length, center_y) + np.random.normal(0, 2, length)
        times = np.linspace(0, 500, length)

        sequence = np.column_stack([
            x_coords, y_coords, times,
            np.random.uniform(0, 0.2, length),
            np.random.normal(0, 0.05, length),
            np.random.normal(0, 0.05, length)
        ])

        label = {
            'quality_score': 0.7 + np.random.uniform(-0.1, 0.1),
            'pattern': 'stationary',
            'anomaly_score': 0.2 + np.random.uniform(-0.1, 0.1)
        }

        return sequence, label

    def _generate_random_sequence(self, length: int) -> Tuple[np.ndarray, Dict]:
        """Generate a random/erratic movement sequence."""
        x_coords = np.random.uniform(0, 1000, length)
        y_coords = np.random.uniform(0, 1000, length)
        times = np.sort(np.random.uniform(0, 2000, length))

        sequence = np.column_stack([
            x_coords, y_coords, times,
            np.random.uniform(0, 1, length),
            np.random.normal(0, 0.2, length),
            np.random.normal(0, 0.2, length)
        ])

        label = {
            'quality_score': 0.3 + np.random.uniform(-0.1, 0.1),
            'pattern': 'random',
            'anomaly_score': 0.7 + np.random.uniform(-0.1, 0.1)
        }

        return sequence, label

    def pretrain(self, num_epochs: int = 50, batch_size: int = 32, learning_rate: float = 0.001):
        """
        Pre-train the model on synthetic data to learn general touch patterns.
        """
        logger.info("Starting pre-training phase...")

        # Generate synthetic data
        sequences, labels = self.generate_synthetic_data(num_sequences=5000)

        # Prepare data
        dataset = TouchSequenceDataset(sequences, labels, self.max_length)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Setup training
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        quality_criterion = nn.MSELoss()
        pattern_criterion = nn.CrossEntropyLoss()
        anomaly_criterion = nn.MSELoss()

        # Encode pattern labels
        pattern_labels = [label['pattern'] for label in labels]
        self.pattern_encoder.fit(pattern_labels)

        self.model.train()
        for epoch in range(num_epochs):
            total_loss = 0.0

            for batch_sequences, batch_labels in dataloader:
                batch_sequences = batch_sequences.to(self.device)

                # Convert batch_labels to list if it's a tensor
                if isinstance(batch_labels, torch.Tensor):
                    batch_labels = batch_labels.tolist()

                # Prepare targets - handle both dict and tensor cases
                try:
                    if isinstance(batch_labels[0], dict):
                        quality_targets = torch.FloatTensor([label['quality_score'] for label in batch_labels]).to(self.device)
                        pattern_targets = torch.LongTensor([
                            self.pattern_encoder.transform([label['pattern']])[0] for label in batch_labels
                        ]).to(self.device)
                        anomaly_targets = torch.FloatTensor([label['anomaly_score'] for label in batch_labels]).to(self.device)
                    else:
                        # Fallback for non-dict labels
                        quality_targets = torch.FloatTensor([0.5] * len(batch_labels)).to(self.device)
                        pattern_targets = torch.LongTensor([0] * len(batch_labels)).to(self.device)
                        anomaly_targets = torch.FloatTensor([0.1] * len(batch_labels)).to(self.device)
                except (KeyError, IndexError, TypeError) as e:
                    logger.warning(f"Label processing error: {e}, using default values")
                    quality_targets = torch.FloatTensor([0.5] * len(batch_labels)).to(self.device)
                    pattern_targets = torch.LongTensor([0] * len(batch_labels)).to(self.device)
                    anomaly_targets = torch.FloatTensor([0.1] * len(batch_labels)).to(self.device)

                # Forward pass
                optimizer.zero_grad()
                outputs = self.model(batch_sequences)

                # Calculate losses
                quality_loss = quality_criterion(outputs['quality_score'].squeeze(), quality_targets)
                pattern_loss = pattern_criterion(outputs['pattern_logits'], pattern_targets)
                anomaly_loss = anomaly_criterion(outputs['anomaly_score'].squeeze(), anomaly_targets)

                total_loss_batch = quality_loss + pattern_loss + anomaly_loss

                # Backward pass
                total_loss_batch.backward()
                optimizer.step()

                total_loss += total_loss_batch.item()

            if epoch % 10 == 0:
                logger.info(f"Pre-training Epoch {epoch}/{num_epochs}, Loss: {total_loss/len(dataloader):.4f}")

        self.is_pretrained = True
        logger.info("Pre-training completed successfully!")

    def save_model(self, filepath: str):
        """Save the trained model and associated components."""
        save_dict = {
            'model_state_dict': self.model.state_dict(),
            'scaler': self.scaler,
            'pattern_encoder': self.pattern_encoder,
            'is_pretrained': self.is_pretrained,
            'is_finetuned': self.is_finetuned,
            'model_config': {
                'input_dim': self.input_dim,
                'hidden_dim': self.hidden_dim,
                'max_length': self.max_length
            }
        }

        torch.save(save_dict, filepath)
        logger.info(f"Model saved to {filepath}")

    def load_model(self, filepath: str):
        """Load a trained model and associated components."""
        save_dict = torch.load(filepath, map_location=self.device)

        self.model.load_state_dict(save_dict['model_state_dict'])
        self.scaler = save_dict['scaler']
        self.pattern_encoder = save_dict['pattern_encoder']
        self.is_pretrained = save_dict['is_pretrained']
        self.is_finetuned = save_dict['is_finetuned']

        logger.info(f"Model loaded from {filepath}")

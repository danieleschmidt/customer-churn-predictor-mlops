"""
Test Data Management System.

This module provides comprehensive test data management including:
- Synthetic test data generation with realistic patterns and distributions
- Test data versioning and lineage tracking with Git-like versioning
- Data masking for privacy compliance in test environments
- Test environment isolation and data cleanup automation
- Fixtures and factories for consistent test data across suites
- Data validation and integrity checking for test datasets
- Performance-optimized data loading and caching mechanisms
"""

import os
import json
import hashlib
import tempfile
import sqlite3
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional, Union, Generator
from dataclasses import dataclass, asdict, field
from collections import defaultdict
import shutil
import uuid

# Data generation libraries
try:
    import pandas as pd
    import numpy as np
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

try:
    from faker import Faker
    FAKER_AVAILABLE = True
except ImportError:
    FAKER_AVAILABLE = False


@dataclass 
class DataSchema:
    """Schema definition for synthetic data generation."""
    name: str
    fields: Dict[str, Dict[str, Any]]
    constraints: List[Dict[str, Any]] = field(default_factory=list)
    relationships: List[Dict[str, Any]] = field(default_factory=list)
    size_parameters: Dict[str, int] = field(default_factory=dict)


@dataclass
class TestDataset:
    """Represents a test dataset with metadata."""
    dataset_id: str
    name: str
    version: str
    schema_name: str
    file_path: str
    checksum: str
    created_at: datetime
    size_bytes: int
    row_count: int
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DataMaskingRule:
    """Rule for masking sensitive data."""
    field_name: str
    masking_type: str  # 'hash', 'random', 'anonymize', 'redact'
    parameters: Dict[str, Any] = field(default_factory=dict)


class SyntheticDataGenerator:
    """Generates synthetic test data based on schemas."""
    
    def __init__(self):
        self.fake = Faker() if FAKER_AVAILABLE else None
        self.generators = {
            'string': self._generate_string,
            'integer': self._generate_integer,
            'float': self._generate_float,
            'boolean': self._generate_boolean,
            'date': self._generate_date,
            'datetime': self._generate_datetime,
            'email': self._generate_email,
            'name': self._generate_name,
            'address': self._generate_address,
            'phone': self._generate_phone,
            'uuid': self._generate_uuid,
            'categorical': self._generate_categorical,
            'normal_distribution': self._generate_normal,
            'uniform_distribution': self._generate_uniform
        }
    
    def generate_dataset(self, schema: DataSchema, num_rows: int) -> pd.DataFrame:
        """Generate a complete dataset based on schema."""
        if not PANDAS_AVAILABLE:
            raise ImportError("Pandas not available for data generation")
        
        print(f"🎲 Generating {num_rows} rows for schema '{schema.name}'")
        
        data = {}
        
        # Generate base fields
        for field_name, field_config in schema.fields.items():
            field_type = field_config.get('type', 'string')
            
            if field_type in self.generators:
                generator = self.generators[field_type]
                data[field_name] = [
                    generator(field_config) for _ in range(num_rows)
                ]
            else:
                # Default to string if unknown type
                data[field_name] = [
                    self._generate_string(field_config) for _ in range(num_rows)
                ]
        
        df = pd.DataFrame(data)
        
        # Apply constraints
        for constraint in schema.constraints:
            df = self._apply_constraint(df, constraint)
        
        # Apply relationships
        for relationship in schema.relationships:
            df = self._apply_relationship(df, relationship)
        
        print(f"✅ Generated dataset with shape {df.shape}")
        return df
    
    def _generate_string(self, config: Dict[str, Any]) -> str:
        """Generate string data."""
        min_length = config.get('min_length', 5)
        max_length = config.get('max_length', 20)
        pattern = config.get('pattern', None)
        
        if pattern == 'name' and self.fake:
            return self.fake.name()
        elif pattern == 'company' and self.fake:
            return self.fake.company()
        elif pattern == 'text' and self.fake:
            return self.fake.text(max_nb_chars=max_length)
        else:
            # Generate random string
            import string
            import random
            length = random.randint(min_length, max_length)
            return ''.join(random.choices(string.ascii_letters + string.digits, k=length))
    
    def _generate_integer(self, config: Dict[str, Any]) -> int:
        """Generate integer data."""
        min_val = config.get('min', 0)
        max_val = config.get('max', 1000)
        return np.random.randint(min_val, max_val + 1)
    
    def _generate_float(self, config: Dict[str, Any]) -> float:
        """Generate float data."""
        min_val = config.get('min', 0.0)
        max_val = config.get('max', 1.0)
        decimals = config.get('decimals', 2)
        value = np.random.uniform(min_val, max_val)
        return round(value, decimals)
    
    def _generate_boolean(self, config: Dict[str, Any]) -> bool:
        """Generate boolean data."""
        true_probability = config.get('true_probability', 0.5)
        return np.random.random() < true_probability
    
    def _generate_date(self, config: Dict[str, Any]) -> str:
        """Generate date data."""
        if self.fake:
            start_date = config.get('start_date', '-30y')
            end_date = config.get('end_date', 'now')
            return self.fake.date_between(start_date=start_date, end_date=end_date).isoformat()
        else:
            # Simple date generation
            base_date = datetime(2020, 1, 1)
            random_days = np.random.randint(0, 1460)  # 4 years
            return (base_date + timedelta(days=int(random_days))).date().isoformat()
    
    def _generate_datetime(self, config: Dict[str, Any]) -> str:
        """Generate datetime data."""
        if self.fake:
            return self.fake.date_time_between(start_date='-1y', end_date='now').isoformat()
        else:
            base_date = datetime(2020, 1, 1)
            random_seconds = np.random.randint(0, 365 * 24 * 3600)
            return (base_date + timedelta(seconds=int(random_seconds))).isoformat()
    
    def _generate_email(self, config: Dict[str, Any]) -> str:
        """Generate email data."""
        if self.fake:
            return self.fake.email()
        else:
            import string
            import random
            username_length = random.randint(5, 15)
            username = ''.join(random.choices(string.ascii_lowercase, k=username_length))
            domains = ['test.com', 'example.org', 'sample.net']
            domain = random.choice(domains)
            return f"{username}@{domain}"
    
    def _generate_name(self, config: Dict[str, Any]) -> str:
        """Generate name data."""
        if self.fake:
            return self.fake.name()
        else:
            first_names = ['John', 'Jane', 'Alice', 'Bob', 'Charlie', 'Diana', 'Eve', 'Frank']
            last_names = ['Smith', 'Johnson', 'Williams', 'Brown', 'Jones', 'Garcia', 'Miller']
            return f"{np.random.choice(first_names)} {np.random.choice(last_names)}"
    
    def _generate_address(self, config: Dict[str, Any]) -> str:
        """Generate address data."""
        if self.fake:
            return self.fake.address().replace('\n', ', ')
        else:
            street_num = np.random.randint(100, 9999)
            streets = ['Main St', 'Oak Ave', 'Park Rd', 'First St', 'Second Ave']
            return f"{street_num} {np.random.choice(streets)}"
    
    def _generate_phone(self, config: Dict[str, Any]) -> str:
        """Generate phone number data."""
        if self.fake:
            return self.fake.phone_number()
        else:
            area_code = np.random.randint(200, 999)
            exchange = np.random.randint(200, 999)
            number = np.random.randint(1000, 9999)
            return f"({area_code}) {exchange}-{number}"
    
    def _generate_uuid(self, config: Dict[str, Any]) -> str:
        """Generate UUID data."""
        return str(uuid.uuid4())
    
    def _generate_categorical(self, config: Dict[str, Any]) -> str:
        """Generate categorical data."""
        categories = config.get('categories', ['A', 'B', 'C'])
        weights = config.get('weights', None)
        
        if weights and len(weights) == len(categories):
            return np.random.choice(categories, p=weights)
        else:
            return np.random.choice(categories)
    
    def _generate_normal(self, config: Dict[str, Any]) -> float:
        """Generate normally distributed data."""
        mean = config.get('mean', 0)
        std = config.get('std', 1)
        decimals = config.get('decimals', 2)
        value = np.random.normal(mean, std)
        return round(value, decimals)
    
    def _generate_uniform(self, config: Dict[str, Any]) -> float:
        """Generate uniformly distributed data."""
        low = config.get('low', 0)
        high = config.get('high', 1)
        decimals = config.get('decimals', 2)
        value = np.random.uniform(low, high)
        return round(value, decimals)
    
    def _apply_constraint(self, df: pd.DataFrame, constraint: Dict[str, Any]) -> pd.DataFrame:
        """Apply data constraint."""
        constraint_type = constraint.get('type')
        
        if constraint_type == 'unique':
            field = constraint['field']
            df[field] = df[field].astype(str) + '_' + df.index.astype(str)
        
        elif constraint_type == 'not_null':
            field = constraint['field']
            # Replace any null values with default
            default_value = constraint.get('default', 'N/A')
            df[field] = df[field].fillna(default_value)
        
        elif constraint_type == 'range':
            field = constraint['field']
            min_val = constraint.get('min')
            max_val = constraint.get('max')
            
            if min_val is not None:
                df[field] = df[field].clip(lower=min_val)
            if max_val is not None:
                df[field] = df[field].clip(upper=max_val)
        
        return df
    
    def _apply_relationship(self, df: pd.DataFrame, relationship: Dict[str, Any]) -> pd.DataFrame:
        """Apply data relationship."""
        rel_type = relationship.get('type')
        
        if rel_type == 'foreign_key':
            child_field = relationship['child_field']
            parent_field = relationship['parent_field']
            
            # Ensure child field values exist in parent field
            unique_parents = df[parent_field].unique()
            df[child_field] = np.random.choice(unique_parents, size=len(df))
        
        elif rel_type == 'calculated':
            target_field = relationship['target_field']
            source_fields = relationship['source_fields']
            formula = relationship['formula']
            
            # Simple formula evaluation (extend as needed)
            if formula == 'sum':
                df[target_field] = df[source_fields].sum(axis=1)
            elif formula == 'avg':
                df[target_field] = df[source_fields].mean(axis=1)
        
        return df


class DataMasker:
    """Masks sensitive data for test environments."""
    
    def __init__(self):
        self.fake = Faker() if FAKER_AVAILABLE else None
    
    def mask_dataset(self, df: pd.DataFrame, masking_rules: List[DataMaskingRule]) -> pd.DataFrame:
        """Apply masking rules to a dataset."""
        print(f"🎭 Applying {len(masking_rules)} masking rules")
        
        masked_df = df.copy()
        
        for rule in masking_rules:
            if rule.field_name not in masked_df.columns:
                print(f"⚠️ Field {rule.field_name} not found in dataset")
                continue
            
            masked_df[rule.field_name] = self._apply_masking_rule(
                masked_df[rule.field_name], rule
            )
        
        return masked_df
    
    def _apply_masking_rule(self, series: pd.Series, rule: DataMaskingRule) -> pd.Series:
        """Apply a single masking rule to a pandas Series."""
        
        if rule.masking_type == 'hash':
            # Hash the values
            salt = rule.parameters.get('salt', 'default_salt')
            return series.apply(lambda x: self._hash_value(str(x), salt))
        
        elif rule.masking_type == 'random':
            # Replace with random values of same type
            data_type = rule.parameters.get('type', 'string')
            return series.apply(lambda x: self._generate_random_replacement(data_type))
        
        elif rule.masking_type == 'anonymize':
            # Replace with fake data
            field_type = rule.parameters.get('field_type', 'name')
            return series.apply(lambda x: self._generate_fake_replacement(field_type))
        
        elif rule.masking_type == 'redact':
            # Replace with redacted placeholder
            placeholder = rule.parameters.get('placeholder', '[REDACTED]')
            return pd.Series([placeholder] * len(series), index=series.index)
        
        elif rule.masking_type == 'partial':
            # Show only part of the value
            show_start = rule.parameters.get('show_start', 2)
            show_end = rule.parameters.get('show_end', 2)
            mask_char = rule.parameters.get('mask_char', '*')
            
            def partial_mask(value):
                str_val = str(value)
                if len(str_val) <= show_start + show_end:
                    return mask_char * len(str_val)
                
                start_part = str_val[:show_start]
                end_part = str_val[-show_end:] if show_end > 0 else ''
                middle_part = mask_char * (len(str_val) - show_start - show_end)
                
                return start_part + middle_part + end_part
            
            return series.apply(partial_mask)
        
        else:
            print(f"⚠️ Unknown masking type: {rule.masking_type}")
            return series
    
    def _hash_value(self, value: str, salt: str) -> str:
        """Hash a value with salt."""
        combined = f"{value}{salt}"
        return hashlib.sha256(combined.encode()).hexdigest()[:16]
    
    def _generate_random_replacement(self, data_type: str) -> Any:
        """Generate random replacement value."""
        if data_type == 'string':
            import string
            import random
            return ''.join(random.choices(string.ascii_letters, k=8))
        elif data_type == 'integer':
            return np.random.randint(1, 1000)
        elif data_type == 'float':
            return round(np.random.uniform(0, 100), 2)
        elif data_type == 'email':
            return f"user{np.random.randint(1000, 9999)}@example.com"
        else:
            return "MASKED"
    
    def _generate_fake_replacement(self, field_type: str) -> str:
        """Generate fake replacement using Faker."""
        if not self.fake:
            return self._generate_random_replacement('string')
        
        if field_type == 'name':
            return self.fake.name()
        elif field_type == 'email':
            return self.fake.email()
        elif field_type == 'address':
            return self.fake.address().replace('\n', ', ')
        elif field_type == 'phone':
            return self.fake.phone_number()
        elif field_type == 'company':
            return self.fake.company()
        else:
            return self.fake.text(max_nb_chars=20)


class TestDataVersionManager:
    """Manages versioning of test data with Git-like functionality."""
    
    def __init__(self, data_dir: str = "test_data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.db_path = self.data_dir / "data_registry.db"
        self._init_database()
    
    def _init_database(self):
        """Initialize the data registry database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS datasets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                dataset_id TEXT UNIQUE NOT NULL,
                name TEXT NOT NULL,
                version TEXT NOT NULL,
                schema_name TEXT,
                file_path TEXT NOT NULL,
                checksum TEXT NOT NULL,
                created_at TEXT NOT NULL,
                size_bytes INTEGER,
                row_count INTEGER,
                tags TEXT,
                metadata TEXT
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS schemas (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                definition TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)
        
        conn.commit()
        conn.close()
    
    def register_schema(self, schema: DataSchema) -> None:
        """Register a data schema."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        schema_json = json.dumps(asdict(schema))
        current_time = datetime.now().isoformat()
        
        cursor.execute("""
            INSERT OR REPLACE INTO schemas (name, definition, created_at, updated_at)
            VALUES (?, ?, COALESCE((SELECT created_at FROM schemas WHERE name = ?), ?), ?)
        """, (schema.name, schema_json, schema.name, current_time, current_time))
        
        conn.commit()
        conn.close()
        
        print(f"📝 Schema '{schema.name}' registered")
    
    def save_dataset(self, dataset: pd.DataFrame, name: str, schema_name: str = None,
                    version: str = None, tags: List[str] = None, 
                    metadata: Dict[str, Any] = None) -> TestDataset:
        """Save a dataset with versioning."""
        
        # Generate version if not provided
        if version is None:
            existing_versions = self._get_dataset_versions(name)
            if existing_versions:
                latest_version = max(existing_versions)
                major, minor, patch = map(int, latest_version.split('.'))
                version = f"{major}.{minor}.{patch + 1}"
            else:
                version = "1.0.0"
        
        # Generate unique dataset ID
        dataset_id = f"{name}_{version}_{uuid.uuid4().hex[:8]}"
        
        # Save dataset file
        file_path = self.data_dir / f"{dataset_id}.parquet"
        
        if PANDAS_AVAILABLE:
            try:
                dataset.to_parquet(file_path, index=False)
            except:
                # Fallback to CSV if parquet fails
                file_path = self.data_dir / f"{dataset_id}.csv"
                dataset.to_csv(file_path, index=False)
        else:
            raise ImportError("Pandas not available for data saving")
        
        # Calculate checksum
        checksum = self._calculate_file_checksum(file_path)
        
        # Create dataset metadata
        test_dataset = TestDataset(
            dataset_id=dataset_id,
            name=name,
            version=version,
            schema_name=schema_name or "unknown",
            file_path=str(file_path),
            checksum=checksum,
            created_at=datetime.now(),
            size_bytes=file_path.stat().st_size,
            row_count=len(dataset),
            tags=tags or [],
            metadata=metadata or {}
        )
        
        # Register in database
        self._register_dataset(test_dataset)
        
        print(f"💾 Dataset '{name}' v{version} saved ({test_dataset.row_count} rows, {test_dataset.size_bytes} bytes)")
        
        return test_dataset
    
    def load_dataset(self, name: str, version: str = "latest") -> Tuple[pd.DataFrame, TestDataset]:
        """Load a dataset by name and version."""
        
        if version == "latest":
            versions = self._get_dataset_versions(name)
            if not versions:
                raise ValueError(f"No versions found for dataset '{name}'")
            version = max(versions)
        
        # Get dataset metadata
        dataset_info = self._get_dataset_info(name, version)
        if not dataset_info:
            raise ValueError(f"Dataset '{name}' version '{version}' not found")
        
        # Load dataset file
        file_path = Path(dataset_info['file_path'])
        
        if not file_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {file_path}")
        
        # Verify checksum
        current_checksum = self._calculate_file_checksum(file_path)
        if current_checksum != dataset_info['checksum']:
            print("⚠️ Dataset checksum mismatch - file may have been modified")
        
        # Load data
        if file_path.suffix == '.parquet':
            dataset = pd.read_parquet(file_path)
        elif file_path.suffix == '.csv':
            dataset = pd.read_csv(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
        
        # Reconstruct TestDataset object
        test_dataset = TestDataset(
            dataset_id=dataset_info['dataset_id'],
            name=dataset_info['name'],
            version=dataset_info['version'],
            schema_name=dataset_info['schema_name'],
            file_path=dataset_info['file_path'],
            checksum=dataset_info['checksum'],
            created_at=datetime.fromisoformat(dataset_info['created_at']),
            size_bytes=dataset_info['size_bytes'],
            row_count=dataset_info['row_count'],
            tags=json.loads(dataset_info['tags'] or '[]'),
            metadata=json.loads(dataset_info['metadata'] or '{}')
        )
        
        print(f"📂 Loaded dataset '{name}' v{version} ({len(dataset)} rows)")
        
        return dataset, test_dataset
    
    def list_datasets(self, name_pattern: str = None) -> List[Dict[str, Any]]:
        """List available datasets."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        if name_pattern:
            cursor.execute("""
                SELECT name, version, schema_name, created_at, row_count, size_bytes
                FROM datasets 
                WHERE name LIKE ?
                ORDER BY name, version DESC
            """, (f"%{name_pattern}%",))
        else:
            cursor.execute("""
                SELECT name, version, schema_name, created_at, row_count, size_bytes
                FROM datasets 
                ORDER BY name, version DESC
            """)
        
        results = cursor.fetchall()
        conn.close()
        
        datasets = []
        for row in results:
            datasets.append({
                'name': row[0],
                'version': row[1],
                'schema_name': row[2],
                'created_at': row[3],
                'row_count': row[4],
                'size_bytes': row[5]
            })
        
        return datasets
    
    def delete_dataset(self, name: str, version: str = None) -> None:
        """Delete a dataset version or all versions."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        if version:
            # Delete specific version
            cursor.execute("""
                SELECT file_path FROM datasets WHERE name = ? AND version = ?
            """, (name, version))
            result = cursor.fetchone()
            
            if result:
                file_path = Path(result[0])
                if file_path.exists():
                    file_path.unlink()
                
                cursor.execute("""
                    DELETE FROM datasets WHERE name = ? AND version = ?
                """, (name, version))
                
                print(f"🗑️ Deleted dataset '{name}' v{version}")
            else:
                print(f"❌ Dataset '{name}' v{version} not found")
        else:
            # Delete all versions
            cursor.execute("""
                SELECT file_path FROM datasets WHERE name = ?
            """, (name,))
            results = cursor.fetchall()
            
            for (file_path,) in results:
                path = Path(file_path)
                if path.exists():
                    path.unlink()
            
            cursor.execute("""
                DELETE FROM datasets WHERE name = ?
            """, (name,))
            
            print(f"🗑️ Deleted all versions of dataset '{name}'")
        
        conn.commit()
        conn.close()
    
    def _register_dataset(self, dataset: TestDataset) -> None:
        """Register a dataset in the database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO datasets 
            (dataset_id, name, version, schema_name, file_path, checksum, created_at, 
             size_bytes, row_count, tags, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            dataset.dataset_id,
            dataset.name,
            dataset.version,
            dataset.schema_name,
            dataset.file_path,
            dataset.checksum,
            dataset.created_at.isoformat(),
            dataset.size_bytes,
            dataset.row_count,
            json.dumps(dataset.tags),
            json.dumps(dataset.metadata)
        ))
        
        conn.commit()
        conn.close()
    
    def _get_dataset_versions(self, name: str) -> List[str]:
        """Get all versions of a dataset."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT version FROM datasets WHERE name = ? ORDER BY version
        """, (name,))
        
        results = cursor.fetchall()
        conn.close()
        
        return [row[0] for row in results]
    
    def _get_dataset_info(self, name: str, version: str) -> Optional[Dict[str, Any]]:
        """Get dataset information."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT * FROM datasets WHERE name = ? AND version = ?
        """, (name, version))
        
        result = cursor.fetchone()
        conn.close()
        
        if result:
            columns = ['id', 'dataset_id', 'name', 'version', 'schema_name', 'file_path',
                      'checksum', 'created_at', 'size_bytes', 'row_count', 'tags', 'metadata']
            return dict(zip(columns, result))
        
        return None
    
    def _calculate_file_checksum(self, file_path: Path) -> str:
        """Calculate MD5 checksum of a file."""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()


class TestDataFixtureManager:
    """Manages test data fixtures and factories."""
    
    def __init__(self, data_manager: TestDataVersionManager):
        self.data_manager = data_manager
        self.fixtures = {}
        self.factories = {}
    
    def register_fixture(self, name: str, dataset_name: str, version: str = "latest", 
                        filters: Dict[str, Any] = None):
        """Register a test data fixture."""
        self.fixtures[name] = {
            'dataset_name': dataset_name,
            'version': version,
            'filters': filters or {}
        }
        
        print(f"🏭 Fixture '{name}' registered")
    
    def get_fixture(self, name: str) -> pd.DataFrame:
        """Get a test data fixture."""
        if name not in self.fixtures:
            raise ValueError(f"Fixture '{name}' not found")
        
        fixture_config = self.fixtures[name]
        
        # Load base dataset
        dataset, _ = self.data_manager.load_dataset(
            fixture_config['dataset_name'], 
            fixture_config['version']
        )
        
        # Apply filters
        for column, value in fixture_config['filters'].items():
            if isinstance(value, dict):
                # Complex filter
                if 'gt' in value:
                    dataset = dataset[dataset[column] > value['gt']]
                if 'lt' in value:
                    dataset = dataset[dataset[column] < value['lt']]
                if 'eq' in value:
                    dataset = dataset[dataset[column] == value['eq']]
                if 'in' in value:
                    dataset = dataset[dataset[column].isin(value['in'])]
            else:
                # Simple equality filter
                dataset = dataset[dataset[column] == value]
        
        return dataset.copy()
    
    def register_factory(self, name: str, generator_func: callable, **kwargs):
        """Register a data factory function."""
        self.factories[name] = {
            'generator': generator_func,
            'parameters': kwargs
        }
        
        print(f"🏗️ Factory '{name}' registered")
    
    def create_from_factory(self, name: str, count: int, **override_params) -> pd.DataFrame:
        """Create data using a factory."""
        if name not in self.factories:
            raise ValueError(f"Factory '{name}' not found")
        
        factory_config = self.factories[name]
        generator = factory_config['generator']
        
        # Merge parameters
        parameters = factory_config['parameters'].copy()
        parameters.update(override_params)
        parameters['count'] = count
        
        return generator(**parameters)


def create_sample_schemas() -> List[DataSchema]:
    """Create sample data schemas for testing."""
    
    # Customer schema
    customer_schema = DataSchema(
        name="customers",
        fields={
            'customer_id': {'type': 'integer', 'min': 1000, 'max': 99999},
            'first_name': {'type': 'name'},
            'last_name': {'type': 'name'},
            'email': {'type': 'email'},
            'phone': {'type': 'phone'},
            'birth_date': {'type': 'date', 'start_date': '-80y', 'end_date': '-18y'},
            'registration_date': {'type': 'datetime', 'start_date': '-3y', 'end_date': 'now'},
            'account_balance': {'type': 'normal_distribution', 'mean': 1000, 'std': 500, 'decimals': 2},
            'account_status': {'type': 'categorical', 'categories': ['active', 'inactive', 'suspended'], 'weights': [0.8, 0.15, 0.05]},
            'is_premium': {'type': 'boolean', 'true_probability': 0.3}
        },
        constraints=[
            {'type': 'unique', 'field': 'customer_id'},
            {'type': 'unique', 'field': 'email'},
            {'type': 'range', 'field': 'account_balance', 'min': 0}
        ]
    )
    
    # Transaction schema
    transaction_schema = DataSchema(
        name="transactions",
        fields={
            'transaction_id': {'type': 'uuid'},
            'customer_id': {'type': 'integer', 'min': 1000, 'max': 99999},
            'amount': {'type': 'uniform_distribution', 'low': 10, 'high': 5000, 'decimals': 2},
            'transaction_type': {'type': 'categorical', 'categories': ['purchase', 'refund', 'transfer'], 'weights': [0.8, 0.1, 0.1]},
            'category': {'type': 'categorical', 'categories': ['groceries', 'electronics', 'clothing', 'entertainment', 'utilities']},
            'timestamp': {'type': 'datetime', 'start_date': '-1y', 'end_date': 'now'},
            'description': {'type': 'string', 'pattern': 'text', 'min_length': 10, 'max_length': 100}
        },
        relationships=[
            {'type': 'foreign_key', 'child_field': 'customer_id', 'parent_field': 'customer_id'}
        ]
    )
    
    # ML Features schema
    ml_features_schema = DataSchema(
        name="ml_features",
        fields={
            'sample_id': {'type': 'integer', 'min': 1, 'max': 1000000},
            'feature_1': {'type': 'normal_distribution', 'mean': 0, 'std': 1, 'decimals': 4},
            'feature_2': {'type': 'normal_distribution', 'mean': 0, 'std': 1, 'decimals': 4},
            'feature_3': {'type': 'uniform_distribution', 'low': 0, 'high': 1, 'decimals': 4},
            'feature_4': {'type': 'categorical', 'categories': ['A', 'B', 'C', 'D'], 'weights': [0.4, 0.3, 0.2, 0.1]},
            'feature_5': {'type': 'boolean', 'true_probability': 0.6},
            'target': {'type': 'boolean', 'true_probability': 0.3}
        },
        relationships=[
            {'type': 'calculated', 'target_field': 'feature_sum', 'source_fields': ['feature_1', 'feature_2'], 'formula': 'sum'}
        ]
    )
    
    return [customer_schema, transaction_schema, ml_features_schema]


def main():
    """Main function for test data management."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Data Management System")
    parser.add_argument("command", choices=["generate", "list", "load", "delete", "mask"],
                       help="Command to execute")
    parser.add_argument("--name", help="Dataset name")
    parser.add_argument("--version", help="Dataset version")
    parser.add_argument("--schema", help="Schema name")
    parser.add_argument("--rows", type=int, default=1000, help="Number of rows to generate")
    parser.add_argument("--output", help="Output file path")
    
    args = parser.parse_args()
    
    if not PANDAS_AVAILABLE:
        print("❌ Pandas not available - limited functionality")
        return
    
    # Initialize managers
    data_manager = TestDataVersionManager()
    generator = SyntheticDataGenerator()
    masker = DataMasker()
    
    if args.command == "generate":
        if not args.name or not args.schema:
            print("❌ Name and schema required for generate command")
            return
        
        # Get sample schemas
        schemas = {s.name: s for s in create_sample_schemas()}
        
        if args.schema not in schemas:
            print(f"❌ Schema '{args.schema}' not found. Available: {list(schemas.keys())}")
            return
        
        # Register schema
        schema = schemas[args.schema]
        data_manager.register_schema(schema)
        
        # Generate dataset
        print(f"🎲 Generating {args.rows} rows for schema '{args.schema}'")
        dataset = generator.generate_dataset(schema, args.rows)
        
        # Save dataset
        test_dataset = data_manager.save_dataset(
            dataset, args.name, args.schema, tags=['synthetic', 'test']
        )
        
        print(f"✅ Generated and saved dataset '{args.name}' with {len(dataset)} rows")
        
        if args.output:
            dataset.to_csv(args.output, index=False)
            print(f"💾 Also saved to {args.output}")
    
    elif args.command == "list":
        datasets = data_manager.list_datasets(args.name)
        
        if not datasets:
            print("📭 No datasets found")
            return
        
        print("📂 Available datasets:")
        for ds in datasets:
            print(f"  • {ds['name']} v{ds['version']} ({ds['row_count']} rows, {ds['size_bytes']} bytes)")
    
    elif args.command == "load":
        if not args.name:
            print("❌ Name required for load command")
            return
        
        version = args.version or "latest"
        dataset, metadata = data_manager.load_dataset(args.name, version)
        
        print(f"📊 Dataset summary:")
        print(f"  Shape: {dataset.shape}")
        print(f"  Columns: {list(dataset.columns)}")
        
        if args.output:
            dataset.to_csv(args.output, index=False)
            print(f"💾 Saved to {args.output}")
        else:
            print("\n🔍 First 5 rows:")
            print(dataset.head())
    
    elif args.command == "delete":
        if not args.name:
            print("❌ Name required for delete command")
            return
        
        data_manager.delete_dataset(args.name, args.version)
    
    elif args.command == "mask":
        if not args.name:
            print("❌ Name required for mask command")
            return
        
        # Load dataset
        version = args.version or "latest"
        dataset, metadata = data_manager.load_dataset(args.name, version)
        
        # Apply sample masking rules
        masking_rules = [
            DataMaskingRule('email', 'anonymize', {'field_type': 'email'}),
            DataMaskingRule('first_name', 'anonymize', {'field_type': 'name'}),
            DataMaskingRule('last_name', 'partial', {'show_start': 1, 'show_end': 0}),
            DataMaskingRule('phone', 'partial', {'show_start': 3, 'show_end': 4}),
        ]
        
        # Filter rules to only applicable fields
        applicable_rules = [rule for rule in masking_rules if rule.field_name in dataset.columns]
        
        if applicable_rules:
            masked_dataset = masker.mask_dataset(dataset, applicable_rules)
            
            # Save masked dataset
            masked_name = f"{args.name}_masked"
            data_manager.save_dataset(
                masked_dataset, masked_name, metadata.schema_name, 
                tags=['masked', 'privacy-safe']
            )
            
            print(f"🎭 Created masked dataset '{masked_name}'")
            
            if args.output:
                masked_dataset.to_csv(args.output, index=False)
                print(f"💾 Saved to {args.output}")
        else:
            print("❌ No applicable masking rules found for this dataset")


if __name__ == "__main__":
    main()
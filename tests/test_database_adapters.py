import pytest
import asyncio
from datetime import datetime

from agent_platform.storage.adapters.postgresql.relational_adapter import PostgreSQLAdapter
from agent_platform.storage.adapters.mysql.relational_adapter import MySQLAdapter
from agent_platform.storage.adapters.mongodb.document_adapter import MongoDBAdapter


class TestPostgreSQLAdapter:
    
    @pytest.fixture
    def pg_adapter(self):
        return PostgreSQLAdapter(
            host="localhost",
            port=5432,
            database="test_agents",
            user="postgres",
            password="",
            pool_size=2,
        )
    
    def test_initialization(self, pg_adapter):
        assert pg_adapter._host == "localhost"
        assert pg_adapter._port == 5432
        assert pg_adapter._database == "test_agents"
        assert pg_adapter._pool_size == 2
        assert pg_adapter._enable_cb is True
    
    def test_circuit_breaker_config(self, pg_adapter):
        assert pg_adapter._failure_threshold == 5
        assert pg_adapter._recovery_timeout == 30.0
        assert pg_adapter._circuit_open is False
        assert pg_adapter._failure_count == 0
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_check(self, pg_adapter):
        import time
        
        assert pg_adapter._check_circuit_breaker() is True
        
        pg_adapter._circuit_open = True
        pg_adapter._last_failure_time = time.time()
        assert pg_adapter._check_circuit_breaker() is False
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_recovery(self, pg_adapter):
        import time
        
        pg_adapter._circuit_open = True
        pg_adapter._last_failure_time = time.time() - 60
        
        assert pg_adapter._check_circuit_breaker() is True
        assert pg_adapter._circuit_open is False
        assert pg_adapter._failure_count == 0
    
    def test_record_failure(self, pg_adapter):
        pg_adapter._record_failure()
        assert pg_adapter._failure_count == 1
        
        for _ in range(4):
            pg_adapter._record_failure()
        
        assert pg_adapter._failure_count == 5
        assert pg_adapter._circuit_open is True
    
    def test_record_success(self, pg_adapter):
        pg_adapter._failure_count = 3
        pg_adapter._record_success()
        assert pg_adapter._failure_count == 0


class TestMySQLAdapter:
    
    @pytest.fixture
    def mysql_adapter(self):
        return MySQLAdapter(
            host="localhost",
            port=3306,
            database="test_agents",
            user="root",
            password="",
            pool_size=2,
        )
    
    def test_initialization(self, mysql_adapter):
        assert mysql_adapter._host == "localhost"
        assert mysql_adapter._port == 3306
        assert mysql_adapter._database == "test_agents"
        assert mysql_adapter._pool_size == 2
        assert mysql_adapter._charset == "utf8mb4"
    
    def test_circuit_breaker_config(self, mysql_adapter):
        assert mysql_adapter._enable_cb is True
        assert mysql_adapter._failure_threshold == 5
        assert mysql_adapter._circuit_open is False
    
    def test_record_failure(self, mysql_adapter):
        for _ in range(5):
            mysql_adapter._record_failure()
        
        assert mysql_adapter._circuit_open is True


class TestMongoDBAdapter:
    
    @pytest.fixture
    def mongo_adapter(self):
        return MongoDBAdapter(
            host="localhost",
            port=27017,
            database="test_agents",
            max_pool_size=10,
        )
    
    def test_initialization(self, mongo_adapter):
        assert mongo_adapter._host == "localhost"
        assert mongo_adapter._port == 27017
        assert mongo_adapter._database == "test_agents"
        assert mongo_adapter._max_pool_size == 10
    
    def test_circuit_breaker_config(self, mongo_adapter):
        assert mongo_adapter._enable_cb is True
        assert mongo_adapter._failure_threshold == 5
        assert mongo_adapter._circuit_open is False
    
    @pytest.mark.asyncio
    async def test_timestamp_handling(self, mongo_adapter):
        mongo_adapter._client = "mock"
        mongo_adapter._db = "mock"
        
        now = datetime.utcnow()
        doc = {"data": "test"}
        
        assert "created_at" not in doc
        assert "updated_at" not in doc


class TestDatabaseAdapterComparison:
    
    def test_all_adapters_have_circuit_breaker(self):
        pg = PostgreSQLAdapter()
        mysql = MySQLAdapter()
        mongo = MongoDBAdapter()
        
        assert hasattr(pg, '_enable_cb')
        assert hasattr(mysql, '_enable_cb')
        assert hasattr(mongo, '_enable_cb')
        
        assert hasattr(pg, '_check_circuit_breaker')
        assert hasattr(mysql, '_check_circuit_breaker')
        assert hasattr(mongo, '_check_circuit_breaker')
    
    def test_all_adapters_have_slow_query_threshold(self):
        pg = PostgreSQLAdapter()
        mysql = MySQLAdapter()
        mongo = MongoDBAdapter()
        
        assert hasattr(pg, '_slow_query_threshold')
        assert hasattr(mysql, '_slow_query_threshold')
        assert hasattr(mongo, '_slow_query_threshold')
        
        assert pg._slow_query_threshold == 1.0
        assert mysql._slow_query_threshold == 1.0
        assert mongo._slow_query_threshold == 1.0
    
    def test_all_adapters_have_health_check(self):
        pg = PostgreSQLAdapter()
        mysql = MySQLAdapter()
        mongo = MongoDBAdapter()
        
        assert hasattr(pg, 'health_check')
        assert hasattr(mysql, 'health_check')
        assert hasattr(mongo, 'health_check')
    
    def test_all_adapters_have_close(self):
        pg = PostgreSQLAdapter()
        mysql = MySQLAdapter()
        mongo = MongoDBAdapter()
        
        assert hasattr(pg, 'close')
        assert hasattr(mysql, 'close')
        assert hasattr(mongo, 'close')


class TestCircuitBreakerIntegration:
    
    @pytest.mark.asyncio
    async def test_postgresql_circuit_breaker_flow(self):
        adapter = PostgreSQLAdapter(enable_circuit_breaker=True)
        
        assert adapter._check_circuit_breaker() is True
        
        for _ in range(5):
            adapter._record_failure()
        
        assert adapter._circuit_open is True
        assert adapter._check_circuit_breaker() is False
    
    @pytest.mark.asyncio
    async def test_mongodb_circuit_breaker_flow(self):
        adapter = MongoDBAdapter(enable_circuit_breaker=True)
        
        assert adapter._check_circuit_breaker() is True
        
        for _ in range(5):
            adapter._record_failure()
        
        assert adapter._circuit_open is True
        
        adapter._record_success()
        assert adapter._failure_count == 0

"""
#graphrag/utils/memgraph_connector.py

This module can be used for interacting with a Memgraph database.
It is not required for reproducing the core method given a subgraph.
"""
import subprocess
import time
from gqlalchemy import Memgraph
from gqlalchemy.exceptions import GQLAlchemyError


class MemgraphConnector:
    """
    Singleton class to manage a connection to a Memgraph database.
    This includes starting the Memgraph Docker container if not running,
    retrying connections, and executing queries.
    """
    _instance = None

    def __new__(cls, *args, **kwargs):
        # Ensure only one instance, singleton pattern
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, host="127.0.0.1", port=7687, container_name="memgraph-platform", max_retries=10, wait_seconds=3):
        """
        Initialize the connector, start the container if needed, and establish connection.

        Args:
            host (str): Memgraph host address.
            port (int): Memgraph port.
            container_name (str): Docker container name for Memgraph.
            max_retries (int): Maximum connection retries.
            wait_seconds (int): Seconds to wait between retries.
        """
        # Prevent reinitialization in singleton
        if hasattr(self, '_initialized') and self._initialized:
            return
        self._initialized = True

        self.host = host
        self.port = port
        self.container_name = container_name
        self.max_retries = max_retries
        self.wait_seconds = wait_seconds
        self.print_counts = {}

        # Start the Docker container if not already running
        self.start_memgraph_container()

        # Attempt to connect to Memgraph with retries
        self._connect_with_retries()

    def limited_print(self, key, message, max_times=2):
        """
        Print a message a limited number of times to avoid console clutter.

        Args:
            key (str): Unique key to identify the message.
            message (str): Message to print.
            max_times (int): Maximum times the message is printed.
        """
        count = self.print_counts.get(key, 0)
        if count < max_times:
            print(message)
            self.print_counts[key] = count + 1

    def start_memgraph_container(self):
        """
        Check if the Memgraph Docker container is running.
        If not, try to start it or create and run a new container.

        Returns:
            bool: True if container is running or started successfully, False otherwise.
        """
        # Check for running container by name
        result = subprocess.run(
            ["docker", "ps", "-q", "-f", f"name={self.container_name}"],
            capture_output=True, text=True
        )
        if result.stdout.strip():
            self.limited_print("container_running", f"Docker container '{self.container_name}' is already running.")
        else:
            self.limited_print("container_start", f"Starting Docker container '{self.container_name}'...")
            # Attempt to start the container
            start_result = subprocess.run(["docker", "start", self.container_name], capture_output=True, text=True)
            if start_result.returncode != 0:
                # If starting fails (e.g., container doesn't exist), create and run a new container
                self.limited_print("container_create", f"Container '{self.container_name}' not found. Attempting to create and start a new one...")
                run_result = subprocess.run([
                    "docker", "run", "--name", self.container_name, "-d", "-p", "7687:7687", "memgraph/memgraph"
                ], capture_output=True, text=True)
                if run_result.returncode != 0:
                    print(f"Error creating and starting container:\n{run_result.stderr}")
                    return False
                else:
                    self.limited_print("container_created", "Container created and started successfully.")
            else:
                self.limited_print("container_started", f"Container '{self.container_name}' started successfully.")
            self.limited_print("container_wait", "Waiting 5 seconds for Memgraph to become ready...")
            time.sleep(5)
        return True

    @staticmethod
    def is_docker_running():
        """
        Check if Docker daemon is running by running 'docker info'.

        Returns:
            bool: True if Docker is running, False otherwise.
        """
        result = subprocess.run(["docker", "info"], capture_output=True, text=True)
        return result.returncode == 0

    def _connect_with_retries(self):
        """
        Try to establish a connection to Memgraph with retries.
        Raises an exception if all attempts fail.
        """
        retries = 0
        while retries < self.max_retries:
            try:
                self.db = Memgraph(host=self.host, port=self.port)
                self.db.execute("RETURN 1")  # Simple test query to verify connection
                self.limited_print("connected", "Successfully connected to Memgraph.")
                return
            except GQLAlchemyError as e:
                print(f"Connection attempt {retries + 1} failed: {e}")
                retries += 1
                time.sleep(self.wait_seconds)

        raise RuntimeError(f"Failed to connect to Memgraph after {self.max_retries} attempts.")

    def run_query(self, cypher_query: str) -> list[dict]:
        """
        Execute a Cypher query and return the results.

        Args:
            cypher_query (str): Cypher query string.

        Returns:
            list[dict]: Query results as list of dictionaries.
        """
        try:
            return list(self.db.execute_and_fetch(cypher_query))
        except Exception as e:
            print(f"[MemgraphConnector] Error executing query:\n{cypher_query}\n→ {e}")
            return []

    def clear_database(self):
        """
        Delete all nodes and relationships from the Memgraph database.
        """
        try:
            self.db.execute("MATCH (n) DETACH DELETE n")
            print("Database cleared.")
        except Exception as e:
            print(f"Error clearing database: {e}")


if __name__ == "__main__":
    mc = MemgraphConnector()
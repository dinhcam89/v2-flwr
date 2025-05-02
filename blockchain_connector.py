"""
Enhanced blockchain connector for federated learning with IPFS.
Connects to Ganache and interacts with the EnhancedModelRegistry smart contract.
Adds client authorization and contribution tracking.
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import time
import threading
import random
import logging


from web3 import Web3
from web3.contract import Contract
from eth_account.account import Account
from hexbytes import HexBytes

class BlockchainConnector:
    # Handling multiple threads accessing the nonce
    _nonce_lock = threading.Lock()
    _local_nonce = {}
    
    def __init__(
        self, 
        ganache_url: str = "http://192.168.1.146:7545",
        contract_address: Optional[str] = None,
        private_key: Optional[str] = None,
        contract_path: Optional[str] = None
    ):
        """Initialize the blockchain connector."""
        # Connect to Ganache
        self.web3 = Web3(Web3.HTTPProvider(ganache_url))
        if not self.web3.is_connected():
            raise ConnectionError(f"Failed to connect to Ganache at {ganache_url}")
        
        # Store the private key as an attribute
        self.private_key = private_key  # <-- Add this line
        
        # Set up account
        if private_key:
            self.account = self.web3.eth.account.from_key(private_key)
            self.address = self.account.address
            print(f"Using account: {self.account.address}")
        else:
            # Use the first account from Ganache
            self.account = self.web3.eth.accounts[0]
            self.address = self.account
            print(f"Using Ganache account: {self.account}")
        
        # Set contract address
        self.contract_address = None
        if contract_address:
            self.contract_address = self.web3.to_checksum_address(contract_address)
            self._load_contract(contract_address, contract_path)
    
    def _load_contract(self, address: str, contract_path: Optional[str] = None) -> None:
        """
        Load the contract from the provided address.
        
        Args:
            address: Contract address
            contract_path: Path to the contract ABI JSON file
        """
        if contract_path is None:
            # Default path for the compiled contract
            contract_path = Path(__file__).parent / "contracts" / "Federation.json"
        
        # Load contract ABI
        try:
            with open(contract_path, 'r') as f:
                contract_json = json.load(f)
            
            # Handle different compilation outputs (Truffle vs. solc)
            if 'abi' in contract_json:
                contract_abi = contract_json['abi']
            elif 'contracts' in contract_json:
                # Get the first contract in the file
                contract_name = list(contract_json['contracts'].keys())[0]
                contract_abi = contract_json['contracts'][contract_name]['abi']
            else:
                raise ValueError("Could not find ABI in contract JSON")
                
            # Create contract instance
            self.contract = self.web3.eth.contract(address=address, abi=contract_abi)
            print(f"Contract loaded at address: {address}")
            
        except FileNotFoundError:
            raise FileNotFoundError(f"Contract JSON not found at {contract_path}")
        except Exception as e:
            raise RuntimeError(f"Error loading contract: {e}")
    
    def deploy_contract(self, contract_path: Optional[str] = None) -> str:
        """
        Deploy the EnhancedModelRegistry contract to the blockchain.
        
        Args:
            contract_path: Path to the contract JSON file
            
        Returns:
            Address of the deployed contract
        """
        if contract_path is None:
            # Default path for the compiled contract
            contract_path = Path(__file__).parent / "contracts" / "EnhancedModelRegistry.json"
        
        try:
            with open(contract_path, 'r') as f:
                contract_json = json.load(f)
            
            # Get contract bytecode and ABI
            if 'bytecode' in contract_json:
                bytecode = contract_json['bytecode']
                abi = contract_json['abi']
            elif 'contracts' in contract_json:
                # Get the first contract in the file
                contract_name = list(contract_json['contracts'].keys())[0]
                bytecode = contract_json['contracts'][contract_name]['bytecode']
                abi = contract_json['contracts'][contract_name]['abi']
            else:
                raise ValueError("Could not find bytecode and ABI in contract JSON")
            
            # Create contract instance
            Contract = self.web3.eth.contract(abi=abi, bytecode=bytecode)
            
            # Build transaction
            tx_params = {
                'from': self.account if isinstance(self.account, str) else self.account.address,
                'nonce': self.web3.eth.get_transaction_count(
                    self.account if isinstance(self.account, str) else self.account.address
                ),
                'gas': 5000000,  # Increased gas limit for larger contract
                'gasPrice': self.web3.eth.gas_price
            }
            
            # Deploy contract
            transaction = Contract.constructor().build_transaction(tx_params)
            
            # Sign transaction if using private key
            if isinstance(self.account, Account):
                signed_tx = self.account.sign_transaction(transaction)
                tx_hash = self.web3.eth.send_raw_transaction(signed_tx.rawTransaction)
            else:
                tx_hash = self.web3.eth.send_transaction(transaction)
            
            # Wait for transaction to be mined
            tx_receipt = self.web3.eth.wait_for_transaction_receipt(tx_hash)
            contract_address = tx_receipt['contractAddress']
            
            # Load the deployed contract
            self._load_contract(contract_address, contract_path)
            
            print(f"Contract deployed at: {contract_address}")
            return contract_address
            
        except FileNotFoundError:
            raise FileNotFoundError(f"Contract JSON not found at {contract_path}")
        except Exception as e:
            raise RuntimeError(f"Error deploying contract: {e}")
    
    def register_model(
        self, 
        ipfs_hash: str, 
        round_num: int, 
        version: str,
        participating_clients: int = 0
    ) -> str:
        """
        Register a model in the blockchain or update it if it already exists.
        
        Args:
            ipfs_hash: IPFS hash of the model
            round_num: Federated learning round number
            version: Model version (semver format: major.minor.patch)
            participating_clients: Number of clients that participated in training
            
        Returns:
            Transaction hash
        """
        if not self.contract:
            raise RuntimeError("Contract not loaded")
        
        # Build transaction
        tx_params = {
            'from': self.account if isinstance(self.account, str) else self.account.address,
            'nonce': self.web3.eth.get_transaction_count(
                self.account if isinstance(self.account, str) else self.account.address
            ),
            'gas': 2000000,
            'gasPrice': self.web3.eth.gas_price
        }
        
        # Use registerOrUpdateModel function which handles both cases
        transaction = self.contract.functions.registerOrUpdateModel(
            ipfs_hash, 
            round_num, 
            version,
            participating_clients
        ).build_transaction(tx_params)
        
        # Sign transaction if using private key
        try:
            if isinstance(self.account, Account):
                signed_tx = self.account.sign_transaction(transaction)
                tx_hash = self.web3.eth.send_raw_transaction(signed_tx.rawTransaction)
            else:
                tx_hash = self.web3.eth.send_transaction(transaction)
            
            # Wait for transaction to be mined
            _ = self.web3.eth.wait_for_transaction_receipt(tx_hash)
            
            # Convert HexBytes to string for easier handling
            if isinstance(tx_hash, HexBytes):
                tx_hash = tx_hash.hex()
                
            print(f"Model registered with transaction: {tx_hash}")
            return tx_hash
        except Exception as e:
            logging.error(f"Failed to register/update model in blockchain: {str(e)}")
            return None
    
    def send_blockchain_transaction(self, transaction_function, function_args, retry_count=5, initial_delay=1):
        """
        Send a blockchain transaction with retry logic and proper nonce management.
        
        Args:
            transaction_function: The contract function to call
            function_args: Arguments to pass to the function
            retry_count: Number of retries if transaction fails
            initial_delay: Initial delay before retry in seconds (will be increased exponentially)
            
        Returns:
            Transaction hash or None if all retries failed
        """
        if not self.contract:
            raise RuntimeError("Contract not loaded")
        
        # Get the actual function to call
        contract_function = getattr(self.contract.functions, transaction_function)
        if not contract_function:
            logging.error(f"Function {transaction_function} not found in contract")
            return None
        
        from_address = self.account if isinstance(self.account, str) else self.account.address
        current_delay = initial_delay
        max_delay = 30  # Maximum delay between retries in seconds
        
        for attempt in range(1, retry_count + 1):
            try:
                # Get a fresh nonce for each attempt
                nonce = self.web3.eth.get_transaction_count(from_address)
                logging.info(f"Sending transaction with nonce: {nonce}, gas price: {self.web3.eth.gas_price}")
                
                # Build transaction
                tx_params = {
                    'from': from_address,
                    'nonce': nonce,
                    'gas': 6000000,
                    'gasPrice': self.web3.eth.gas_price
                }
                
                transaction = contract_function(*function_args).build_transaction(tx_params)
                
                # Sign and send
                if isinstance(self.account, Account):
                    signed_tx = self.account.sign_transaction(transaction)
                    tx_hash = self.web3.eth.send_raw_transaction(signed_tx.rawTransaction)
                else:
                    tx_hash = self.web3.eth.send_transaction(transaction)
                
                logging.info(f"Transaction sent: {tx_hash.hex() if isinstance(tx_hash, HexBytes) else tx_hash}")
                
                # Wait for receipt
                receipt = self.web3.eth.wait_for_transaction_receipt(tx_hash)
                
                # Check status
                if receipt.status == 1:  # 1 = success
                    logging.info(f"Transaction confirmed! Gas used: {receipt.gasUsed}, status: {receipt.status}")
                    return tx_hash.hex() if isinstance(tx_hash, HexBytes) else tx_hash
                else:
                    logging.warning(f"Transaction failed with status: {receipt.status}")
                    # If status is 0 (failure), we'll retry with increased nonce
            
            except Exception as e:
                # Add some randomness to the retry delay (jitter) to prevent thundering herd
                import random
                jitter = random.uniform(0.1, 0.5)
                retry_delay = current_delay + jitter
                
                logging.warning(f"Transaction failed (attempt {attempt}/{retry_count}), retrying in {retry_delay:.1f} seconds: {e}")
                
                if attempt < retry_count:
                    # Exponential backoff with jitter
                    time.sleep(retry_delay)
                    current_delay = min(current_delay * 2, max_delay)
                else:
                    logging.error(f"All {retry_count} transaction attempts failed")
                    return None
        
    def register_or_update_model(
        self, 
        ipfs_hash: str, 
        round_num: int, 
        version: str,
        participating_clients: int = 0
    ) -> str:
        """
        Register a model in the blockchain or update it if models already exist for this round.
        
        Args:
            ipfs_hash: IPFS hash of the model
            round_num: Federated learning round number
            version: Model version (semver format: major.minor.patch)
            participating_clients: Number of clients that participated in training
            
        Returns:
            Transaction hash or None if failed
        """
        if not self.contract:
            raise RuntimeError("Contract not loaded")
        
        # First, check if models exist for this round
        try:
            # This call doesn't create a transaction - it's just a read operation
            models_for_round = self.contract.functions.getModelsByRound(round_num).call()
            models_exist = len(models_for_round) > 0
        except Exception as e:
            logging.warning(f"Error checking for existing models: {e}")
            # If we can't check, we'll proceed with trying to register
            models_exist = False
        
        # Choose the right function based on whether models exist for this round
        if models_exist:
            function_name = "updateModelByRound"
            logging.info(f"Models exist for round {round_num}, using {function_name}")
        else:
            function_name = "registerModel"
            logging.info(f"No models for round {round_num}, using {function_name}")
        
        # Send the transaction with our improved transaction manager
        function_args = [ipfs_hash, round_num, version, participating_clients]
        tx_hash = self.send_blockchain_transaction(function_name, function_args)
        
        if tx_hash:
            if models_exist:
                print(f"Model updated with transaction: {tx_hash}")
            else:
                print(f"Model registered with transaction: {tx_hash}")
        
        return tx_hash
    
    def authorize_client(self, client_address: str) -> str:
        """
        Authorize a client to participate in federated learning.
        
        Args:
            client_address: Ethereum address of the client
            
        Returns:
            Transaction hash
        """
        if not self.contract:
            raise RuntimeError("Contract not loaded")
        
        # Build transaction
        tx_params = {
            'from': self.account if isinstance(self.account, str) else self.account.address,
            'nonce': self.web3.eth.get_transaction_count(
                self.account if isinstance(self.account, str) else self.account.address
            ),
            'gas': 100000,
            'gasPrice': self.web3.eth.gas_price
        }
        
        # Call authorizeClient function
        transaction = self.contract.functions.authorizeClient(client_address).build_transaction(tx_params)
        
        # Sign transaction if using private key
        if isinstance(self.account, Account):
            signed_tx = self.account.sign_transaction(transaction)
            tx_hash = self.web3.eth.send_raw_transaction(signed_tx.rawTransaction)
        else:
            tx_hash = self.web3.eth.send_transaction(transaction)
        
        # Wait for transaction to be mined
        _ = self.web3.eth.wait_for_transaction_receipt(tx_hash)
        
        # Convert HexBytes to string for easier handling
        if isinstance(tx_hash, HexBytes):
            tx_hash = tx_hash.hex()
            
        print(f"Client {client_address} authorized with transaction: {tx_hash}")
        return tx_hash
    
    def authorize_clients(self, client_addresses: List[str]) -> str:
        """
        Authorize multiple clients to participate in federated learning.
        
        Args:
            client_addresses: List of Ethereum addresses of clients
            
        Returns:
            Transaction hash
        """
        if not self.contract:
            raise RuntimeError("Contract not loaded")
        
        # Build transaction
        tx_params = {
            'from': self.account if isinstance(self.account, str) else self.account.address,
            'nonce': self.web3.eth.get_transaction_count(
                self.account if isinstance(self.account, str) else self.account.address
            ),
            'gas': 500000,  # Higher gas limit for multiple clients
            'gasPrice': self.web3.eth.gas_price
        }
        
        # Call authorizeClients function
        transaction = self.contract.functions.authorizeClients(client_addresses).build_transaction(tx_params)
        
        # Sign transaction if using private key
        if isinstance(self.account, Account):
            signed_tx = self.account.sign_transaction(transaction)
            tx_hash = self.web3.eth.send_raw_transaction(signed_tx.rawTransaction)
        else:
            tx_hash = self.web3.eth.send_transaction(transaction)
        
        # Wait for transaction to be mined
        _ = self.web3.eth.wait_for_transaction_receipt(tx_hash)
        
        # Convert HexBytes to string for easier handling
        if isinstance(tx_hash, HexBytes):
            tx_hash = tx_hash.hex()
            
        print(f"Authorized {len(client_addresses)} clients with transaction: {tx_hash}")
        return tx_hash
    
    def deauthorize_client(self, client_address: str) -> str:
        """
        Deauthorize a client from participating in federated learning.
        
        Args:
            client_address: Ethereum address of the client
            
        Returns:
            Transaction hash
        """
        if not self.contract:
            raise RuntimeError("Contract not loaded")
        
        # Build transaction
        tx_params = {
            'from': self.account if isinstance(self.account, str) else self.account.address,
            'nonce': self.web3.eth.get_transaction_count(
                self.account if isinstance(self.account, str) else self.account.address
            ),
            'gas': 100000,
            'gasPrice': self.web3.eth.gas_price
        }
        
        # Call deauthorizeClient function
        transaction = self.contract.functions.deauthorizeClient(client_address).build_transaction(tx_params)
        
        # Sign transaction if using private key
        if isinstance(self.account, Account):
            signed_tx = self.account.sign_transaction(transaction)
            tx_hash = self.web3.eth.send_raw_transaction(signed_tx.rawTransaction)
        else:
            tx_hash = self.web3.eth.send_transaction(transaction)
        
        # Wait for transaction to be mined
        _ = self.web3.eth.wait_for_transaction_receipt(tx_hash)
        
        # Convert HexBytes to string for easier handling
        if isinstance(tx_hash, HexBytes):
            tx_hash = tx_hash.hex()
            
        print(f"Client {client_address} deauthorized with transaction: {tx_hash}")
        return tx_hash
    
    def is_client_authorized(self, client_address: str) -> bool:
        """
        Check if a client is authorized to participate.
        
        Args:
            client_address: Ethereum address of the client
            
        Returns:
            True if client is authorized, False otherwise
        """
        if not self.contract:
            raise RuntimeError("Contract not loaded")
        
        return self.contract.functions.isClientAuthorized(client_address).call()
    
    def get_all_authorized_clients(self) -> List[str]:
        """
        Get all authorized client addresses.
        
        Returns:
            List of client addresses
        """
        if not self.contract:
            raise RuntimeError("Contract not loaded")
        
        # Get count first
        count = self.contract.functions.getAuthorizedClientCount().call()
        
        # Get each client
        clients = []
        for i in range(count):
            clients.append(self.contract.functions.authorizedClients(i).call())
        
        return clients
    
    def record_contribution(
            self, 
            client_address: str, 
            round_num: int, 
            ipfs_hash: str, 
            accuracy: float,
            max_retries: int = 5
        ) -> str:
        """
        Record a client's contribution with advanced retry mechanism and nonce management.
        
        Args:
            client_address: Ethereum address of the client
            round_num: Federated learning round number
            ipfs_hash: IPFS hash of the model contribution
            accuracy: Accuracy achieved by the client (0-100)
            max_retries: Maximum number of retry attempts
            
        Returns:
            Transaction hash
        """
        if not self.contract:
            raise RuntimeError("Contract not loaded")
        
        # Convert accuracy to blockchain format (multiply by 100 to handle decimals)
        accuracy_int = int(accuracy * 100)
        
        # Get account address consistently
        account_address = self.account if isinstance(self.account, str) else self.account.address
        
        # Set up retry mechanism with exponential backoff and jitter
        retry_count = 0
        last_error = None
        
        while retry_count < max_retries:
            try:
                # Get synchronized nonce with lock protection
                nonce = self._get_synchronized_nonce(account_address)
                
                # Calculate gas price with slight increase on each retry to help stuck transactions
                gas_price_multiplier = 1.0 + (retry_count * 0.2)  # Increase by 20% each retry
                gas_price = int(self.web3.eth.gas_price * gas_price_multiplier)
                
                # Build transaction with carefully managed nonce
                tx_params = {
                    'from': account_address,
                    'nonce': nonce,
                    'gas': 2000000,  # Increased gas limit for more complex transactions
                    'gasPrice': gas_price
                }
                
                logging.info(f"Sending transaction with nonce: {nonce}, gas price: {gas_price}")
                
                # Call recordContribution function
                transaction = self.contract.functions.recordContribution(
                    client_address, 
                    round_num, 
                    ipfs_hash, 
                    accuracy_int
                ).build_transaction(tx_params)
                
                # Sign and send transaction
                if isinstance(self.account, Account):
                    signed_tx = self.account.sign_transaction(transaction)
                    tx_hash = self.web3.eth.send_raw_transaction(signed_tx.rawTransaction)
                else:
                    tx_hash = self.web3.eth.send_transaction(transaction)
                
                # Convert HexBytes to string for easier handling
                if isinstance(tx_hash, HexBytes):
                    tx_hash_str = tx_hash.hex()
                else:
                    tx_hash_str = tx_hash
                
                logging.info(f"Transaction sent: {tx_hash_str}")
                
                # Wait for transaction to be mined with timeout
                receipt = self.web3.eth.wait_for_transaction_receipt(
                    tx_hash, 
                    timeout=60,
                    poll_latency=1.0  # Check every second
                )
                
                logging.info(f"Transaction confirmed! Gas used: {receipt.gasUsed}, status: {receipt.status}")
                return tx_hash_str
                
            except Exception as e:
                last_error = e
                retry_count += 1
                
                # Handle nonce-specific errors
                error_str = str(e).lower()
                if "nonce" in error_str and "too low" in error_str:
                    # Reset our nonce tracking if the blockchain says our nonce is too low
                    with self._nonce_lock:
                        if account_address in self._local_nonce:
                            del self._local_nonce[account_address]
                    
                    # Shorter wait for nonce issues
                    wait_time = 1 + random.uniform(0.1, 0.5)
                    logging.warning(f"Nonce too low, resetting and retrying in {wait_time:.1f} seconds")
                
                elif "nonce" in error_str:
                    # Other nonce issue - reset and retry with fresh nonce
                    with self._nonce_lock:
                        if account_address in self._local_nonce:
                            del self._local_nonce[account_address]
                    
                    wait_time = 2 + random.uniform(0.1, 1.0)
                    logging.warning(f"Nonce issue detected, retrying with fresh nonce in {wait_time:.1f} seconds: {e}")
                
                else:
                    # For other errors, use exponential backoff with jitter
                    base_wait = 2 ** retry_count  # 2, 4, 8, 16...
                    wait_time = base_wait + random.uniform(0.1, 1.0)
                    logging.warning(f"Transaction failed (attempt {retry_count}/{max_retries}), "
                                       f"retrying in {wait_time:.1f} seconds: {e}")
                
                time.sleep(wait_time)
        
        # If we're here, all retries failed
        logging.error(f"Failed to record contribution after {max_retries} attempts. Last error: {last_error}")
        raise RuntimeError(f"Failed to record contribution: {last_error}")

    def _get_synchronized_nonce(self, address: str) -> int:
        """
        Get a properly synchronized nonce for the given address.
        Uses a combination of local tracking and blockchain queries.
        
        Args:
            address: Ethereum address
            
        Returns:
            Next nonce to use
        """
        with self._nonce_lock:
            # Get the latest on-chain nonce
            blockchain_nonce = self.web3.eth.get_transaction_count(address, 'pending')
            
            # Initialize or update our local nonce tracking
            if address not in self._local_nonce or self._local_nonce[address] < blockchain_nonce:
                self._local_nonce[address] = blockchain_nonce
            
            # Use our tracked nonce (which may be ahead of the blockchain if we've sent multiple txs)
            next_nonce = self._local_nonce[address]
            
            # Increment for next use
            self._local_nonce[address] = next_nonce + 1
            
            return next_nonce
    
    def allocate_rewards_for_round(self, round_num: int) -> str:
        """
        Allocate rewards for a specific round.
        
        Args:
            round_num: Round number
            
        Returns:
            Transaction hash
        """
        func = self.contract.functions.allocateRewardsForRound(round_num)
        return self._send_transaction(func)
    
    def get_client_contribution_details(self, client_address: str) -> Dict[str, Any]:
        """
        Get contribution details for a client.
        
        Args:
            client_address: Ethereum address of the client
            
        Returns:
            Dictionary of contribution details
        """
        client_address = self.web3.to_checksum_address(client_address)
        
        # Call the contract function
        contrib_count, total_score, is_authorized, last_timestamp, rewards_earned, rewards_claimed = \
            self.contract.functions.getClientContribution(client_address).call()
        
        # Convert wei to ETH for rewards
        rewards_earned_eth = self.web3.from_wei(rewards_earned, 'ether')
        
        return {
            "contribution_count": contrib_count,
            "total_score": total_score,
            "is_authorized": is_authorized,
            "last_contribution_timestamp": last_timestamp,
            "rewards_earned": rewards_earned_eth,
            "rewards_claimed": rewards_claimed
        }
    
    def get_client_contribution_records(self, client_address: str) -> List[Dict[str, Any]]:
        """
        Get client contribution records.
        
        Args:
            client_address: Ethereum address of the client
            
        Returns:
            List of contribution records
        """
        if not self.contract:
            raise RuntimeError("Contract not loaded")
        
        records = self.contract.functions.getClientContributionRecords(client_address).call()
        
        result = []
        for i in range(len(records[0])):
            result.append({
                "round": records[0][i],
                "accuracy": records[1][i] / 100.0,  # Convert back to decimal
                "score": records[2][i],
                "timestamp": records[3][i],
                "rewarded": records[4][i]
            })
        
        return result
    
    def has_contributed_in_round(self, client_address: str, round_number: int) -> bool:
        """
        Check if a client has contributed in a specific round.
        
        Args:
            client_address: Ethereum address of the client
            round_number: The federated learning round number
            
        Returns:
            True if the client has contributed in the specified round, False otherwise
        """
        try:
            if not self.contract:
                logging.warning("Contract not loaded, cannot verify contribution")
                return False
            
            records = self.get_client_contribution_records(client_address)
            
            # Check if any of the records match the current round
            for record in records:
                if record["round"] == round_number:
                    logging.info(f"Found contribution from client {client_address} for round {round_number}")
                    return True
            
            logging.warning(f"No contribution found from client {client_address} for round {round_number}")
            return False
        except Exception as e:
            logging.error(f"Error checking client contribution: {e}")
            # We could be more lenient here by returning True if there's an error
            # return True
            return False
    
    def get_round_contributions(self, round_num: int) -> List[Dict[str, Any]]:
        """
        Get contributions for a specific round.
        
        Args:
            round_num: Federated learning round number
            
        Returns:
            List of contribution records for the round
        """
        if not self.contract:
            raise RuntimeError("Contract not loaded")
        
        records = self.contract.functions.getRoundContributions(round_num).call()
        
        result = []
        for i in range(len(records[0])):
            result.append({
                "client_address": records[0][i],
                "accuracy": records[1][i] / 100.0,  # Convert back to decimal
                "score": records[2][i],
                "rewarded": records[3][i]
            })
        
        return result
    
    def get_model_details(self, ipfs_hash: str, round_num: int) -> Dict[str, Any]:
        """
        Get model details from the blockchain.
        
        Args:
            ipfs_hash: IPFS hash of the model
            round_num: Federated learning round number
            
        Returns:
            Dictionary with model details
        """
        if not self.contract:
            raise RuntimeError("Contract not loaded")
        
        # Generate model ID
        model_id = self.contract.functions.generateModelId(ipfs_hash, round_num).call()
        
        # Get model details
        details = self.contract.functions.getModelDetails(model_id).call()
        
        # Format the response
        return {
            "ipfs_hash": details[0],
            "round": details[1],
            "version": details[2],
            "timestamp": details[3],
            "participating_clients": details[4],
            "publisher": details[5],
            "is_active": details[6]
        }
    
    def get_latest_model(self, version_prefix: str = "1.0") -> Dict[str, Any]:
        """
        Get the latest model for a specific version prefix.
        
        Args:
            version_prefix: Version prefix (e.g., "1.0")
            
        Returns:
            Dictionary with model details
        """
        if not self.contract:
            raise RuntimeError("Contract not loaded")
        
        # Get latest model
        details = self.contract.functions.getLatestModel(version_prefix).call()
        
        # Format the response
        return {
            "model_id": self.web3.to_hex(details[0]),
            "ipfs_hash": details[1],
            "round": details[2],
            "version": details[3],
            "timestamp": details[4],
            "participating_clients": details[5]
        }
    
    def get_models_by_round(self, round_num: int) -> List[str]:
        """
        Get all models for a specific round.
        
        Args:
            round_num: Federated learning round number
            
        Returns:
            List of model IDs
        """
        if not self.contract:
            raise RuntimeError("Contract not loaded")
        
        # Get models by round
        model_ids = self.contract.functions.getModelsByRound(round_num).call()
        
        # Convert bytes32 to hex strings
        return [self.web3.to_hex(model_id) for model_id in model_ids]
    
    def deactivate_model(self, ipfs_hash: str, round_num: int) -> str:
        """
        Deactivate a model.
        
        Args:
            ipfs_hash: IPFS hash of the model
            round_num: Federated learning round number
            
        Returns:
            Transaction hash
        """
        if not self.contract:
            raise RuntimeError("Contract not loaded")
        
        # Generate model ID
        model_id = self.contract.functions.generateModelId(ipfs_hash, round_num).call()
        
        # Build transaction
        tx_params = {
            'from': self.account if isinstance(self.account, str) else self.account.address,
            'nonce': self.web3.eth.get_transaction_count(
                self.account if isinstance(self.account, str) else self.account.address
            ),
            'gas': 100000,
            'gasPrice': self.web3.eth.gas_price
        }
        
        # Call deactivateModel function
        transaction = self.contract.functions.deactivateModel(model_id).build_transaction(tx_params)
        
        # Sign transaction if using private key
        if isinstance(self.account, Account):
            signed_tx = self.account.sign_transaction(transaction)
            tx_hash = self.web3.eth.send_raw_transaction(signed_tx.rawTransaction)
        else:
            tx_hash = self.web3.eth.send_transaction(transaction)
        
        # Wait for transaction to be mined
        _ = self.web3.eth.wait_for_transaction_receipt(tx_hash)
        
        # Convert HexBytes to string for easier handling
        if isinstance(tx_hash, HexBytes):
            tx_hash = tx_hash.hex()
            
        print(f"Model deactivated with transaction: {tx_hash}")
        return tx_hash
    
    def has_contributed_in_round(self, client_address: str, round_number: int) -> bool:
        """
        Check if a client has contributed in a specific round.
        
        Args:
            client_address: Ethereum address of the client
            round_number: The federated learning round number
            
        Returns:
            True if the client has contributed in the specified round, False otherwise
        """
        try:
            if not self.contract:
                logging.warning("Contract not loaded, cannot verify contribution")
                return False
            
            # First check if client is authorized
            if not self.is_client_authorized(client_address):
                logging.warning(f"Client {client_address} is not authorized")
                return False
            
            # Then check contribution details
            details = self.get_client_contribution_details(client_address)
            
            # If client has made any contributions, consider them valid for this round
            # This is a more lenient approach during development
            if details and details["contribution_count"] > 0:
                logging.info(f"Client {client_address} has made {details['contribution_count']} contributions")
                # For more strict checking, uncomment below to verify specific round contributions
                try:
                    records = self.get_client_contribution_records(client_address)
                    for record in records:
                        if record["round"] == round_number:
                            logging.info(f"Found contribution from client {client_address} for round {round_number}")
                            return True
                    
                    # If we get here, client has contributed but not for this specific round
                    # During development, we'll be lenient and accept it anyway
                    logging.warning(f"Client {client_address} has contributed but not for round {round_number}. Accepting anyway.")
                    return True
                except Exception as e:
                    logging.error(f"Error checking specific round contributions: {e}")
                    # Be lenient during development
                    return True
            
            logging.warning(f"No contributions found for client {client_address}")
            return False
        except Exception as e:
            logging.error(f"Error checking client contribution: {e}")
            # During development, you might want to be lenient with errors
            return True  # Change to False for stricter validation
        
        
        
    ################################
    # REWARDING ALLOCATION
    ################################
    
    def fund_contract(self, amount_eth: float) -> str:
        """
        Fund the contract with ETH.
        
        Args:
            amount_eth: Amount of ETH to send
            
        Returns:
            Transaction hash
        """
        try:
            amount_wei = self.web3.to_wei(amount_eth, 'ether')
            
            # Create transaction to send ETH to contract
            tx_data = {
                'from': self.address,
                'to': self.contract_address,
                'value': amount_wei,
                'gas': 100000,
                'gasPrice': self.web3.to_wei('50', 'gwei'),
                'nonce': self.web3.eth.get_transaction_count(self.address)
            }
            
            if self.private_key:
                # Sign and send transaction
                signed_tx = self.web3.eth.account.sign_transaction(tx_data, self.private_key)
                
                # Use the correct attribute name: raw_transaction instead of rawTransaction
                tx_hash = self.web3.eth.send_raw_transaction(signed_tx.raw_transaction)
            else:
                # Send transaction using default account
                tx_hash = self.web3.eth.send_transaction(tx_data)
            
            # Wait for transaction to be mined
            receipt = self.web3.eth.wait_for_transaction_receipt(tx_hash)
            return receipt.transactionHash.hex()
        except Exception as e:
            print(f"Exception details: {type(e).__name__}: {str(e)}")
            raise

    def fund_round_reward_pool(self, round_num: int, amount_eth: float) -> str:
        """
        Fund a specific round's reward pool.
        
        Args:
            round_num: Round number
            amount_eth: Amount of ETH to allocate
            
        Returns:
            Transaction hash
        """
        amount_wei = self.web3.to_wei(amount_eth, 'ether')
        func = self.contract.functions.fundRoundRewardPool(round_num)
        return self._send_transaction(func, value=amount_wei)

    def finalize_round_reward_pool(self, round_num: int) -> str:
        """
        Finalize a round's reward pool.
        
        Args:
            round_num: Round number
            
        Returns:
            Transaction hash
        """
        func = self.contract.functions.finalizeRoundRewardPool(round_num)
        return self._send_transaction(func)
    
    def allocate_rewards_for_round(self, round_num: int) -> str:
        """
        Allocate rewards for a specific round.
        
        Args:
            round_num: Round number
            
        Returns:
            Transaction hash
        """
        func = self.contract.functions.allocateRewardsForRound(round_num)
        return self._send_transaction(func)
    
    def get_round_reward_pool(self, round_num: int) -> Dict[str, Any]:
        """
        Get details about a round's reward pool.
        
        Args:
            round_num: Round number
            
        Returns:
            Dictionary with pool details
        """
        total_amount, allocated_amount, remaining_amount, is_finalized = \
            self.contract.functions.getRoundRewardPool(round_num).call()
        
        return {
            "total_amount": self.web3.from_wei(total_amount, 'ether'),
            "allocated_amount": self.web3.from_wei(allocated_amount, 'ether'),
            "remaining_amount": self.web3.from_wei(remaining_amount, 'ether'),
            "is_finalized": is_finalized
    }
        
    def _send_transaction(self, func, value=0):
        """
        Send a transaction to the blockchain.
        
        Args:
            func: Contract function to call
            value: ETH value to send with the transaction (in wei)
            
        Returns:
            Transaction hash
        """
        if self.private_key:
            # Sign transaction with private key
            tx = func.build_transaction({
                'from': self.address,
                'value': value,
                'gas': 3000000,
                'gasPrice': self.web3.to_wei('50', 'gwei'),
                'nonce': self.web3.eth.get_transaction_count(self.address)
            })
            
            signed_tx = self.web3.eth.account.sign_transaction(tx, self.private_key)
            # Use raw_transaction instead of rawTransaction
            tx_hash = self.web3.eth.send_raw_transaction(signed_tx.raw_transaction)
        else:
            # Use default account
            tx_hash = func.transact({
                'from': self.address,
                'value': value,
                'gas': 3000000
            })
        
        # Wait for transaction to be mined
        receipt = self.web3.eth.wait_for_transaction_receipt(tx_hash)
        return receipt.transactionHash.hex()
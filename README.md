# FL-v2

Welcome to the **FL-v2** repository! This project is primarily written in Python (98.3%) with a small HTML component (1.7%). It serves as a robust and scalable solution for managing federated learning systems with blockchain integration for secure and incentivized collaboration.

This repository is maintained and developed by [@dinhcam89](https://github.com/dinhcam89).

---

## Key Highlights

- **Python-based**: The core functionality is built using Python, ensuring flexibility and ease of development.
- **Blockchain Integration**: Features Ethereum smart contracts for secure and incentivized collaboration between clients.
- **IPFS Support**: Leverages IPFS for decentralized storage and exchange of federated learning models.
- **Metrics Visualization**: Provides tools for visualizing key performance metrics of federated learning systems.
- **Command-Line Interface**: Includes a CLI to manage clients, contributions, and blockchain interactions.

---

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Ganache or any Ethereum-compatible blockchain setup
- IPFS node for decentralized storage

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/dinhcam89/FL-v2.git
   cd FL-v2
   ```

2. Set up a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Configuration

Update the configuration settings in the relevant Python scripts (e.g., `blockchain_connector.py`) for:
- Blockchain URL
- Smart contract address
- IPFS API URL

---

## Usage

### Running the CLI Tool

The `client_manager.py` script provides a command-line interface for managing clients and blockchain interactions. Below are some key commands:

- **List all clients**:
  ```bash
  python client_manager.py list
  ```

- **Authorize a client**:
  ```bash
  python client_manager.py authorize --client_address [Ethereum Address]
  ```

- **Fund the contract**:
  ```bash
  python client_manager.py fund-contract --amount_eth [Amount]
  ```

### Visualizing Metrics

The `metrics_visualization.py` script generates visualizations for federated learning metrics:
```bash
python metrics_visualization.py --metrics metrics/metrics_history.json
```

---

## Contributing

We welcome contributions from the community! To contribute:

1. Fork the repository.
2. Create a branch for your feature (`git checkout -b feature-name`).
3. Commit your changes (`git commit -m "Add feature"`).
4. Push the branch to your fork (`git push origin feature-name`).
5. Submit a pull request for review.

For detailed contributing guidelines, please see [CONTRIBUTING.md](CONTRIBUTING.md) (if applicable).

---

## License

This project is licensed under the [MIT License](LICENSE).

---

## Contact

If you have any questions, suggestions, or feedback, feel free to contact [@dinhcam89](https://github.com/dinhcam89) or open an issue in this repository.

We hope you find **FL-v2** useful and appreciate your interest in contributing to its growth!

---

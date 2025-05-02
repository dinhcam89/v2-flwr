import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import ipfsapi
import matplotlib.dates as mdates
from matplotlib.ticker import MaxNLocator
from web3 import Web3

class FederatedLearningVisualizer:
    """
    A visualization tool for federated learning metrics and blockchain data.
    """
    
    def __init__(
        self, 
        metrics_path="metrics/metrics_history.json",
        client_stats_path="metrics/client_stats.json",
        output_dir="visualizations",
        ipfs_url="http://127.0.0.1:5001",
        ganache_url="http://127.0.0.1:7545",
        contract_address=None
    ):
        """
        Initialize the visualizer with paths to metrics files.
        
        Args:
            metrics_path: Path to metrics history JSON file
            client_stats_path: Path to client stats JSON file
            output_dir: Directory to save visualization outputs
            ipfs_url: IPFS API URL
            ganache_url: Ganache blockchain URL
            contract_address: Federation contract address
        """
        self.metrics_path = metrics_path
        self.client_stats_path = client_stats_path
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Load metrics data
        self.metrics_data = []
        if os.path.exists(metrics_path):
            try:
                with open(metrics_path, 'r') as f:
                    self.metrics_data = json.load(f)
                print(f"Loaded metrics data from {metrics_path}")
            except Exception as e:
                print(f"Error loading metrics data: {e}")
        
        # Load client stats data
        self.client_stats = {}
        if os.path.exists(client_stats_path):
            try:
                with open(client_stats_path, 'r') as f:
                    self.client_stats = json.load(f)
                print(f"Loaded client stats from {client_stats_path}")
            except Exception as e:
                print(f"Error loading client stats: {e}")
        
        # Set plot style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 12
        
        # Try to connect to IPFS
        self.ipfs = None
        try:
            # Parse the URL to get host and port
            from urllib.parse import urlparse
            parsed_url = urlparse(ipfs_url)
            host = parsed_url.hostname or '127.0.0.1'
            port = parsed_url.port or 5001
            
            self.ipfs = ipfsapi.connect(host, port)
            print(f"Connected to IPFS at {ipfs_url}")
        except Exception as e:
            print(f"Failed to connect to IPFS: {e}")
            print("Continuing without IPFS connection")
        
        # Try to connect to blockchain
        self.web3 = None
        self.contract = None
        if ganache_url and contract_address:
            try:
                self.web3 = Web3(Web3.HTTPProvider(ganache_url))
                # Would need ABI to fully initialize contract
                print(f"Connected to blockchain at {ganache_url}")
            except Exception as e:
                print(f"Failed to connect to blockchain: {e}")
    
    def parse_metrics_data(self):
        """Parse metrics data into training and evaluation dataframes."""
        # Extract training and evaluation rounds
        training_rounds = []
        eval_rounds = []
        
        for item in self.metrics_data:
            # Check if it's a training or evaluation record
            if "metrics" in item and "fit" not in item.get("eval_metrics", {}):
                # Training record
                round_num = item.get("round", 0)
                num_clients = item.get("num_clients", 0)
                timestamp = item.get("timestamp", "")
                
                metrics = item.get("metrics", {})
                auth_clients = metrics.get("authorized_clients", 0)
                unauth_clients = metrics.get("unauthorized_clients", 0)
                total_clients = metrics.get("total_clients", 0)
                
                training_rounds.append({
                    "round": round_num,
                    "num_clients": num_clients,
                    "timestamp": timestamp,
                    "authorized_clients": auth_clients,
                    "unauthorized_clients": unauth_clients,
                    "total_clients": total_clients
                })
            
            elif "eval_metrics" in item:
                # Evaluation record
                round_num = item.get("round", 0)
                eval_loss = item.get("eval_loss", 0)
                timestamp = item.get("timestamp", "")
                
                eval_metrics = item.get("eval_metrics", {})
                avg_accuracy = eval_metrics.get("avg_accuracy", 0)
                auth_clients = eval_metrics.get("authorized_clients", 0)
                unauth_clients = eval_metrics.get("unauthorized_clients", 0)
                total_clients = eval_metrics.get("total_clients", 0)
                
                eval_rounds.append({
                    "round": round_num,
                    "eval_loss": eval_loss,
                    "timestamp": timestamp,
                    "avg_accuracy": avg_accuracy,
                    "authorized_clients": auth_clients,
                    "unauthorized_clients": unauth_clients,
                    "total_clients": total_clients
                })
        
        # Convert to dataframes
        train_df = pd.DataFrame(training_rounds)
        eval_df = pd.DataFrame(eval_rounds)
        
        if not train_df.empty:
            # Convert timestamp to datetime
            train_df['datetime'] = pd.to_datetime(train_df['timestamp'])
            train_df = train_df.sort_values('round')
        
        if not eval_df.empty:
            # Convert timestamp to datetime
            eval_df['datetime'] = pd.to_datetime(eval_df['timestamp'])
            eval_df = eval_df.sort_values('round')
        
        return train_df, eval_df
    
    def parse_client_stats(self):
        """Parse client stats data into a dataframe."""
        client_records = []
        
        for wallet_address, stats in self.client_stats.items():
            details = stats.get("details", {})
            records = stats.get("records", {})
            
            contribution_count = details.get("contribution_count", 0)
            total_score = details.get("total_score", 0)
            is_authorized = details.get("is_authorized", False)
            last_time = details.get("last_contribution_timestamp", 0)
            rewards_earned = details.get("rewards_earned", 0)
            rewards_claimed = details.get("rewards_claimed", False)
            
            # Check if records is a list or dictionary
            if isinstance(records, list):
                # Handle the case where records is a list
                for record in records:
                    if isinstance(record, dict):
                        client_records.append({
                            "wallet_address": wallet_address,
                            "round": record.get("round", 0),
                            "accuracy": record.get("accuracy", 0),
                            "score": record.get("score", 0),
                            "timestamp": record.get("timestamp", 0),
                            "rewarded": record.get("rewarded", False),
                            "contribution_count": contribution_count,
                            "total_score": total_score,
                            "is_authorized": is_authorized,
                            "rewards_earned": rewards_earned,
                            "rewards_claimed": rewards_claimed
                        })
            else:
                # Handle the case where records is a dictionary
                rounds = records.get("rounds", [])
                accuracies = records.get("accuracies", [])
                scores = records.get("scores", [])
                timestamps = records.get("timestamps", [])
                rewarded = records.get("rewarded", [])
                
                # Create per-round records
                for i in range(len(rounds)):
                    if i < len(accuracies) and i < len(scores) and i < len(timestamps) and i < len(rewarded):
                        client_records.append({
                            "wallet_address": wallet_address,
                            "round": rounds[i],
                            "accuracy": accuracies[i],
                            "score": scores[i],
                            "timestamp": timestamps[i],
                            "rewarded": rewarded[i],
                            "contribution_count": contribution_count,
                            "total_score": total_score,
                            "is_authorized": is_authorized,
                            "rewards_earned": rewards_earned,
                            "rewards_claimed": rewards_claimed
                        })
        
        # Convert to dataframe
        df = pd.DataFrame(client_records)
        
        if not df.empty:
            # Convert timestamp to datetime
            if 'timestamp' in df.columns:
                df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
            df = df.sort_values(['round', 'wallet_address'])
        
        return df
    
    def plot_training_loss(self, train_df, eval_df):
        """Plot training and evaluation loss over rounds."""
        plt.figure(figsize=(14, 8))
        
        if not eval_df.empty and 'eval_loss' in eval_df.columns:
            plt.plot(eval_df['round'], eval_df['eval_loss'], 'o-', color='#1f77b4', linewidth=2, label='Evaluation Loss')
        
        plt.title('Federated Learning Loss per Round', fontsize=18)
        plt.xlabel('Round', fontsize=14)
        plt.ylabel('Loss', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=12)
        
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        
        # Add annotations
        if not eval_df.empty and 'eval_loss' in eval_df.columns:
            for i, row in eval_df.iterrows():
                plt.annotate(f"{row['eval_loss']:.2f}", 
                            (row['round'], row['eval_loss']),
                            xytext=(5, 5),
                            textcoords='offset points')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'training_loss.png'), dpi=300)
        plt.close()
    
    def plot_evaluation_accuracy(self, eval_df):
        """Plot evaluation accuracy over rounds."""
        if eval_df.empty or 'avg_accuracy' not in eval_df.columns:
            print("No evaluation accuracy data available")
            return
        
        plt.figure(figsize=(14, 8))
        
        plt.plot(eval_df['round'], eval_df['avg_accuracy'], 'o-', color='#2ca02c', linewidth=2)
        
        plt.title('Federated Learning Accuracy per Round', fontsize=18)
        plt.xlabel('Round', fontsize=14)
        plt.ylabel('Accuracy (%)', fontsize=14)
        plt.grid(True, alpha=0.3)
        
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        
        # Add annotations
        for i, row in eval_df.iterrows():
            plt.annotate(f"{row['avg_accuracy']:.1f}%", 
                        (row['round'], row['avg_accuracy']),
                        xytext=(5, 5),
                        textcoords='offset points')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'evaluation_accuracy.png'), dpi=300)
        plt.close()
    
    def plot_client_participation(self, train_df):
        """Plot client participation over rounds."""
        if train_df.empty:
            print("No training data available")
            return
        
        plt.figure(figsize=(14, 8))
        
        bar_width = 0.35
        index = np.arange(len(train_df))
        
        if 'authorized_clients' in train_df.columns and 'unauthorized_clients' in train_df.columns:
            plt.bar(index, train_df['authorized_clients'], bar_width, label='Authorized', color='#2ca02c')
            plt.bar(index, train_df['unauthorized_clients'], bar_width, bottom=train_df['authorized_clients'], 
                   label='Unauthorized', color='#d62728')
            
            # Add total as text
            for i, row in train_df.iterrows():
                total = row['authorized_clients'] + row['unauthorized_clients']
                plt.text(i, total + 0.1, f"{total}", ha='center')
        
        plt.title('Client Participation per Round', fontsize=18)
        plt.xlabel('Round', fontsize=14)
        plt.ylabel('Number of Clients', fontsize=14)
        plt.xticks(index, train_df['round'])
        plt.legend(fontsize=12)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'client_participation.png'), dpi=300)
        plt.close()
    
    def plot_client_contributions(self, client_df):
        """Plot client contributions and rewards."""
        if client_df.empty:
            print("No client data available")
            return
        
        # Get unique clients and rounds
        clients = client_df['wallet_address'].unique()
        rounds = sorted(client_df['round'].unique())
        
        # Create a pivot table of accuracy by client and round
        pivot_df = client_df.pivot_table(
            index='wallet_address', 
            columns='round', 
            values='accuracy'
        ).fillna(0)
        
        # Plot heatmap of client accuracies
        plt.figure(figsize=(max(12, len(rounds) * 2), max(8, len(clients) * 0.8)))
        
        sns.heatmap(pivot_df, annot=True, cmap='YlGnBu', fmt='.1f',
                   linewidths=.5, cbar_kws={'label': 'Accuracy'})
        
        plt.title('Client Contribution Accuracy by Round', fontsize=18)
        plt.xlabel('Round', fontsize=14)
        plt.ylabel('Client Wallet Address', fontsize=14)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'client_accuracy_heatmap.png'), dpi=300)
        plt.close()
        
        # Plot client rewards if available
        if 'rewards_earned' in client_df.columns:
            # Group by client and get latest record for each
            latest_client_stats = client_df.sort_values('round').groupby('wallet_address').last().reset_index()
            
            plt.figure(figsize=(14, max(8, len(clients) * 0.8)))
            
            # Plot rewards earned
            bars = plt.barh(latest_client_stats['wallet_address'], latest_client_stats['rewards_earned'], color='#1f77b4')
            
            plt.title('Total Rewards Earned by Clients', fontsize=18)
            plt.xlabel('Rewards', fontsize=14)
            plt.ylabel('Client Wallet Address', fontsize=14)
            
            # Add reward values as text
            for bar in bars:
                width = bar.get_width()
                plt.text(width + 0.1, bar.get_y() + bar.get_height()/2, f"{width:.1f}", 
                        ha='left', va='center')
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'client_rewards.png'), dpi=300)
            plt.close()
    
    def generate_model_evolution_report(self, eval_df):
        """Generate a report on model evolution over rounds."""
        if eval_df.empty or 'eval_loss' not in eval_df.columns or 'avg_accuracy' not in eval_df.columns:
            print("Insufficient data for model evolution report")
            return
        
        # Calculate changes between rounds
        eval_df['loss_change'] = eval_df['eval_loss'].diff()
        eval_df['accuracy_change'] = eval_df['avg_accuracy'].diff()
        
        # Plot loss and accuracy together
        fig, ax1 = plt.subplots(figsize=(14, 8))
        
        color1 = '#1f77b4'
        ax1.set_xlabel('Round', fontsize=14)
        ax1.set_ylabel('Loss', fontsize=14, color=color1)
        ax1.plot(eval_df['round'], eval_df['eval_loss'], 'o-', color=color1, linewidth=2, label='Loss')
        ax1.tick_params(axis='y', labelcolor=color1)
        
        ax2 = ax1.twinx()
        color2 = '#2ca02c'
        ax2.set_ylabel('Accuracy (%)', fontsize=14, color=color2)
        ax2.plot(eval_df['round'], eval_df['avg_accuracy'], 's-', color=color2, linewidth=2, label='Accuracy')
        ax2.tick_params(axis='y', labelcolor=color2)
        
        plt.title('Model Evolution: Loss and Accuracy', fontsize=18)
        
        # Add legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='best', fontsize=12)
        
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'model_evolution.png'), dpi=300)
        plt.close()
        
        # Create a summary table
        summary_data = []
        
        for i, row in eval_df.iterrows():
            round_num = row['round']
            loss = row['eval_loss']
            accuracy = row['avg_accuracy']
            loss_change = row['loss_change'] if not pd.isna(row['loss_change']) else None
            acc_change = row['accuracy_change'] if not pd.isna(row['accuracy_change']) else None
            
            # Determine if the model improved
            if i > 0:
                # Using accuracy as primary metric for improvement
                if acc_change > 0:
                    status = "Improved"
                elif acc_change == 0:
                    status = "Unchanged"
                else:
                    status = "Declined"
            else:
                status = "Baseline"
            
            summary_data.append({
                "round": round_num,
                "loss": loss,
                "accuracy": accuracy,
                "loss_change": loss_change,
                "accuracy_change": acc_change,
                "status": status
            })
        
        # Save summary to CSV
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(os.path.join(self.output_dir, 'model_evolution_summary.csv'), index=False)
        
        # Create a text report
        with open(os.path.join(self.output_dir, 'model_evolution_report.txt'), 'w') as f:
            f.write("=== Federated Learning Model Evolution Report ===\n\n")
            f.write(f"Total Rounds: {len(eval_df)}\n")
            
            if len(eval_df) > 1:
                initial_accuracy = eval_df.iloc[0]['avg_accuracy']
                final_accuracy = eval_df.iloc[-1]['avg_accuracy']
                
                f.write(f"Initial Accuracy: {initial_accuracy:.2f}%\n")
                f.write(f"Final Accuracy: {final_accuracy:.2f}%\n")
                f.write(f"Overall Improvement: {final_accuracy - initial_accuracy:.2f}%\n\n")
            
            f.write("Round by Round Summary:\n")
            for data in summary_data:
                f.write(f"Round {data['round']}: Loss={data['loss']:.4f}, Accuracy={data['accuracy']:.2f}%")
                
                if data['loss_change'] is not None and data['accuracy_change'] is not None:
                    f.write(f", Loss Change={data['loss_change']:.4f}, Accuracy Change={data['accuracy_change']:.2f}%")
                
                f.write(f" - {data['status']}\n")
    
    def generate_html_dashboard(self):
        """Generate an HTML dashboard with all visualizations."""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Federated Learning Dashboard</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .dashboard {{ max-width: 1200px; margin: 0 auto; }}
                .section {{ margin-bottom: 40px; }}
                .viz {{ margin-bottom: 30px; }}
                h1, h2 {{ color: #333; }}
                img {{ max-width: 100%; border: 1px solid #ddd; border-radius: 4px; }}
                .metrics {{ display: flex; flex-wrap: wrap; gap: 20px; }}
                .metric-card {{ 
                    background: #f8f9fa; 
                    border-radius: 8px; 
                    padding: 15px; 
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                    flex: 1;
                    min-width: 200px;
                }}
                .metric-value {{ font-size: 24px; font-weight: bold; margin: 10px 0; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                tr:nth-child(even) {{ background-color: #f9f9f9; }}
            </style>
        </head>
        <body>
            <div class="dashboard">
                <h1>Federated Learning Dashboard</h1>
                <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                
                <div class="section">
                    <h2>Training Metrics</h2>
                    <div class="metrics">
        """
        
        # Add summary metrics
        train_df, eval_df = self.parse_metrics_data()
        client_df = self.parse_client_stats()
        
        if not eval_df.empty and 'avg_accuracy' in eval_df.columns:
            final_accuracy = eval_df.iloc[-1]['avg_accuracy'] if len(eval_df) > 0 else "N/A"
            html_content += f"""
                        <div class="metric-card">
                            <div>Final Accuracy</div>
                            <div class="metric-value">{final_accuracy}%</div>
                        </div>
            """
        
        if not train_df.empty:
            total_rounds = max(train_df['round']) if len(train_df) > 0 else "N/A"
            html_content += f"""
                        <div class="metric-card">
                            <div>Total Rounds</div>
                            <div class="metric-value">{total_rounds}</div>
                        </div>
            """
        
        if not client_df.empty:
            total_clients = len(client_df['wallet_address'].unique())
            html_content += f"""
                        <div class="metric-card">
                            <div>Total Clients</div>
                            <div class="metric-value">{total_clients}</div>
                        </div>
            """
        
        html_content += """
                    </div>
                </div>
                
                <div class="section">
                    <h2>Model Performance</h2>
        """
        
        # Add visualizations if they exist
        viz_files = [
            ('model_evolution.png', 'Model Evolution'),
            ('evaluation_accuracy.png', 'Evaluation Accuracy'),
            ('training_loss.png', 'Training Loss')
        ]
        
        for filename, title in viz_files:
            if os.path.exists(os.path.join(self.output_dir, filename)):
                html_content += f"""
                    <div class="viz">
                        <h3>{title}</h3>
                        <img src="{filename}" alt="{title}">
                    </div>
                """
        
        html_content += """
                </div>
                
                <div class="section">
                    <h2>Client Participation</h2>
        """
        
        # Add client visualizations if they exist
        client_viz_files = [
            ('client_participation.png', 'Client Participation by Round'),
            ('client_accuracy_heatmap.png', 'Client Contribution Accuracy'),
            ('client_rewards.png', 'Client Rewards')
        ]
        
        for filename, title in client_viz_files:
            if os.path.exists(os.path.join(self.output_dir, filename)):
                html_content += f"""
                    <div class="viz">
                        <h3>{title}</h3>
                        <img src="{filename}" alt="{title}">
                    </div>
                """
        
        # Add summary table if available
        if os.path.exists(os.path.join(self.output_dir, 'model_evolution_summary.csv')):
            summary_df = pd.read_csv(os.path.join(self.output_dir, 'model_evolution_summary.csv'))
            
            html_content += """
                <div class="section">
                    <h2>Model Evolution Summary</h2>
                    <table>
                        <tr>
                            <th>Round</th>
                            <th>Loss</th>
                            <th>Accuracy</th>
                            <th>Loss Change</th>
                            <th>Accuracy Change</th>
                            <th>Status</th>
                        </tr>
            """
            
            for _, row in summary_df.iterrows():
                html_content += """
                        <tr>
                            <td>{row['round']}</td>
                            <td>{row['loss']:.4f}</td>
                            <td>{row['accuracy']:.2f}%</td>
                            <td>{row['loss_change']:.4f if not pd.isna(row['loss_change']) else 'N/A'}</td>
                            <td>{row['accuracy_change']:.2f}% if not pd.isna(row['accuracy_change']) else 'N/A'}</td>
                            <td>{row['status']}</td>
                        </tr>
                """
            
            html_content += """
                    </table>
                </div>
            """
        
        html_content += """
            </div>
        </body>
        </html>
        """
        
        # Save HTML file
        with open(os.path.join(self.output_dir, 'dashboard.html'), 'w') as f:
            f.write(html_content)
        
        print(f"Generated HTML dashboard at {os.path.join(self.output_dir, 'dashboard.html')}")
    
    def generate_visualizations(self):
        """Generate all visualizations."""
        # Parse data
        train_df, eval_df = self.parse_metrics_data()
        client_df = self.parse_client_stats()
        
        # Generate plots
        self.plot_training_loss(train_df, eval_df)
        self.plot_evaluation_accuracy(eval_df)
        self.plot_client_participation(train_df)
        self.plot_client_contributions(client_df)
        
        # Generate model evolution report
        self.generate_model_evolution_report(eval_df)
        
        # Generate HTML dashboard
        self.generate_html_dashboard()
        
        print(f"All visualizations saved to {self.output_dir}")


def main():
    """Main function to run the visualizer."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Visualize federated learning metrics")
    parser.add_argument("--metrics", type=str, default="metrics/metrics_history.json", 
                       help="Path to metrics history JSON file")
    parser.add_argument("--client-stats", type=str, default="metrics/client_stats.json",
                       help="Path to client stats JSON file")
    parser.add_argument("--output", type=str, default="visualizations",
                       help="Directory to save visualization outputs")
    parser.add_argument("--ipfs-url", type=str, default="http://127.0.0.1:5001",
                       help="IPFS API URL")
    parser.add_argument("--ganache-url", type=str, default="http://127.0.0.1:7545",
                       help="Ganache blockchain URL")
    parser.add_argument("--contract-address", type=str, default=None,
                       help="Federation contract address")
    
    args = parser.parse_args()
    
    visualizer = FederatedLearningVisualizer(
        metrics_path=args.metrics,
        client_stats_path=args.client_stats,
        output_dir=args.output,
        ipfs_url=args.ipfs_url,
        ganache_url=args.ganache_url,
        contract_address=args.contract_address
    )
    
    visualizer.generate_visualizations()


if __name__ == "__main__":
    main()
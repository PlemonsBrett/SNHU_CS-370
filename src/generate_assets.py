import matplotlib.pyplot as plt
import numpy as np
import os

# Constants for plot labels
TRAINING_EPOCH_LABEL = 'Training Epoch'

def _plot_training_performance(win_history):
    """Plot training performance metrics"""
    plt.figure(figsize=(15, 5))
    epochs = range(len(win_history))
    
    # Plot 1: Win rate
    plt.subplot(1, 3, 1)
    plt.plot(epochs, win_history, 'b-', linewidth=2)
    plt.title('Win Rate Over Training Epochs', fontsize=14, fontweight='bold')
    plt.xlabel(TRAINING_EPOCH_LABEL)
    plt.ylabel('Win Rate (%)')
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 105)
    
    # Plot 2: Moving average
    plt.subplot(1, 3, 2)
    window = min(50, len(win_history) // 4)
    if len(win_history) > window:
        moving_avg = np.convolve(win_history, np.ones(window)/window, mode='valid')
        plt.plot(range(window-1, len(win_history)), moving_avg, 'r-', linewidth=2)
        plt.title(f'Moving Average Win Rate\n(Window: {window} epochs)', fontsize=14, fontweight='bold')
        plt.xlabel(TRAINING_EPOCH_LABEL)
        plt.ylabel('Average Win Rate (%)')
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 105)
    
    # Plot 3: Epsilon values
    plt.subplot(1, 3, 3)
    epsilon_values = [0.05 if len(win_history) > 100 and np.mean(win_history[-100:]) > 90 else 0.1 
                      for _ in epochs]
    plt.plot(epochs, epsilon_values, 'g-', linewidth=2)
    plt.title('Exploration Rate (Epsilon)', fontsize=14, fontweight='bold')
    plt.xlabel(TRAINING_EPOCH_LABEL)
    plt.ylabel('Epsilon Value')
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 0.15)
    
    plt.tight_layout()
    plt.savefig('../assets/plots/training_performance.png', dpi=300, bbox_inches='tight')
    plt.show()

def _run_agent_path(model, qmaze):
    """Run agent and return path taken"""
    qmaze.reset((0, 0))
    path = [(0, 0)]
    envstate = qmaze.observe()
    step_count = 0
    max_steps = 100
    
    print("🎯 Running trained agent from start to treasure...")
    
    while step_count < max_steps:
        q_values = model.predict(envstate, verbose=0)
        action = np.argmax(q_values[0])
        envstate, _, game_status = qmaze.act(action)
        
        new_pirate_row, new_pirate_col, _ = qmaze.state
        path.append((new_pirate_row, new_pirate_col))
        step_count += 1
        
        if game_status in ['win', 'lose']:
            print(f"✅ Game completed with status: {game_status} in {step_count} steps")
            break
    
    return path

def _plot_solution_analysis(model, qmaze, path):
    """Plot solution analysis visualizations"""
    plt.figure(figsize=(12, 10))
    
    # Plot 1: Original maze
    plt.subplot(2, 2, 1)
    canvas = np.copy(qmaze._maze)
    plt.imshow(canvas, cmap='RdYlBu_r', interpolation='nearest')
    plt.title('Original Maze Layout', fontsize=14, fontweight='bold')
    plt.xlabel('Column')
    plt.ylabel('Row')
    plt.plot(0, 0, 'go', markersize=15, label='Start')
    plt.plot(7, 7, 'r*', markersize=20, label='Treasure')
    plt.legend()
    
    # Plot 2: Final state
    plt.subplot(2, 2, 2)
    qmaze.show()
    plt.title(f'Final State - Agent Path\n({len(path)} steps)', fontsize=14, fontweight='bold')
    
    # Plot 3: Path gradient
    plt.subplot(2, 2, 3)
    canvas = np.copy(qmaze._maze)
    for i, (row, col) in enumerate(path[:-1]):
        canvas[row, col] = 0.3 + (i / len(path)) * 0.4
    plt.imshow(canvas, cmap='viridis', interpolation='nearest')
    plt.title('Agent Path Visualization', fontsize=14, fontweight='bold')
    plt.xlabel('Column')
    plt.ylabel('Row')
    cbar = plt.colorbar()
    cbar.set_label('Step in Path')
    
    # Plot 4: Q-values
    plt.subplot(2, 2, 4)
    qmaze.reset((0, 0))
    start_state = qmaze.observe()
    q_values = model.predict(start_state, verbose=0)[0]
    
    actions = ['Left', 'Up', 'Right', 'Down']
    colors = ['red', 'blue', 'green', 'orange']
    bars = plt.bar(actions, q_values, color=colors, alpha=0.7)
    plt.title('Q-Values at Starting Position', fontsize=14, fontweight='bold')
    plt.ylabel('Q-Value')
    plt.xticks(rotation=45)
    
    for bar, value in zip(bars, q_values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('../assets/plots/solution_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def _calculate_convergence_epoch(win_history):
    """Calculate epoch when model converged"""
    for i in range(50, len(win_history)):
        if np.mean(win_history[max(0, i-50):i]) > 90:
            return i
    return None

def generate_portfolio_assets(model, qmaze, win_history):
    """Generate visualizations for portfolio presentation"""
    
    # Create assets directory if it doesn't exist
    os.makedirs('../assets/plots', exist_ok=True)
    
    # Generate plots
    _plot_training_performance(win_history)
    path = _run_agent_path(model, qmaze)
    _plot_solution_analysis(model, qmaze, path)
    
    # Print performance summary
    convergence_epoch = _calculate_convergence_epoch(win_history)
    
    print("\n" + "="*60)
    print("🏆 TRAINING SUMMARY")
    print("="*60)
    print(f"📊 Total Training Epochs: {len(win_history)}")
    print(f"🎯 Final Win Rate: {win_history[-1]:.1f}%")
    print(f"📈 Best Win Rate: {max(win_history):.1f}%")
    print(f"⚡ Steps to Solution: {len(path)}")
    print(f"🧠 Network Parameters: {model.count_params()}")
    
    if len(win_history) >= 100:
        recent_avg = np.mean(win_history[-100:])
        print(f"📋 Recent Performance (last 100 epochs): {recent_avg:.1f}%")
    
    if convergence_epoch:
        print(f"🚀 Convergence Achieved at Epoch: {convergence_epoch}")
    
    print("="*60)
    
    return {
        'total_epochs': len(win_history),
        'final_win_rate': win_history[-1],
        'best_win_rate': max(win_history),
        'solution_steps': len(path),
        'convergence_epoch': convergence_epoch,
        'model_parameters': model.count_params()
    }

# Call this function at the end of your training
# Example usage:
# stats = generate_portfolio_assets(model, qmaze, win_history)
# print(f"Training completed with {stats['total_epochs']} epochs")
# Agent Vision and Behavior Analysis

## 1. Agent Interaction Experiment
![Agent Interaction](holding_one_agent_still.gif)

In this experiment, we demonstrate:
- One agent is constrained to remain stationary
- The other agent has full mobility and is programmed to eliminate the stationary agent
- This setup helps us understand emergent behaviors in competitive scenarios

## 2. Vision Encoding Analysis
![Vision Activation Patterns](tank_vision_activations.gif)

### Key Observations:
- The visualization shows neural activation patterns in the agents' vision encoders
- Brighter regions (white) indicate areas of high importance in the visual field
- Darker regions represent areas of lower significance in the neural processing

### Activation Disparity:
- **Agent 1**: Shows clear, distinct activation patterns, indicating effective visual processing
- **Agent 2**: Displays minimal activation, suggesting impaired or ineffective visual processing
- This disparity highlights a significant performance gap between the two agents

## 3. Vision System Improvement

### Before Optimization
![Initial Vision State](before_vision_improvement.gif)

**Initial State Analysis:**
- First agent demonstrates proper visual processing capabilities
- Second agent shows significantly degraded vision performance
- Clear performance gap between agents' visual systems

### After Optimization
![Improved Vision State](after_vision_improvement.gif)

**Improvement Process:**
1. Copied successful vision architecture from Agent 1 to Agent 2
2. Conducted targeted policy optimization over multiple epochs
3. Achieved improved visual processing capabilities in Agent 2

## Technical Details
- Vision encoder activations are visualized through heat maps
- White regions indicate high neural activation
- Dark regions indicate low or no activation
- Policy optimization was focused on vision system parameters
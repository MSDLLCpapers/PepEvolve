from dataclasses import dataclass, field

@dataclass
class LearningConfig:
    agent_baseline_weight: float = 0.99  
    batch_size: int = 50
    learning_rate: float = 0.0001
    score_multiplier: int = 20
    distance_threshold: int = -20

    # Routing period parameters
    routing_steps: int = 500 
    router_baseline_weight: float = 0.99   
    router_entropy_frac: float = 0.6       
    router_entropy_start: float = 0.1      
    router_entropy_end: float = 0.0        
    router_learning_rate: float = 5e-2
    
    num_positions: int = 2
    num_samples: int = 4
    num_gen_per_sample: int = 32

    # Evolving period parameters
    evolving_steps: int = 50
    evolve_mode: str = 'single'
    agent_context_visibility: str = 'self_mask'
    K: int = 16 
    G: int = 8
    num_original: int = 0
    positions: list = field(default_factory=list)

    # PepINVENT
    number_steps: int = 200

    # Ablation study parameters
    use_group_relative_advantage: bool = True